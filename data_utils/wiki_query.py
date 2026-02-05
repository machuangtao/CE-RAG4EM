import asyncio
import json
import os
import collections
from typing import Any, Dict, List, Optional, Tuple, Literal, Deque

import pandas as pd
from dotenv import load_dotenv
import aiohttp
import aiofiles

load_dotenv()

wikidata_vector_database_url = "https://wd-vectordb.wmcloud.org"
wikidata_api_url = "https://www.wikidata.org/w/api.php"

class RateLimiter:
    def __init__(self, max_calls: int, period: float):
        """
        max_calls: how many calls are allowed
        period: time window in seconds
        """
        self.max_calls = max_calls
        self.period = period
        self._calls: Deque[float] = collections.deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """
        Wait until making another call is allowed.
        """
        while True:
            async with self._lock:
                now = asyncio.get_running_loop().time()

                # Drop timestamps older than `period`
                while self._calls and self._calls[0] <= now - self.period:
                    self._calls.popleft()

                if len(self._calls) < self.max_calls:
                    # We’re allowed to proceed
                    self._calls.append(now)
                    return

                # Need to wait until the earliest call falls out of the window
                earliest = self._calls[0]
                sleep_for = self.period - (now - earliest)

            # Sleep *outside* the lock
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)

# global rate limiter for vectorDB: 8 requests per 60 seconds -> maximal 10 per 60 seconds
wikidb_rate_limiter = RateLimiter(max_calls=9, period=60.0)


def _iter_dataframe_rows(df: pd.DataFrame, id_column: str, query_column: str) -> List[Tuple[str, str]]:
    return [(str(row[id_column]), str(row[query_column])) for _, row in df.iterrows()]


async def _http_get_json(
        session: aiohttp.ClientSession,
        url: str,
        *,
        params: Optional[Dict[str, Any]] = None,
) -> Any:
    """Perform a GET and return JSON."""
    max_retries = 3
    backoff_base = 0.5
    headers = {"User-Agent": "CE-RAG/1.0 (https://github.com/CE-RAG4EM) aiohttp/3.8.0"}
    for attempt in range(1, max_retries + 1):
        try:
            async with session.get(url, params=params, headers=headers) as response:
                response.raise_for_status()
                return await response.json()
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as e:
            # Log what went wrong
            if isinstance(e, asyncio.TimeoutError):
                print(
                    f"[_http_get_json] Timeout on attempt {attempt}/{max_retries} "
                    f"for {url} with params={params}", flush=True,
                )
            elif isinstance(e, aiohttp.ClientResponseError):
                # This one has extra info (status, etc.)
                print(
                    f"[_http_get_json] HTTP {e.status} error on attempt "
                    f"{attempt}/{max_retries} for {url} with params={params}: {e.message}", flush=True,
                )
            else:
                print(
                    f"[_http_get_json] Error on attempt {attempt}/{max_retries} "
                    f"for {url} with params={params}: {repr(e)}",
                    flush=True,
                )

            if attempt < max_retries:
                wait_time = backoff_base * (2 ** (attempt - 1))
                print(f"[_http_get_json] Retrying in {wait_time:.2f}s...", flush=True,)
                await asyncio.sleep(wait_time)
            else:
                print(
                    f"[_http_get_json] Giving up after {max_retries} attempts for {url} "
                    f"with params={params}", flush=True,
                )
                return None


async def query_ids_from_wiki_db(
        session: aiohttp.ClientSession,
        query_text: str,
        query_type: Literal["item", "property"],
):
    params = {"query": query_text, "K": 10}
    url = f"{wikidata_vector_database_url}/{query_type}/query"

    await wikidb_rate_limiter.acquire()

    return await _http_get_json(session, url, params=params)


async def relevant_ids_async(
        query_df: pd.DataFrame,
        id_column: str,
        query_column: str,
        output_path: str,
        *,
        max_concurrency: int = 10,
        request_timeout_seconds: float = 90.0,
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """
    Fetch and save relevant QIDs and PIDs for each query row concurrently.
    """
    timeout = aiohttp.ClientTimeout(total=request_timeout_seconds)
    connector = aiohttp.TCPConnector(limit_per_host=max_concurrency)
    semaphore = asyncio.Semaphore(max_concurrency)

    rows = _iter_dataframe_rows(query_df, id_column, query_column)
    relevant_items_by_query_id: Dict[str, List[Dict[str, Any]]] = {}
    relevant_properties_by_query_id: Dict[str, List[Dict[str, Any]]] = {}

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        async def process_one(_query_id: str, _query_text: str) -> Tuple[
            str, List[Dict[str, Any]], List[Dict[str, Any]]]:
            async with semaphore:
                # run both lookups in parallel for this row
                item_task = asyncio.create_task(query_ids_from_wiki_db(session, _query_text, "item"))
                property_task = asyncio.create_task(query_ids_from_wiki_db(session, _query_text, "property"))
                _relevant_items, _relevant_properties = await asyncio.gather(item_task, property_task)
                return _query_id, _relevant_items, _relevant_properties

        print(f"[relevant_ids_async] Start fetching relevant ids from wiki vector db...", flush=True,)
        completed = 0
        tasks = [process_one(qid, text) for qid, text in rows]
        for coro in asyncio.as_completed(tasks):
            query_id, relevant_items, relevant_properties = await coro
            relevant_items_by_query_id[query_id] = relevant_items
            relevant_properties_by_query_id[query_id] = relevant_properties
            completed += 1
            print(f"[relevant_ids_async] Finished {completed} / {len(tasks)}", flush=True,)

    results = {
        "relevant_qids": relevant_items_by_query_id,
        "relevant_pids": relevant_properties_by_query_id,
    }

    async with aiofiles.open(output_path, "w", encoding="utf-8") as f:
        await f.write(json.dumps(results, indent=2))
    print(f"[relevant_ids_async] Completed fetching relevant ids from wiki vector db. Results saved to {output_path}", flush=True,)
    print("[relevant_ids_async] Sleeping 10 seconds to cool down.", flush=True)
    await asyncio.sleep(10)
    return results


async def label_and_description_for_ids_async(
        session: aiohttp.ClientSession,
        ids: List[str],
        language: str = "en",
) -> Dict[str, Dict[str, str]]:
    """
    Retrieve labels and descriptions for Wikidata entities (QIDs/PIDs).
    """
    params = {
        "action": "wbgetentities",
        "format": "json",
        "ids": "|".join(ids),
        "props": "labels|descriptions",
        "languages": language,
        "origin": "*",
    }

    data = await _http_get_json(session, wikidata_api_url, params=params)
    entities = data.get("entities", {}) or {}
    if not entities:
        print(
            f"[label_and_description_for_ids_async] The query to extract labels and descriptions for {ids} failed "
            f"after three attempts", flush=True, )

    results: Dict[str, Dict[str, str]] = {}
    for id_ in ids:
        entity = entities.get(id_, {}) or {}
        labels = entity.get("labels", {}) or {}
        descriptions = entity.get("descriptions", {}) or {}

        label = (labels.get(language) or {}).get("value", "")  # default to ""
        description = (descriptions.get(language) or {}).get("value", "")  # default to ""

        results[id_] = {
            "label": label,
            "description": description,
        }

    return results


async def labels_and_descriptions_async(
        ids: List[str],
        output_path: str,
        language: str = "en",
        *,
        chunk_size: int = 10,
        max_concurrency: int = 10,
        request_timeout_seconds: float = 30.0,
) -> None:
    """
    Fetch Wikidata labels/descriptions in chunks concurrently and save as JSON.
    """
    unique_ids = list(set(ids))

    timeout = aiohttp.ClientTimeout(total=request_timeout_seconds)
    connector = aiohttp.TCPConnector(limit_per_host=max_concurrency)
    semaphore = asyncio.Semaphore(max_concurrency)

    # Prepare chunks of ids (e.g., 10 per request)
    chunks: List[List[str]] = [
        unique_ids[i: i + chunk_size] for i in range(0, len(unique_ids), chunk_size)
    ]

    all_results: Dict[str, Dict[str, str]] = {}

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        async def process_chunk(id_chunk: List[str]) -> Dict[str, Dict[str, str]]:
            async with semaphore:
                return await label_and_description_for_ids_async(session, id_chunk, language)

        print("[labels_and_descriptions_async] Start fetching labels and descriptions...", flush=True,)
        completed = 0
        tasks = [process_chunk(chunk) for chunk in chunks]
        for coro in asyncio.as_completed(tasks):
            chunk_results = await coro
            all_results.update(chunk_results)
            completed += 1
            print(f"[labels_and_descriptions_async] Finished {completed}/{len(tasks)}.", flush=True,)

    async with aiofiles.open(output_path, "w", encoding="utf-8") as f:
        await f.write(json.dumps(all_results, indent=2, ensure_ascii=False))
    print(f"[labels_and_descriptions_async] Completed fetching labels and descriptions. Results saved to {output_path}", flush=True,)
    print("[labels_and_descriptions_async] Sleeping 10 seconds to cool down.", flush=True)
    await asyncio.sleep(10)


async def triplet_by_id_async(
        session: aiohttp.ClientSession,
        given_id: str,
        language: str = "en",
        relevant_properties: Optional[set] = None,
        only_pretty_string: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Query Wikidata and get triples for a single QID or PID (limited to selected properties).
    """
    if not relevant_properties:
        relevant_properties = {"P31", "P279", "P361", "P366"}
    params = {"action": "wbgetentities", "format": "json", "ids": given_id, "props": "claims",
              "languages": language, "origin": "*", }
    data = await _http_get_json(session, wikidata_api_url, params=params)

    entities = data.get("entities", {})
    if given_id not in entities:
        print(f"[triplet_by_id_async] The query to extract triplets from data of {given_id} failed after "
              f"three attempts", flush=True,)
        return {}

    claims = entities[given_id].get("claims", {})
    triples: List[Tuple[str, str, str]] = []
    all_ids: set = {given_id}

    for prop, values in claims.items():
        if prop not in relevant_properties:
            continue

        for value in values:
            mainsnak = (value or {}).get("mainsnak", {})
            datavalue = mainsnak.get("datavalue", {})
            if not datavalue:
                continue

            if datavalue.get("type") == "wikibase-entityid":
                obj = (datavalue.get("value") or {}).get("id", "")
            elif datavalue.get("type") == "string":
                obj = datavalue.get("value", "")
            else:
                continue

            triples.append((given_id, prop, obj))
            all_ids.add(prop)
            if isinstance(obj, str) and obj.startswith("Q"):
                all_ids.add(obj)

    if not triples:
        return {}

    # Enrich with labels/descriptions
    entity_info = await label_and_description_for_ids_async(session, list(all_ids), language)

    detailed_triples: List[Dict[str, Any]] = []
    for subj, pred, obj in triples:
        subj_info = entity_info.get(subj, {})
        pred_info = entity_info.get(pred, {})
        obj_is_qid = isinstance(obj, str) and obj.startswith("Q")
        obj_info = entity_info.get(obj, {}) if obj_is_qid else {}

        pretty_string = (f"({subj_info.get('label', subj)}, {pred_info.get('label', pred)}, "
                         f"{obj_info.get('label', obj) if obj_is_qid else obj})")

        if only_pretty_string:
            detailed_triples.append({"pretty_string": pretty_string, })
        else:
            detailed_triples.append(
                {
                    "subject": {
                        "id": subj,
                        "label": subj_info.get("label", subj),
                        "description": subj_info.get("description", ""),
                    },
                    "predicate": {
                        "id": pred,
                        "label": pred_info.get("label", pred),
                        "description": pred_info.get("description", ""),
                    },
                    "object": {
                        "id": obj,
                        "label": obj_info.get("label", obj) if obj_is_qid else obj,
                        "description": obj_info.get("description", "") if obj_is_qid else "",
                    },
                    "pretty_string": pretty_string,
                }
            )

    return {given_id: detailed_triples}


async def triplets_async(
        ids: List[str],
        output_path: str,
        language: str = "en",
        *,
        max_concurrency: int = 10,
        request_timeout_seconds: float = 900.0,
        only_pretty_string: bool = True,
) -> None:
    """
    Fetch Wikidata triples concurrently with retries.
    Results are saved incrementally to reduce data loss risk.
    """
    results: Dict[str, Any] = {}
    ids = list(set(ids))

    timeout = aiohttp.ClientTimeout(total=request_timeout_seconds)
    connector = aiohttp.TCPConnector(limit_per_host=max_concurrency)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        print(f"[triplets_async] Start fetching triplets...", flush=True,)
        tasks = [
            asyncio.create_task(triplet_by_id_async(session, id_, language=language))
            for id_ in ids
        ]
        completed, total = 0, len(tasks)
        for task in asyncio.as_completed(tasks):
            try:
                result_for_one = await task
            except Exception as e:
                completed += 1
                print(f"[triplets_async] [{completed}/{total}] Error fetching triplets for one id: {e!r}", flush=True,)
                continue

            if not result_for_one:
                completed += 1
                print(f"[triplets_async] [{completed}/{total}] Empty result for one id.", flush=True,)
                continue

            # result_for_one is expected to be {id_: triples}
            id_, triples = next(iter(result_for_one.items()))
            results[id_] = triples
            completed += 1
            print(
                f"[triplets_async] [{completed}/{total}] Finished id {id_} "
                f"with {len(triples)} triples.", flush=True,
            )

    async with aiofiles.open(output_path, "w", encoding="utf-8") as f:
        await f.write(json.dumps(results, indent=2, ensure_ascii=False))

    print(f"[triplets_async] Completed fetching triplets. Results saved to {output_path}", flush=True,)
    print("[triplets_async] Sleeping 10 seconds to cool down.", flush=True)
    await asyncio.sleep(10)


def fetch_and_save_relevant_ids(
        query_df: pd.DataFrame,
        id_column: str,
        query_column: str,
        output_path: str,
        **kwargs: Any,
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Fetch and save relevant Wikidata IDs for given queries, with async support.

    Args:
        query_df (pd.DataFrame): DataFrame containing query information.
        id_column (str): Name of column containing query IDs.
        query_column (str): Name of column containing query texts.
        output_path (str): File path where JSON results will be saved.

    Returns:
        dict: A dictionary containing the relevant QIDs and PIDs, structured as:
            {
                "relevant_qids": {query_id: [list of relevant QIDs], ...},
                "relevant_pids": {query_id: [list of relevant PIDs], ...}
            }

    Example:
        >>> df = pd.DataFrame({
        ...     'id': ['q1', 'q2'],
        ...     'query': ['science', 'art']
        ... })
        >>> fetch_and_save_relevant_ids(df, 'id', 'query', 'results.json')
    """
    return asyncio.run(
        relevant_ids_async(
            query_df, id_column, query_column, output_path, **kwargs
        )
    )


def fetch_and_save_wikidata(
        ids: List[str],
        output_path: str,
        language: str = "en",
        **kwargs: Any,
) -> None:
    """Fetch and save label and description from wikidata for multiple ids, with async support.

    Args:
        ids (list[str]): List of Wikidata QIDs or PIDs (e.g., ["Q123", "Q456"]).
        output_path (str): Path where the JSON results will be saved.
        lang (str, optional): Language code for labels/descriptions. Defaults to 'en'.

    Returns:
        None

    Example:
        >>> ids = ["Q123", "P31", "Q456"]
        >>> fetch_and_save_wikidata(ids, "wikidata_info.json", "en")
    """
    return asyncio.run(
        labels_and_descriptions_async(ids, output_path, language, **kwargs)
    )


def fetch_and_save_triplets(
        ids: List[str],
        output_path: str,
        language: str = "en",
        only_pretty_string: bool = True,
        **kwargs: Any,
) -> None:
    """Fetch and save Wikidata triplets with async support.

    Args:
        ids (list[str]): List of Wikidata entity IDs to fetch triplets for.
        output_path (str): Path where the JSON results will be saved.
        lang (str, optional): Language code for labels/descriptions. Defaults to 'en'.

    Returns:
        None

    Example:
        >>> ids = ["Q123", "Q456", "Q789"]
        >>> fetch_and_save_triplets(ids, "triplets.json", "en")
    """
    return asyncio.run(
        triplets_async(ids, output_path, language, **kwargs)
    )