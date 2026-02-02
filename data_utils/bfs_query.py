import asyncio
import time
from collections import deque
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from tqdm import tqdm

wikidata_api_url = "https://www.wikidata.org/w/api.php"

MAX_EDGES_PER_NODE = 10
MAX_PATHS = 5
BFS_TIMEOUT = 30.0
REQUEST_TIMEOUT = 800.0


async def _http_get_json(
    session: aiohttp.ClientSession,
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
) -> Any:
    """Perform a GET and return JSON."""
    headers = {"User-Agent": "KG-RAG/1.0 (https://github.com/KG-RAG4EM/rag-em) aiohttp/3.8.0"}
    max_retries = 3
    backoff_base = 0.5
    for attempt in range(1, max_retries + 1):
        try:
            async with session.get(url, params=params, headers=headers) as response:
                response.raise_for_status()
                return await response.json()
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            print(f"HTTP request failed (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                await asyncio.sleep(backoff_base * (2 ** (attempt - 1)))
            else:
                print(f"All HTTP requests failed for URL: {url}")
                return None


async def query_entity_claims_async(
    session: aiohttp.ClientSession,
    entity_id: str,
) -> Optional[Dict[str, Any]]:
    """Query Wikidata API for entity claims with robot policy compliance."""
    params = {
        "action": "wbgetentities",
        "format": "json",
        "ids": entity_id,
        "props": "claims",
        "languages": "en",
    }
    data = await _http_get_json(session, wikidata_api_url, params=params)
    if data and "entities" in data:
        entity_data = data["entities"].get(entity_id)
        if entity_data and not entity_data.get("missing"):
            return entity_data
        else:
            print(f"Entity {entity_id} not found in Wikidata")
    else:
        print(f"No response data for entity {entity_id} from API")
    


def extract_edges_from_claims(entity_data: Optional[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """Extract edges from entity claims for BFS traversal."""
    if not entity_data or "claims" not in entity_data:
        return []

    edges = []
    edge_count = 0
    for prop, claims in entity_data["claims"].items():
        if edge_count >= MAX_EDGES_PER_NODE:
            break
        for claim in claims[:5]:
            if edge_count >= MAX_EDGES_PER_NODE:
                break
            mainsnak = claim.get("mainsnak", {})
            datavalue = mainsnak.get("datavalue", {})
            if datavalue.get("type") == "wikibase-entityid":
                target_id = datavalue.get("value", {}).get("id", "")
                if target_id:
                    edges.append((target_id, prop))
                    edge_count += 1
    return edges


async def bfs_async(
    session: aiohttp.ClientSession,
    start_entity: str,
    end_entity: Optional[str],
    max_depth: int,
) -> List[List[Tuple[str, str, str]]]:
    """BFS implementation with async API calls."""
    paths = []
    graph = {}
    visited = {start_entity}
    queue = deque([(start_entity, [], time.time())])
    start_time = time.time()

    while queue and len(paths) < MAX_PATHS:
        if time.time() - start_time > BFS_TIMEOUT:
            break

        current_entity, path, entry_time = queue.popleft()

        if time.time() - entry_time > BFS_TIMEOUT / 2:
            continue

        if end_entity and current_entity == end_entity and path:
            paths.append(path)
            continue

        if len(path) >= max_depth:
            if not end_entity:
                paths.append(path)
            continue

        if current_entity not in graph:
            entity_data = await query_entity_claims_async(session, current_entity)
            edges = extract_edges_from_claims(entity_data) if entity_data else []
            graph[current_entity] = edges

        for target_id, prop in graph[current_entity][:MAX_EDGES_PER_NODE]:
            new_path = path + [(current_entity, prop, target_id)]

            if end_entity and target_id == end_entity:
                paths.append(new_path)
                continue

            if target_id not in visited:
                visited.add(target_id)
                queue.append((target_id, new_path, time.time()))

                if not end_entity and len(new_path) == max_depth:
                    paths.append(new_path)

    return paths


async def bfs_search_async(
    entities: List[str],
    max_depth: int,
    *,
    max_concurrency: int = 10,
    request_timeout_seconds: float = REQUEST_TIMEOUT,
) -> Dict[str, Any]:
    """Perform BFS search on entities (single or pairs) asynchronously."""
    timeout = aiohttp.ClientTimeout(total=request_timeout_seconds)
    connector = aiohttp.TCPConnector(limit_per_host=max_concurrency)

    bfs_results = {
        "single_entity_paths": {},
        "entity_pair_paths": {},
        "stats": {"single": 0, "pairs": 0, "success": 0, "error": 0},
    }

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        if len(entities) == 1:
            entity = entities[0]
            bfs_results["stats"]["single"] += 1
            try:
                paths = await bfs_async(session, entity, None, max_depth)
                # Keep paths structure for multi-hop representation
                if paths:
                    bfs_results["single_entity_paths"][entity] = paths
                    bfs_results["stats"]["success"] += 1
            except Exception:
                bfs_results["stats"]["error"] += 1

        elif len(entities) > 1:
            entity_pairs = list(combinations(entities, 2))
            bfs_results["stats"]["pairs"] = len(entity_pairs)
            print(f"Exploring {len(entity_pairs)} entity pairs...")

            with tqdm(total=len(entity_pairs), desc="BFS search") as pbar:
                for start_entity, end_entity in entity_pairs:
                    try:
                        paths = await bfs_async(session, start_entity, end_entity, max_depth)

                        if paths:
                            pair_key = f"{start_entity}->{end_entity}"
                            bfs_results["entity_pair_paths"][pair_key] = paths
                            bfs_results["stats"]["success"] += 1
                        else:
                            paths_reverse = await bfs_async(session, end_entity, start_entity, max_depth)

                            if paths_reverse:
                                pair_key = f"{end_entity}->{start_entity}"
                                bfs_results["entity_pair_paths"][pair_key] = paths_reverse
                                bfs_results["stats"]["success"] += 1
                            else:
                                start_paths = await bfs_async(session, start_entity, None, 2)
                                end_paths = await bfs_async(session, end_entity, None, 2)
                                combined_paths = (start_paths[:1] if start_paths else []) + (end_paths[:1] if end_paths else [])

                                if combined_paths:
                                    pair_key = f"{start_entity}<->{end_entity}"
                                    bfs_results["entity_pair_paths"][pair_key] = combined_paths
                                    bfs_results["stats"]["success"] += 1
                                    

                    except Exception as e:
                        print(f"Error exploring pair {start_entity}-{end_entity}: {e}")
                        bfs_results["stats"]["error"] += 1

                        pbar.update(1)
                    

    return bfs_results


def bfs_search_with_entity(
    entities: List[str],
    max_depth: int,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Perform BFS search on entities (single or pairs).

    Args:
        entities (list[str]): List of entity IDs (QIDs).
        max_depth (int): Maximum depth for BFS search.

    Returns:
        dict: BFS search results containing single_entity_paths, entity_pair_paths, and stats.

    Example:
        >>> entities = ["Q42"]
        >>> bfs_search_with_entity(entities, max_depth=2)
    """
    return asyncio.run(bfs_search_async(entities, max_depth, **kwargs))
