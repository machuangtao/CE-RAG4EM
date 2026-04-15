import json
import os
import time
from urllib import request as urlrequest
from typing import Any, Dict, Iterable, List, Optional
from functools import lru_cache

import pandas as pd
from dotenv import load_dotenv



DEFAULT_KG20C_ENTITY_INFO_PATH = "kg/KG20C/all_entity_info.txt"
DEFAULT_KG20C_TRIPLES_PATH = "kg/KG20C/kg20c_triples.jsonl"
DEFAULT_ASTRA_COLLECTION = "kg20c_entities_v1"


class JinaEmbeddingsV3:

    def __init__(
        self,
        model_name: str = "jinaai/jina-embeddings-v3",
        embedding_dim: int = 512,
        batch_size: int = 64,
        normalize: bool = True,
    ) -> None:
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.normalize = normalize
        self.model = self._load_model()

    def _load_model(self) -> Any:
        load_errors: List[str] = []

        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            return SentenceTransformer(self.model_name, trust_remote_code=True)
        except Exception as e:  # pragma: no cover
            load_errors.append(f"sentence-transformers failed: {e!r}")

        try:
            from transformers import AutoModel  # type: ignore

            return AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
        except Exception as e:  # pragma: no cover
            load_errors.append(f"transformers failed: {e!r}")

        raise RuntimeError(
            "Unable to load jina embeddings model. Install one of:\n"
            "  pip install sentence-transformers\n"
            "or\n"
            "  pip install transformers torch\n"
            f"Details: {' | '.join(load_errors)}"
        )

    def _ensure_dim(self, vector: Iterable[float]) -> List[float]:
        v = [float(x) for x in vector]
        if len(v) < self.embedding_dim:
            raise ValueError(
                f"Embedding dim mismatch: got {len(v)} < requested {self.embedding_dim}."
            )
        return v[: self.embedding_dim]

    def _encode_with_fallback(self, texts: List[str], *, is_query: bool) -> List[List[float]]:
        # Jina v3 supports task/prompt kwargs in trust_remote_code encode methods.
        task = "retrieval.query" if is_query else "retrieval.passage"
        prompt_name = "query" if is_query else "passage"

        encode_variants: List[Dict[str, Any]] = [
            {
                "task": task,
                "prompt_name": prompt_name,
                "batch_size": self.batch_size,
                "normalize_embeddings": self.normalize,
                "truncate_dim": self.embedding_dim,
            },
            {
                "task": task,
                "batch_size": self.batch_size,
                "normalize_embeddings": self.normalize,
                "truncate_dim": self.embedding_dim,
            },
            {
                "batch_size": self.batch_size,
                "normalize_embeddings": self.normalize,
                "truncate_dim": self.embedding_dim,
            },
            {
                "batch_size": self.batch_size,
                "normalize_embeddings": self.normalize,
            },
        ]

        last_error: Optional[Exception] = None
        for kwargs in encode_variants:
            try:
                vectors = self.model.encode(texts, **kwargs)
                return [self._ensure_dim(v) for v in vectors]
            except TypeError as e:
                last_error = e
                continue
            except Exception as e:
                last_error = e
                continue

        raise RuntimeError(f"Embedding encode failed for all fallbacks: {last_error!r}")

    def encode_documents(self, texts: List[str]) -> List[List[float]]:
        return self._encode_with_fallback(texts, is_query=False)

    def encode_query(self, text: str) -> List[float]:
        return self._encode_with_fallback([text], is_query=True)[0]


class AstraKG20CVectorDB:

    def __init__(
        self,
        collection_name: str = DEFAULT_ASTRA_COLLECTION,
        embedding_dim: int = 512,
        metric: str = "cosine",
        recreate_if_non_vector: bool = False,
        validate_vector_search: bool = True,
    ) -> None:
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.metric = metric
        self.recreate_if_non_vector = recreate_if_non_vector
        self.validate_vector_search = validate_vector_search

        token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
        api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
        keyspace = os.getenv("ASTRA_DB_KEYSPACE")
        self.token = token
        self.api_endpoint = api_endpoint
        self.keyspace = keyspace or "default_keyspace"

        if not token or not api_endpoint:
            raise ValueError(
                "Missing Astra credentials. Set ASTRA_DB_APPLICATION_TOKEN and ASTRA_DB_API_ENDPOINT."
            )

        try:
            from astrapy import DataAPIClient  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "astrapy is required. Install with: pip install astrapy"
            ) from e

        client = DataAPIClient(token)
        if keyspace:
            self.db = client.get_database(api_endpoint=api_endpoint, keyspace=keyspace)
        else:
            self.db = client.get_database(api_endpoint=api_endpoint)

        self.collection = self._ensure_collection()
        if self.validate_vector_search:
            self._ensure_vector_search_enabled()

    def _create_collection_compatible(self) -> None:
        errors: List[str] = []

        create_calls = [
            lambda: self.db.create_collection(self.collection_name, self.embedding_dim),
            lambda: self.db.create_collection(
                self.collection_name, self.embedding_dim, self.metric
            ),
            lambda: self.db.create_collection(
                self.collection_name,
                {"vector": {"dimension": self.embedding_dim, "metric": self.metric}},
            ),
            lambda: self.db.create_collection(
                self.collection_name,
                {"dimension": self.embedding_dim, "metric": self.metric},
            ),
            lambda: self.db.create_collection(
                self.collection_name,
                dimension=self.embedding_dim,
                metric=self.metric,
            ),
            lambda: self.db.create_collection(
                name=self.collection_name,
                dimension=self.embedding_dim,
                metric=self.metric,
            ),
            lambda: self.db.create_collection(
                self.collection_name,
                vector_dimension=self.embedding_dim,
                metric=self.metric,
            ),
            lambda: self.db.create_collection(
                name=self.collection_name,
                vector_dimension=self.embedding_dim,
                metric=self.metric,
            ),
            lambda: self.db.create_collection(
                self.collection_name,
                options={
                    "vector": {"dimension": self.embedding_dim, "metric": self.metric}
                },
            ),
            lambda: self.db.create_collection(
                name=self.collection_name,
                options={
                    "vector": {"dimension": self.embedding_dim, "metric": self.metric}
                },
            ),
        ]

        try:
            from astrapy.info import CollectionOptions, CollectionVectorOptions  # type: ignore

            create_calls.extend(
                [
                    lambda: self.db.create_collection(
                        self.collection_name,
                        options=CollectionOptions(
                            vector=CollectionVectorOptions(
                                dimension=self.embedding_dim,
                                metric=self.metric,
                            )
                        ),
                    ),
                    lambda: self.db.create_collection(
                        name=self.collection_name,
                        options=CollectionOptions(
                            vector=CollectionVectorOptions(
                                dimension=self.embedding_dim,
                                metric=self.metric,
                            )
                        ),
                    ),
                ]
            )
        except Exception:
            pass

        if hasattr(self.db, "command"):
            create_calls.append(self._create_collection_via_command)
        # Raw HTTP fallback bypassing SDK compatibility issues.
        create_calls.append(self._create_collection_via_http)

        for call in create_calls:
            try:
                call()
                return
            except Exception as e:
                errors.append(f"{type(e).__name__}: {e}")

        raise RuntimeError(
            "Failed to create vector-enabled Astra collection with compatible signatures. "
            f"Tried {len(create_calls)} variants. Errors: {' | '.join(errors)}"
        )

    def _create_collection_via_command(self) -> None:
        """
        Create vector collection via Data API keyspace command.
        Retries transient timeouts.
        """
        last_error: Optional[Exception] = None
        payload = {
            "createCollection": {
                "name": self.collection_name,
                "options": {
                    "vector": {
                        "dimension": self.embedding_dim,
                        "metric": self.metric,
                    }
                },
            }
        }
        for attempt in range(1, 4):
            try:
                self.db.command(payload)
                return
            except Exception as e:
                last_error = e
                # Retry only likely transient timeout/network issues
                msg = str(e).lower()
                if "timeout" in msg and attempt < 4:
                    sleep_for = 1.5 * attempt
                    print(
                        f"[AstraKG20CVectorDB] createCollection timeout, retrying in {sleep_for:.1f}s "
                        f"(attempt {attempt}/3)...",
                        flush=True,
                    )
                    time.sleep(sleep_for)
                    continue
                raise
        if last_error:
            raise last_error

    def _create_collection_via_http(self) -> None:
        if not self.api_endpoint or not self.token:
            raise RuntimeError("Missing Astra API endpoint or token for HTTP fallback.")
        payload = {
            "createCollection": {
                "name": self.collection_name,
                "options": {
                    "vector": {
                        "dimension": self.embedding_dim,
                        "metric": self.metric,
                    }
                },
            }
        }
        url = f"{self.api_endpoint.rstrip('/')}/api/json/v1/{self.keyspace}"
        req = urlrequest.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.token}",
                "Token": self.token,
            },
            method="POST",
        )
        with urlrequest.urlopen(req, timeout=30) as resp:
            _ = resp.read()

    def _delete_collection_compatible(self) -> None:
        errors: List[str] = []
        delete_calls = [
            lambda: self.db.drop_collection(self.collection_name),
            lambda: self.db.delete_collection(self.collection_name),
            lambda: self.db.command({"deleteCollection": {"name": self.collection_name}}),
            lambda: self.db.command({"dropCollection": {"name": self.collection_name}}),
            self._delete_collection_via_http,
        ]
        for call in delete_calls:
            try:
                call()
                return
            except Exception as e:
                errors.append(f"{type(e).__name__}: {e}")
        raise RuntimeError(
            f"Failed to delete collection '{self.collection_name}'. Errors: {' | '.join(errors)}"
        )

    def _delete_collection_via_http(self) -> None:
        if not self.api_endpoint or not self.token:
            raise RuntimeError("Missing Astra API endpoint or token for HTTP fallback.")
        payload = {"deleteCollection": {"name": self.collection_name}}
        url = f"{self.api_endpoint.rstrip('/')}/api/json/v1/{self.keyspace}"
        req = urlrequest.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.token}",
                "Token": self.token,
            },
            method="POST",
        )
        with urlrequest.urlopen(req, timeout=30) as resp:
            _ = resp.read()

    def _ensure_vector_search_enabled(self) -> None:
        # Astra rejects zero and near-zero cosine query vectors, so use a simple unit vector.
        probe_vector = [0.0] * self.embedding_dim
        if self.embedding_dim > 0:
            probe_vector[0] = 1.0
        try:
            try:
                cursor = self.collection.find(
                    {},
                    sort={"$vector": probe_vector},
                    limit=1,
                    include_similarity=True,
                )
            except TypeError:
                cursor = self.collection.find(
                    {},
                    sort={"$vector": probe_vector},
                    limit=1,
                )
            # force request execution
            _ = list(cursor)
        except Exception as e:
            msg = str(e)
            if "Zero and near-zero vectors cannot be indexed or queried with cosine similarity" in msg:
                raise RuntimeError(
                    "Vector search validation used an invalid zero probe vector. "
                    "This collection may still be valid; retry with the updated code."
                ) from e
            if (
                "VECTOR_SEARCH_NOT_SUPPORTED" in msg
                or "Vector search is not enabled" in msg
                or "INVALID_DATABASE_QUERY" in msg
                or "requires the column to be indexed" in msg
            ):
                if self.recreate_if_non_vector:
                    print(
                        f"[AstraKG20CVectorDB] Recreating non-vector collection '{self.collection_name}' as vector-enabled...",
                        flush=True,
                    )
                    self._delete_collection_compatible()
                    self._create_collection_compatible()
                    self.collection = self.db.get_collection(self.collection_name)
                    # One retry after recreate
                    try:
                        try:
                            cursor = self.collection.find(
                                {},
                                sort={"$vector": probe_vector},
                                limit=1,
                                include_similarity=True,
                            )
                        except TypeError:
                            cursor = self.collection.find(
                                {},
                                sort={"$vector": probe_vector},
                                limit=1,
                            )
                        _ = list(cursor)
                        return
                    except Exception as e2:
                        raise RuntimeError(
                            "Collection recreated, but vector search is still unavailable. "
                            f"Original: {e}; After recreate: {e2}"
                        ) from e2

                raise RuntimeError(
                    f"Collection '{self.collection_name}' exists but is not vector-enabled. "
                    "Delete it and recreate as vector-enabled, or run index with recreate flag. "
                    f"Original error: {e}"
                ) from e
            raise

    def _ensure_collection(self) -> Any:
        existing = set(self.db.list_collection_names())
        if self.collection_name not in existing:
            self._create_collection_compatible()
        return self.db.get_collection(self.collection_name)

    def upsert_entity_vectors(
        self,
        entity_docs: List[Dict[str, Any]],
        vectors: List[List[float]],
    ) -> None:
        if len(entity_docs) != len(vectors):
            raise ValueError("entity_docs and vectors lengths must match.")

        for entity, vector in zip(entity_docs, vectors):
            doc = {
                "_id": entity["entity_id"],
                "entity_id": entity["entity_id"],
                "name": entity["name"],
                "type": entity["type"],
                "text": entity["text"],
                "$vector": vector,
            }
            # idempotent upsert semantics, close to wikidata helper behavior
            self.collection.replace_one({"_id": doc["_id"]}, doc, upsert=True)

    def query_topk(self, query_vector: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        try:
            try:
                cursor = self.collection.find(
                    {},
                    sort={"$vector": query_vector},
                    limit=top_k,
                    include_similarity=True,
                )
            except TypeError:
                cursor = self.collection.find(
                    {},
                    sort={"$vector": query_vector},
                    limit=top_k,
                )
        except Exception as e:
            msg = str(e)
            if (
                "VECTOR_SEARCH_NOT_SUPPORTED" in msg
                or "Vector search is not enabled" in msg
                or "INVALID_DATABASE_QUERY" in msg
                or "requires the column to be indexed" in msg
            ):
                raise RuntimeError(
                    f"Collection '{self.collection_name}' is not vector-query capable. "
                    "Recreate it as a vector collection and re-index KG20C entities."
                ) from e
            raise

        results: List[Dict[str, Any]] = []
        try:
            for row in cursor:
                sim = row.get("$similarity")
                if sim is None:
                    sim = row.get("similarity")
                if sim is None:
                    sim = row.get("score")
                results.append(
                    {
                        "entity_id": row.get("entity_id", row.get("_id")),
                        "name": row.get("name", ""),
                        "type": row.get("type", ""),
                        "text": row.get("text", ""),
                        "similarity_score": float(sim) if sim is not None else None,
                        "source": "Astra Vector Search",
                    }
                )
        except Exception as e:
            msg = str(e)
            if (
                "VECTOR_SEARCH_NOT_SUPPORTED" in msg
                or "Vector search is not enabled" in msg
                or "INVALID_DATABASE_QUERY" in msg
                or "requires the column to be indexed" in msg
            ):
                raise RuntimeError(
                    f"Collection '{self.collection_name}' is not vector-query capable. "
                    "Recreate it as a vector collection and re-index KG20C entities."
                ) from e
            raise
        return results


def _load_kg20c_entities(entity_info_path: str = DEFAULT_KG20C_ENTITY_INFO_PATH) -> pd.DataFrame:
    return pd.read_csv(entity_info_path, sep="\t", dtype=str).fillna("")


def _build_entity_docs(entity_df: pd.DataFrame) -> List[Dict[str, str]]:
    docs: List[Dict[str, str]] = []
    for _, row in entity_df.iterrows():
        entity_id = str(row["id"]).strip()
        name = str(row.get("name", "")).strip()
        entity_type = str(row.get("type", "")).strip()
        text = f"name: {name}; type: {entity_type}"
        docs.append(
            {
                "entity_id": entity_id,
                "name": name,
                "type": entity_type,
                "text": text,
            }
        )
    return docs


@lru_cache(maxsize=4)
def _load_kg20c_triple_index(
    triples_path: str = DEFAULT_KG20C_TRIPLES_PATH,
) -> Dict[str, List[Dict[str, str]]]:
    entity_to_triples: Dict[str, List[Dict[str, str]]] = {}
    with open(triples_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            triple = {
                "subject": item.get("head_id", ""),
                "predicate": item.get("relation", ""),
                "object": item.get("tail_id", ""),
                "subject_label": item.get("head_label", item.get("head_id", "")),
                "predicate_label": item.get("relation", ""),
                "object_label": item.get("tail_label", item.get("tail_id", "")),
                "pretty_string": (
                    f"({item.get('head_label', item.get('head_id', ''))}, "
                    f"{item.get('relation', '')}, "
                    f"{item.get('tail_label', item.get('tail_id', ''))})"
                ),
            }
            head_id = item.get("head_id", "")
            tail_id = item.get("tail_id", "")
            if head_id:
                entity_to_triples.setdefault(head_id, []).append(triple)
            if tail_id:
                entity_to_triples.setdefault(tail_id, []).append(triple)
    return entity_to_triples


def fetch_kg20c_triples_for_entity_ids(
    entity_hits: List[Dict[str, Any]],
    triples_path: str = DEFAULT_KG20C_TRIPLES_PATH,
    max_entities: int = 10,
    max_triples_per_entity: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Expand top KG20C entity hits into KG20C triples, preserving entity ranking order.
    """
    triple_index = _load_kg20c_triple_index(triples_path)
    collected: List[Dict[str, Any]] = []

    ranked_hits = sorted(
        [hit for hit in entity_hits if isinstance(hit, dict) and hit.get("entity_id")],
        key=lambda x: x.get("similarity_score", 0.0),
        reverse=True,
    )[:max_entities]

    for hit in ranked_hits:
        entity_id = str(hit.get("entity_id", ""))
        triples = triple_index.get(entity_id, [])
        if max_triples_per_entity is not None:
            triples = triples[:max_triples_per_entity]

        for triple in triples:
            collected.append(
                {
                    **triple,
                    "source_entity_id": entity_id,
                    "source_entity_label": hit.get("name", entity_id),
                    "source_entity_score": hit.get("similarity_score", 0.0),
                }
            )

    return collected


def index_kg20c_entities_to_astra(
    entity_info_path: str = DEFAULT_KG20C_ENTITY_INFO_PATH,
    collection_name: str = DEFAULT_ASTRA_COLLECTION,
    embedding_dim: int = 512,
    model_name: str = "jinaai/jina-embeddings-v3",
    batch_size: int = 64,
    recreate_collection: bool = False,
) -> None:
    entity_df = _load_kg20c_entities(entity_info_path)
    docs = _build_entity_docs(entity_df)

    embedder = JinaEmbeddingsV3(
        model_name=model_name,
        embedding_dim=embedding_dim,
        batch_size=batch_size,
    )
    db = AstraKG20CVectorDB(
        collection_name=collection_name,
        embedding_dim=embedding_dim,
        recreate_if_non_vector=recreate_collection,
        validate_vector_search=False,
    )

    for i in range(0, len(docs), batch_size):
        chunk = docs[i : i + batch_size]
        vectors = embedder.encode_documents([d["text"] for d in chunk])
        db.upsert_entity_vectors(chunk, vectors)
        print(f"[index_kg20c_entities_to_astra] Upserted {min(i + batch_size, len(docs))}/{len(docs)}", flush=True)


def retrieve_relevant_kg20c_entities(
    query_text: str,
    top_k: int = 10,
    collection_name: str = DEFAULT_ASTRA_COLLECTION,
    embedding_dim: int = 512,
    model_name: str = "jinaai/jina-embeddings-v3",
) -> List[Dict[str, Any]]:
    embedder = JinaEmbeddingsV3(model_name=model_name, embedding_dim=embedding_dim)
    db = AstraKG20CVectorDB(
        collection_name=collection_name,
        embedding_dim=embedding_dim,
        recreate_if_non_vector=False,
        validate_vector_search=True,
    )

    query_vector = embedder.encode_query(query_text)
    return db.query_topk(query_vector, top_k=top_k)


def fetch_and_save_relevant_kg20c_entities(
    query_df: pd.DataFrame,
    id_column: str,
    query_column: str,
    output_path: str,
    top_k: int = 10,
    collection_name: str = DEFAULT_ASTRA_COLLECTION,
    embedding_dim: int = 512,
    model_name: str = "jinaai/jina-embeddings-v3",
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Keep similar shape to wiki_query.fetch_and_save_relevant_ids:
      {query_id: [{entity_id, similarity_score, ...}, ...], ...}
    """
    embedder = JinaEmbeddingsV3(model_name=model_name, embedding_dim=embedding_dim)
    db = AstraKG20CVectorDB(
        collection_name=collection_name,
        embedding_dim=embedding_dim,
        recreate_if_non_vector=False,
        validate_vector_search=True,
    )

    results: Dict[str, List[Dict[str, Any]]] = {}
    rows = [(str(r[id_column]), str(r[query_column])) for _, r in query_df.iterrows()]

    print("[fetch_and_save_relevant_kg20c_entities] Start querying Astra vector DB...", flush=True)
    for idx, (qid, text) in enumerate(rows, start=1):
        qvec = embedder.encode_query(text)
        results[qid] = db.query_topk(qvec, top_k=top_k)
        print(f"[fetch_and_save_relevant_kg20c_entities] Finished {idx}/{len(rows)}", flush=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(
        f"[fetch_and_save_relevant_kg20c_entities] Completed. Results saved to {output_path}",
        flush=True,
    )
    return results
