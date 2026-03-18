from search_document.search_with_bge import QdrantSearch_bge
from search_document.search_with_e5 import QdrantSearch_e5
from search_document.search_elastic import search_data
import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

logger = logging.getLogger(__name__)

_collections = [c.strip() for c in os.getenv("COLLECTIONS", "vn_law_bge_m3,vn_law_e5").split(",")]
BGE_COLLECTION = _collections[0]
E5_COLLECTION = _collections[1]

# --- Qdrant health check at startup ---
_MIN_POINTS = 1000
try:
    _qclient = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
    for _col in [BGE_COLLECTION, E5_COLLECTION]:
        try:
            _info = _qclient.get_collection(_col)
            _count = _info.points_count
            if _count == 0:
                logger.warning(f"Qdrant collection '{_col}' is EMPTY. Run retrieval/ingest.sh to populate it.")
            elif _count < _MIN_POINTS:
                logger.warning(f"Qdrant collection '{_col}' has only {_count} points (< {_MIN_POINTS}). Search quality may be poor.")
            else:
                logger.info(f"Qdrant collection '{_col}': {_count} points — OK")
        except Exception as e:
            logger.warning(f"Qdrant collection '{_col}' not found or unreachable: {e}")
except Exception as e:
    logger.warning(f"Could not connect to Qdrant Cloud for health check: {e}")

# Khởi tạo các search class ở cấp module để tái sử dụng
bge_search_instance = QdrantSearch_bge(
        collection_name=BGE_COLLECTION,
        model_name="BAAI/bge-m3",
        use_fp16=True
    )

e5_search_instance = QdrantSearch_e5(
        collection_name=E5_COLLECTION,
        model_name="intfloat/multilingual-e5-large",
        use_fp16=True
    )

elastic_params = {
        'index_name': 'legal_data_part2',
        'top_k': 30
    }

class CombinedSearch:
    def __init__(self):
        self.bge_search = bge_search_instance  # Reuse instance
        self.e5_search = e5_search_instance  # Reuse instance
        self.elastic_index = elastic_params['index_name']
        self.elastic_top_k = elastic_params['top_k']

    def search(self, query_text, top_k=30):
        """
        Perform a combined search across BGE, E5, and Elasticsearch in parallel.

        Args:
            query_text (str): The query string.
            top_k (int): Number of top results to retrieve from each method.

        Returns:
            list: Combined search results.
        """
        def _bge(): return self.bge_search.search(query_text, limit=top_k)
        def _e5():  return self.e5_search.search(query_text, limit=top_k)
        def _es():  return search_data(self.elastic_index, query_text, top_k=self.elastic_top_k)

        bge_results = e5_results = None
        elastic_results = []

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(_bge): "bge",
                       executor.submit(_e5):  "e5",
                       executor.submit(_es):  "es"}
            for future in as_completed(futures):
                key = futures[future]
                try:
                    result = future.result()
                    if key == "bge":
                        bge_results = result
                    elif key == "e5":
                        e5_results = result
                    else:
                        elastic_results = result
                except Exception as e:
                    logger.warning(f"Search '{key}' failed: {e}")

        # Combine and normalize results
        combined_results = []

        if bge_results is not None:
            for result in bge_results.points:
                combined_results.append(result.payload["text"])

        if e5_results is not None:
            for result in e5_results.points:
                combined_results.append(result.payload["text"])

        for result in elastic_results:
            combined_results.append(result['text'])

        combined_results = list(set(combined_results))

        return combined_results
