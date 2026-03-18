import json
import os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch

load_dotenv()

_ES_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")

# Connect to Elasticsearch
try:
    es = Elasticsearch([_ES_URL])
    if es.ping():
        print("Connected to Elasticsearch!")
    else:
        print("Could not connect to Elasticsearch.")
except ConnectionError as e:
    print(f"Error connecting to Elasticsearch: {e}")

# Hàm tìm kiếm dữ liệu trong Elasticsearch
def search_data(index_name, query, top_k=10):
    # Thực hiện tìm kiếm với giới hạn top_k
    response = es.search(
        index=index_name,
        body={
            "query": {
                "match": {
                    "text": query  # Tìm kiếm theo nội dung văn bản
                }
            },
            "sort": [
                {"_score": {"order": "desc"}}  # Sắp xếp theo điểm số giảm dần
            ],
            "size": top_k  # Chỉ định số lượng kết quả muốn lấy ra
        }
    )

    # Lấy kết quả từ response
    results = []
    for hit in response["hits"]["hits"]:
        results.append({
            "text": hit["_source"]["text"],
        })

    return results


