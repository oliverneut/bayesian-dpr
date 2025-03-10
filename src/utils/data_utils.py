DATASET_METADATA = {
    "msmarco": {
        "root_dir": "data/msmarco",
        "query_file": "data/prepared/queries-{}.jsonl"
    }
}


def get_query_file(split: str):
    return DATASET_METADATA["msmarco"]["query_file"].format(split)