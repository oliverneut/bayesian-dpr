DATASET_METADATA = {
    "msmarco": {
        "root_dir": "data/msmarco",
        "query_file": "data/prepared/queries-{}.jsonl",
        "qrels_file": "data/msmarco/qrels/{}.tsv"
    }
}


def get_query_file(split: str):
    return DATASET_METADATA["msmarco"]["query_file"].format(split)

def get_qrels_file(split: str):
    return DATASET_METADATA["msmarco"]["qrels_file"].format(split)