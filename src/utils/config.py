from pathlib import Path

TOTAL_DOCUMENTS = 502939
VALIDATION_RATIO = 0.1
CE_SCORE_MARGIN = 3

PROJECT_ROOT = Path(__file__).parent.parent.parent

DATA_DIR = PROJECT_ROOT / 'data'
MSMARCO_DIR = DATA_DIR / 'msmarco'
HARD_NEGATIVES = DATA_DIR / 'msmarco-hard-negatives.jsonl.gz'
PREPARED_DIR = DATA_DIR / 'prepared'

QUERY_FILE = MSMARCO_DIR / 'queries.jsonl'
CORPUS_FILE = MSMARCO_DIR / 'corpus.jsonl'
TRAIN_QRELS_FILE = MSMARCO_DIR / 'qrels/train.tsv'