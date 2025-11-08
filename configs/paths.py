from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / 'models'
LOGS_DIR = BASE_DIR / 'logs'


DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw' / 'dataset-original'
TEST_DATA_DIR = DATA_DIR / 'test'
TRAIN_DATA_DIR = DATA_DIR / 'train'
VAL_DATA_DIR = DATA_DIR / 'validation'