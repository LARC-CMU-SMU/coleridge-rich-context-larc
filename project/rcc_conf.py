
# Resource files
RESEARCH_METHODS_FILE = 'resources/sage_research_methods.json'
RESEARCH_FIELDS_FILE = 'resources/sage_research_fields.json'

# Train Set configuration
PARSED_TRAIN_PUB_PATH = '../data/train_test/files/json/'
TRAIN_PUB_METADATA = '../data/train_test/publications.json'
TRAIN_DATASET_CITATION = '../data/train_test/data_set_citations.json'
TRAIN_CACHE_PUB = '../data/train_test/train_cache.json'

# Dev Set configuration
PARSED_DEV_PUB_PATH = '../data/dev/files/json/'
DEV_PUB_METADATA = '../data/dev/publications.json'
DEV_DATASET_CITATION = '../data/dev/data_set_citations.json'
DEV_CACHE_PUB = '../data/dev/dev_cache.json'

CONF = {
    'train': {
        'parsed_pub_path': PARSED_TRAIN_PUB_PATH,
        'pub_metadata': TRAIN_PUB_METADATA,
        'citation': TRAIN_DATASET_CITATION,
        'rcc_cache': TRAIN_CACHE_PUB
    },
    'dev': {
        'parsed_pub_path': PARSED_DEV_PUB_PATH,
        'pub_metadata': DEV_PUB_METADATA,
        'citation': DEV_DATASET_CITATION,
        'rcc_cache': DEV_CACHE_PUB
    }
}

# Random Seed, for reproducibility
RAND_SEED = 2018
