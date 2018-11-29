import argparse

from sklearn.externals import joblib
from rcc_utils import load_rcc_test_dataset
from dataset_detect_train import FeatureGroupExtractor


MODEL_PATH = 'models/'
DATASET_DETECT_MODEL = MODEL_PATH + 'dataset_detect.model'


def _load_models():
    print('Loading models...')
    svm_clf = joblib.load(DATASET_DETECT_MODEL)
    # load other models
    return {'detector': svm_clf}


def _predict(models, parsed_data, output_dir):
    """ This function predict dataset citations
    """
    print('Test data: {}'.format(args.input_dir))
    print('Loading test data...')
    parsed_test = load_rcc_test_dataset(args.input_dir)

    print('Running prediction...')
    citing_pred = models['detector'].predict(parsed_test)
    print(citing_pred)


def main(args):
    """ This script predict and extract dataset citation from rcc test folder.
        It will generate `data_set_citations.json` and
        `data_set_mentions.json`.
    """
    models = _load_models()
    _predict(models, args.input_dir, args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict dataset citations  \
                                     for all publications in rcc test folder.')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Folder containing parsed publications, and text \
                        folder to be predicted.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for save the output. ')
    args = parser.parse_args()
    main(args)
