import argparse
import json

from sklearn.externals import joblib
from rcc_conf import TEST_FILE, DATASET_CITATION_OUTFILE
from rcc_utils import json_from_file, load_rcc_test_dataset
from dataset_detect_train import FeatureGroupExtractor


MODEL_PATH = 'models/'
DATASET_DETECT_MODEL = MODEL_PATH + 'dataset_detect.model'


def _load_models():
    print('Loading models...')
    svm_clf = joblib.load(DATASET_DETECT_MODEL)
    # Amila: load other models below, we put all models in dictionary

    return {'detector': svm_clf}


def _make_prediction(models, metadata, parsed_pub):
    citing_pred = models['detector'].predict([parsed_pub])
    print(citing_pred[0])

    if citing_pred[0] == 0:
        return None

    # Amila: Code for Dataset recognition section and mention detection below

    return {
        'publication_id': metadata['publication_id'],
        'data_set_id': -1,
        'score': 0.00,
        'mention_list': []
    }


def _predict(models, parsed_data, output_dir):
    """ This function predict dataset citations
    """
    print('Test data: {}'.format(args.input_dir))
    print('Loading test data...')
    parsed_pubs_test = load_rcc_test_dataset(args.input_dir)
    test_list = json_from_file(args.input_dir + TEST_FILE)

    print('Running prediction...')
    predictions = []
    for test in test_list:
        pub = parsed_pubs_test[str(test['publication_id'])]
        pred = _make_prediction(models, test, pub)
        if pred is None:
            continue
        predictions.append(pred)

    # Save predictions to DATASET_CITATION_OUTFILE
    with open(output_dir + DATASET_CITATION_OUTFILE, 'w') as f:
        f.write(json.dumps(predictions))

    # Save extracted mentions to DATASET_MENTION_OUTFILE


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
