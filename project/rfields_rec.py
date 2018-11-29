import argparse

from sklearn.externals import joblib
from rcc_conf import TEST_FILE, RFIELDS_OUTFILE
from rcc_utils import json_from_file, load_rcc_test_dataset
from rfields_rec_train import TitleAbstractExtractor


MODEL_PATH = 'models/'
RFIELDS_REC_MODEL = MODEL_PATH + 'rfields_rec.model.pub.'
RFIELDS_RT_MODEL = MODEL_PATH + 'rfields_rec.model.rt.'
LEVELS = ['l1', 'l2', 'l3']


def _load_models():
    print('Loading models...')
    rec_models = {}
    rt_clf = {}
    for level in LEVELS:
        rec_models[level] = joblib.load(RFIELDS_REC_MODEL + level)
        rt_clf[level] = joblib.load(RFIELDS_RT_MODEL + level)
    return rec_models, rt_clf


def _make_prediction(rfields_rec, rt_clf, parsed_data, output_dir):
    """ This function predict dataset citations
    """
    print('Test data: {}'.format(args.input_dir))
    print('Loading test data...')
    parsed_pubs = load_rcc_test_dataset(args.input_dir)
    test_list = json_from_file(args.input_dir + TEST_FILE)

    print('Running prediction...')
    for level in LEVELS:
        for test in test_list:
            pub = parsed_pubs[str(test['publication_id'])]
            pub_test = {
                'title': test['title'],
                'long_abstract': pub['metadata']['abstractText'] if pub['metadata']['abstractText'] is not None else '.'
            }
            pred = rfields_rec[level].predict([pub_test])
            print(pred)


def main(args):
    """ This script predict and extract dataset citation from rcc test folder.
        It will generate `data_set_citations.json` and
        `data_set_mentions.json`.
    """
    rfields_rec, rt_clf = _load_models()
    _make_prediction(rfields_rec, rt_clf, args.input_dir, args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Recommend research fields  \
                                     for all publications in rcc test folder.')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Location of `publication.json`. We assume that \
                        text and json folder are at this location too.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for save the output. ')
    args = parser.parse_args()
    main(args)
