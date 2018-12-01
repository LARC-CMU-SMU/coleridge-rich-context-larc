import argparse
import json
import numpy as np

from sklearn.externals import joblib
from tqdm import tqdm

from rcc_conf import RESEARCH_METHODS_FILE, TEST_FILE, RMETHODS_OUTFILE
from rcc_utils import json_from_file, load_rcc_test_dataset
from rmethods_rec_train import RMethodContextExtractor


TOP_K = 3
MODEL_PATH = 'models/'
RMETHODS_MODEL = MODEL_PATH + 'rmethods_rec.model'


def _load_models():
    print('Loading models...')
    return joblib.load(RMETHODS_MODEL)


def _make_recommendation(model, k, title, pub):
    if pub['metadata']['sections'] is None:
        contents = '.'
    else:
        contents = '.'.join([s['text'] for s in pub['metadata']['sections']])
    data = {
        'title': title,
        'contents': contents
    }
    proba = model.predict_proba([data])
    top_conf = np.argsort(proba, axis=1)[0, -k:][::-1]
    top_rec = [{'method': model.classes_[k], 'score': proba[0, k]}
               for k in top_conf]
    return top_rec


def _predict(rmethods_mod, input_dir, output_dir):
    """ This function loads test data, recommends research methods, and
        generates output json file.
    """
    print('Test data: {}'.format(input_dir))
    print('Loading test data...')
    parsed_pubs_test = load_rcc_test_dataset(input_dir)
    rmethods_graph = json_from_file(RESEARCH_METHODS_FILE)['@graph']
    rmethods = {method['@id']: method['skos:prefLabel']['@value']
                for method in rmethods_graph
                if method['@type'] == 'skos:Concept'}
    test_list = json_from_file(input_dir + TEST_FILE)

    results = []
    for test in tqdm(test_list, ascii=True,
                     desc='Running research method recommendation'):
        pub_id = test['publication_id']
        pub = parsed_pubs_test[str(pub_id)]
        top_recs = _make_recommendation(rmethods_mod, TOP_K,
                                        test['title'], pub)
        if len(top_recs) == 0:
            continue

        output = [{
            'publication_id': pub_id,
            'method': rmethods[r['method']],
            'score': r['score']
        } for r in top_recs]
        results.append(output)

    # Save predictions to RMETHODS_OUTFILE
    with open(output_dir + RMETHODS_OUTFILE, 'w') as f:
        f.write(json.dumps(results, indent=4))


def main(args):
    """ This script recommends research methods for all publications listed in
        `publications.json` in input_dir. It will generate output
        `methods.json` in output_dir.
    """
    rmethods_mod = _load_models()
    _predict(rmethods_mod, args.input_dir, args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Recommend research methods  \
                                     for all publications in rcc test folder.')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Location of `publications.json`. We assume that \
                        json folder containing parsed publication is at this \
                        location too.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for save the output. ')
    args = parser.parse_args()
    main(args)
