import argparse
import numpy as np
import json


from sklearn.externals import joblib
from rcc_conf import TEST_FILE, RFIELDS_OUTFILE
from rcc_utils import json_from_file, load_rcc_test_dataset
from rfields_rec_train import TitleAbstractExtractor


MODEL_PATH = 'models/'
RFIELDS_REC_MODEL = MODEL_PATH + 'rfields_rec.model.pub.'
RFIELDS_RT_MODEL = MODEL_PATH + 'rfields_rec.model.rt.'
LEVELS = ['l1', 'l2', 'l3']
TOPK = {
    'l1': 3,
    'l2': 4,
    'l3': 5
}
L1_W = [(TOPK['l1'] - i) / TOPK['l1'] for i in range(TOPK['l1'])]
L2_W = [(TOPK['l2'] - i) / TOPK['l2'] for i in range(TOPK['l2'])]
L3_W = [(TOPK['l3'] - i) / TOPK['l3'] for i in range(TOPK['l3'])]


def _load_models():
    print('Loading models...')
    rec_models = {}
    rt_clf = {}
    for level in LEVELS:
        rec_models[level] = joblib.load(RFIELDS_REC_MODEL + level)
        rt_clf[level] = joblib.load(RFIELDS_RT_MODEL + level)
    return rec_models, rt_clf


def _generate_top_k_rec(model, k, test):
    # pred = model.predict([test])
    # print('{}\t{}'.format(test['title'], pred))
    # print(model.classes_)
    confidences = model.decision_function([test])
    top_conf = np.argsort(confidences, axis=1)[0, -k:][::-1]
    # print(confidences)
    top_rec = [model.classes_[k] for k in top_conf]
    return top_rec


def _rerank_top_rec(rec):
    all_rec = []
    l2_from_l3 = []
    for i, v in enumerate(rec['l3']):
        l2_name = v[: v.rfind('-')]
        l1_name = l2_name[: l2_name.rfind('-')]
        if l2_name not in rec['l2']:
            continue
        if l1_name not in rec['l1']:
            continue
        l2_from_l3.append(l2_name)
        idx2 = rec['l2'].index(l2_name)
        idx1 = rec['l1'].index(l1_name)
        score = L3_W[i] * L2_W[idx2] * L1_W[idx1]
        all_rec.append({'name': v, 'score': score})

    for i, v in enumerate(rec['l2']):
        if v in l2_from_l3:  # redundant since we have more specific one
            continue
        l1_name = v[: v.rfind('-')]
        if l1_name not in rec['l1']:
            continue
        idx1 = rec['l1'].index(l1_name)
        score = L2_W[i] * L1_W[idx1]
        all_rec.append({'name': v, 'score': score})
    return sorted(all_rec, key=lambda x: x['score'], reverse=True)


def _make_prediction(rfields_rec, rt_clf, metadata, parsed_pub):
    print(metadata)
    pub_test = {
        'title': metadata['title'],
        'long_abstract': parsed_pub['metadata']['abstractText'] if parsed_pub['metadata']['abstractText'] is not None else '.'
    }
    topk_rec = [_generate_top_k_rec(rfields_rec[l], TOPK[l], pub_test)
               for l in LEVELS]
    pub_rec = dict(zip(LEVELS, topk_rec))
    print(pub_rec)
    print(_rerank_top_rec(pub_rec))


def _predict(rfields_rec, rt_clf, parsed_data, output_dir):
    """ This function predict research fields
    """
    print('Test data: {}'.format(args.input_dir))
    print('Loading test data...')
    parsed_pubs_test = load_rcc_test_dataset(args.input_dir)
    test_list = json_from_file(args.input_dir + TEST_FILE)

    print('Running prediction...')
    for test in test_list:
        pub = parsed_pubs_test[str(test['publication_id'])]
        _make_prediction(rfields_rec, rt_clf, test, pub)


def main(args):
    """ This script predict and extract dataset citation from rcc test folder.
        It will generate `data_set_citations.json` and
        `data_set_mentions.json`.
    """
    rfields_rec, rt_clf = _load_models()
    _predict(rfields_rec, rt_clf, args.input_dir, args.output_dir)


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
