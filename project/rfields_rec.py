import argparse
import pandas as pd
import numpy as np
import json

from sklearn.externals import joblib
from tqdm import tqdm
from rcc_conf import RESEARCH_FIELDS_FILE, TEST_FILE, RFIELDS_OUTFILE
from rcc_utils import json_from_file, load_rcc_test_dataset
from rfields_rec_train import TitleAbstractExtractor


MODEL_PATH = 'models/'
RFIELDS_REC_MODEL = MODEL_PATH + 'rfields_rec.model.pub.'
RFIELDS_RT_MODEL = MODEL_PATH + 'rfields_rec.model.rt.'

# Constant for Research Fields Recommendation
LEVELS = ['l1', 'l2', 'l3']
TOPK = {
    'l1': 3,
    'l2': 4,
    'l3': 5
}
L1_W = [(TOPK['l1'] - i) / TOPK['l1'] for i in range(TOPK['l1'])]
L2_W = [(TOPK['l2'] - i) / TOPK['l2'] for i in range(TOPK['l2'])]
L3_W = [(TOPK['l3'] - i) / TOPK['l3'] for i in range(TOPK['l3'])]
REC_THRESHOLD = 0.4
MAX_REFS = 50
MIN_REFS_SUP = 0.1
PUB_W = 0.7
REF_W = 1 - PUB_W


def _load_models():
    print('Loading models...')
    rec_models = {}
    rt_clf = {}
    for level in LEVELS:
        rec_models[level] = joblib.load(RFIELDS_REC_MODEL + level)
        rt_clf[level] = joblib.load(RFIELDS_RT_MODEL + level)
    return rec_models, rt_clf


def _load_rfields_from_file(filename):
    print('Loading research fields list...')
    rfields_df = pd.read_csv(filename)
    rfields_df['L2_ID'] = [l3[: l3.rfind('-')] for l3 in rfields_df['ID']]
    l2 = dict(zip(rfields_df['L2_ID'], rfields_df['L2']))
    l3 = dict(zip(rfields_df['ID'], rfields_df['L3']))
    return {**l3, **l2}


def _generate_top_k_rec(model, k, test):
    confidences = model.decision_function([test])
    top_conf = np.argsort(confidences, axis=1)[0, -k:][::-1]
    top_rec = [model.classes_[k] for k in top_conf]
    return top_rec


def _generate_top_k_rec_from_refs(model, k, ref_titles):
    proba = model.predict_proba(ref_titles)
    top_proba = np.argsort(proba, axis=1)[:, -k:][:, ::-1]
    top_rec = []
    for arr in top_proba:
        top_rec.append([model.classes_[k] for k in arr])
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


def _make_combined_recs(pub_recs, ref_recs):
    recs = {}
    for r in pub_recs:
        recs[r['name']] = {
            'pub': r['score'],
            'ref': 0.
        }
    for r in ref_recs:
        if r['name'] not in recs:
            recs[r['name']] = {
                'pub': 0.,
                'ref': r['score']
            }
        else:
            recs[r['name']]['ref'] = r['score']
    combined_recs = [{'name': k, 'score': PUB_W * v['pub'] + REF_W * v['ref']}
                     for (k, v) in recs.items()]
    sorted_combined_recs = sorted(combined_recs,
                                  key=lambda x: x['score'],
                                  reverse=True)
    return [rf for rf in sorted_combined_recs if rf['score'] >= REC_THRESHOLD]


def _make_recommendation(rfields_rec, rt_clf, metadata, parsed_pub):
    pub_test = {
        'title': metadata['title'],
        'long_abstract': parsed_pub['metadata']['abstractText'] if parsed_pub['metadata']['abstractText'] is not None else '.'
    }
    topk_rec = [_generate_top_k_rec(rfields_rec[l], TOPK[l], pub_test)
                for l in LEVELS]
    topk_rec_by_level = dict(zip(LEVELS, topk_rec))
    pub_recs = _rerank_top_rec(topk_rec_by_level)

    # If references are available, we incorporate them, else return pub_rec
    refs = parsed_pub['metadata']['references']
    if len(refs) == 0:
        return [{'name': rf['name'], 'score': PUB_W * rf['score']}
                for rf in pub_recs
                if PUB_W * rf['score'] >= REC_THRESHOLD]

    titles = [r['title'] for r in refs]
    # avoiding very long refs, we only take first MAX_REFS references
    if len(titles) > MAX_REFS:
        titles = titles[:MAX_REFS]
    ref_topk_rec = [_generate_top_k_rec_from_refs(rt_clf[l], TOPK[l], titles)
                    for l in LEVELS]
    ref_recs = {}
    for i in range(len(titles)):
        rec_by_level = {
            'l1': ref_topk_rec[0][i],
            'l2': ref_topk_rec[1][i],
            'l3': ref_topk_rec[2][i],
        }
        recs = _rerank_top_rec(rec_by_level)
        for r in recs:
            if r['name'] not in ref_recs:
                ref_recs[r['name']] = []
            ref_recs[r['name']].append(r['score'])
    filtered_ref_recs = [{'name': k, 'score': np.mean(v)}
                         for (k, v) in ref_recs.items()
                         if len(v) >= MIN_REFS_SUP * len(titles)]
    return _make_combined_recs(pub_recs, filtered_ref_recs)


def _predict(rfields_rec, rt_clf, parsed_data, output_dir):
    """ This function load test data, recommend research fields, and
        generate output json file.
    """
    print('Test data: {}'.format(args.input_dir))
    print('Loading test data...')
    parsed_pubs_test = load_rcc_test_dataset(args.input_dir)
    test_list = json_from_file(args.input_dir + TEST_FILE)

    rfields = _load_rfields_from_file(RESEARCH_FIELDS_FILE)
    results = []
    for test in tqdm(test_list, ascii=True,
                     desc='Running research field recommendation'):
        pub_id = test['publication_id']
        pub = parsed_pubs_test[str(pub_id)]
        top_recs = _make_recommendation(rfields_rec, rt_clf, test, pub)
        if len(top_recs) == 0:
            continue

        output = [{
            'publication_id': pub_id,
            'research_field': rfields[r['name']],
            'score': r['score']
        } for r in top_recs]
        results.append(output)

    # Save predictions to RFIELDS_OUTFILE
    with open(output_dir + RFIELDS_OUTFILE, 'w') as f:
        f.write(json.dumps(results))


def main(args):
    """ This script recommends research fields for all publications listed in
        `publications.json` in input_dir. It will generate output
        `research_fields.json` in output_dir.
    """
    rfields_rec, rt_clf = _load_models()
    _predict(rfields_rec, rt_clf, args.input_dir, args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Recommend research fields  \
                                     for all publications in rcc test folder.')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Location of `publications.json`. We assume that \
                        json folder containing parsed publication is at this \
                        location too.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for save the output. ')
    args = parser.parse_args()
    main(args)
