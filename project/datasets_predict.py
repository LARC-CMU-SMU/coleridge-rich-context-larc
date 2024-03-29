import argparse
import json
import numpy as np
import math
import pickle

from nltk import word_tokenize
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
import sentence_filtering
import aux_functions as aux_fun
auxfun=aux_fun.AuxFunClass

from rcc_conf import TEST_FILE, DATASET_CITATION_OUTFILE, DATASET_MENTION_OUTFILE, TEXT_PUB_PATH, RESOURCE_PATH
from rcc_utils import json_from_file, load_rcc_test_dataset
from dataset_detect_train import FeatureGroupExtractor


MODEL_PATH = 'models/'
DATASET_DETECT_MODEL = MODEL_PATH + 'dataset_detect.model'

DATASET_ID_MAPPING = MODEL_PATH + 'dataset_mapping.model.pkl'
DATASET_CON_PROB = MODEL_PATH + 'dataset_con_probs.model.pkl'
DATASET_VECTORIZER = MODEL_PATH + 'dataset_vectorizer.model.pkl'
DATASET_ENTITY_WEIGHT = MODEL_PATH + 'dataset_entity_weights.model.pkl'

ABBS_PATH = RESOURCE_PATH + 'listofabb_dataset.txt'
PHRS_PATH = RESOURCE_PATH + 'listofPhrase_dataset.txt'


def _load_models():
    print('Loading models...')
    svm_clf = joblib.load(DATASET_DETECT_MODEL)
    cv = pickle.load(open(DATASET_VECTORIZER, 'rb'))
    con_prob = pickle.load(open(DATASET_CON_PROB, 'rb'))
    entity_w = pickle.load(open(DATASET_ENTITY_WEIGHT, 'rb'))
    dataset_id = pickle.load(open(DATASET_ID_MAPPING, 'rb'))

    return {
        'detector': svm_clf,
        'dataset_mapping': dataset_id,
        'cond_probs': con_prob,
        'entity_weights': entity_w,
        'vectorizer': cv
            }


#for a given text chunk, this ranks the datasets
def _rank_dataset(sentence, con_prob, vocab, dataset_id, entity_weight=None, is_entity_weight = True):
    words = word_tokenize(sentence.lower())
    dataset_score = np.zeros((con_prob.shape[0],))

    for word in words:
        if word not in vocab.keys():
            continue

        index = vocab[word]

        if is_entity_weight:
            dataset_score = dataset_score + entity_w[index] * np.log(con_prob[:,index])
        else:
            dataset_score = dataset_score + np.log(con_prob[:,index])

    args = np.argsort(dataset_score.reshape(1,-1))[0,-5:]

    prob_list = dataset_score[args]
    id_list = np.array(dataset_id)[args]

    return id_list, prob_list


#select best k datasets
def _select_k_best(arg_list, prob_list):
    top_k = []
    s_margin = 0.2
    threshold = 1/len(prob_list) + s_margin * 1/len(prob_list)

    prob_list_mod = []
    for i in range(len(prob_list)):
        prob_list_mod.append(prob_list[i]-prob_list[0])

    #normalize
    tot = 0
    for i in prob_list_mod:
        tot += i

    if tot != 0:
        for id, i in enumerate(prob_list_mod):
            prob_list_mod[id] = prob_list_mod[id]/tot

        for i in range(len(prob_list_mod)):
            if prob_list_mod[-i-1] > threshold:
                top_k.append(arg_list[-i-1])

    return top_k, prob_list_mod[-len(top_k):]


def _filter_mentions(text, mentions, abbs_phrs_list):
    selected_mentions = []
    for mention in mentions:
        for item in abbs_phrs_list:
            if item.lower() in mention:
                selected_mentions.append(item)

    return list(set(selected_mentions))


def _make_prediction(models, metadata, parsed_pub):
    citing_pred = models['detector'].predict([parsed_pub])
    if citing_pred[0] == 0:
        return None, None

    publication_id = str(metadata['publication_id'])
    PUBLICATION_PATH = args.input_dir + TEXT_PUB_PATH + publication_id + '.txt'
    dataset_text = json.load(open(RESOURCE_PATH + 'dataset_text_dict.json', 'r'))

    sf_obj = sentence_filtering.SentenceFilterClass()
    mentions = sf_obj.final_approach(PUBLICATION_PATH)
    id_list, prob_list = _rank_dataset(' '.join(mentions), models['cond_probs'], models['vectorizer'].vocabulary_, \
        models['dataset_mapping'], entity_weight=None, is_entity_weight = False)
    id_list, prob_list = _select_k_best(id_list, prob_list)

    paper_results = []
    mention_results = []
    abbs_list = auxfun.readtoarr2(auxfun, ABBS_PATH)
    phrs_list = auxfun.readtoarr2(auxfun, PHRS_PATH)
    abbs_phrs_list = abbs_list + phrs_list
    for i, dataset_id in enumerate(id_list):
        selected_mentions = _filter_mentions(dataset_text[str(dataset_id)],
                                             mentions,
                                             abbs_phrs_list)

        paper_results.append({
            'publication_id': metadata['publication_id'],
            'data_set_id': int(id_list[i]),
            'score': round(prob_list[i],2),
            'mention_list': selected_mentions
        })

        for men in selected_mentions:
            mention_results.append({
                'publication_id': metadata['publication_id'],
                'data_set_mention' : men,
                'score': round(prob_list[i],2)
                })

    return paper_results, mention_results


def _predict(models, input_dir, output_dir):
    """ This function predict dataset citations
    """
    print('Test data: {}'.format(input_dir))
    print('Loading test data...')
    parsed_pubs_test = load_rcc_test_dataset(input_dir)
    test_list = json_from_file(args.input_dir + TEST_FILE)

    print('Running prediction...')
    predictions = []
    mention_prediction = []

    for test in tqdm(test_list, ascii=True,
                     desc='Running datasets recognition'):
        pub = parsed_pubs_test[str(test['publication_id'])]
        pred, men_pred = _make_prediction(models, test, pub)
        if pred is None:
            continue

        predictions = predictions + pred
        mention_prediction = mention_prediction + men_pred

    # Save predictions to DATASET_CITATION_OUTFILE
    json.dump(predictions,
              open(output_dir + DATASET_CITATION_OUTFILE, 'w'),
              indent=4)

    # Save extracted mentions to DATASET_MENTION_OUTFILE
    json.dump(mention_prediction,
              open(output_dir + DATASET_MENTION_OUTFILE, 'w'),
              indent=4)


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
