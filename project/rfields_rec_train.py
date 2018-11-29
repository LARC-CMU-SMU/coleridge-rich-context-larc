import argparse

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle
from sklearn.externals import joblib


from rcc_conf import RAND_SEED
from rcc_utils import json_from_file


class TitleAbstractExtractor(BaseEstimator, TransformerMixin):
    """Extract the paper titles and abstract fields.
    """

    def fit(self, x, y=None):
        return self

    def transform(self, parsed_pubs):
        # construct object dtype array with two columns
        # first column = 'title' and second column = 'authors'
        features = np.empty(shape=(len(parsed_pubs), 2), dtype=object)
        for i, pub in enumerate(parsed_pubs):
            features[i, 0] = pub['title']
            features[i, 1] = pub['long_abstract'] if pub['long_abstract'] is not None else pub['short_abstract']
        return features


def _generate_title_model(data_train, labels_train,
                          output_file):
    print('Training reference title model for recommendation...')
    title_clf = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
                          ('clf', ComplementNB())])
    title_clf.fit(data_train, labels_train)
    joblib.dump(title_clf, output_file, compress='zlib')
    print('Model file {} saved.'.format(output_file))


def _generate_pub_model(data_train, labels_train,
                        output_file):
    print('Training research field recommendation model...')
    svm = SGDClassifier(loss='hinge', penalty='l2',
                        alpha=1e-4,
                        random_state=RAND_SEED,
                        max_iter=2000, tol=1e-3)
    rfields_rec = Pipeline([
        ('feature_set_extractor', TitleAbstractExtractor()),
        ('union', ColumnTransformer(
            [
                ('title', TfidfVectorizer(ngram_range=(1, 2)), 0),
                ('abstract', TfidfVectorizer(ngram_range=(1, 2)), 1)
            ],
            transformer_weights={
                'title': 1.1,
                'abstract': 1.0
            }
        )),
        ('clf', svm)])
    rfields_rec.fit(data_train, labels_train)
    joblib.dump(rfields_rec, output_file, compress='zlib')
    print('Model file {} saved.'.format(output_file))


def main(args):
    """ For research field recommendation, we build two classifiers:
        1. Research fields classifier from title only
        2. Research fields classifier from title and abstract
    """
    rfields = shuffle(json_from_file(args.input), random_state=RAND_SEED)
    titles = [d['title'] for d in rfields]
    levels = ['l1', 'l2', 'l3']
    for level in levels:
        label = [d[level] for d in rfields]
        _generate_title_model(titles, label,
                              args.output + '.rt.' + level)

    filtered_rfields = [r for r in rfields
                        if r['long_abstract'] is not None or
                        r['short_abstract'] is not None]
    for level in levels:
        label = [d[level] for d in filtered_rfields]
        _generate_pub_model(filtered_rfields, label,
                            args.output + '.pub.' + level)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train models for \
                                     research fields recommendation')
    parser.add_argument('--input', type=str, required=True,
                        help='Filename of input dataset to train the models')
    parser.add_argument('--output', type=str, required=True,
                        help='Filename of model output')
    args = parser.parse_args()
    main(args)
