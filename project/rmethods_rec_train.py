import argparse
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle
from sklearn.externals import joblib

from rcc_conf import RAND_SEED
from rcc_utils import json_from_file


class RMethodContextExtractor(BaseEstimator, TransformerMixin):
    """Extract title and contents features.
    """

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        # construct object dtype array with two columns
        # first column = 'title' and second column = 'contents'
        features = np.empty(shape=(len(data), 2), dtype=object)
        for i, d in enumerate(data):
            features[i, 0] = d['title']
            features[i, 1] = d['contents']
        return features


def _generate_train_model(data_train, labels_train,
                          output_file):
    print('Training research method model for recommendation...')
    features = ColumnTransformer(
        [
            ('title', TfidfVectorizer(ngram_range=(1, 2)), 0),
            ('contents', TfidfVectorizer(ngram_range=(1, 1)), 1),
        ],
        transformer_weights={
            'title': 1.0,
            'contents': 1.0
        }
    )
    sgd = SGDClassifier(loss='log', penalty='l2',
                        alpha=1e-4,
                        random_state=RAND_SEED,
                        max_iter=2000, tol=1e-3)
    pipeline = Pipeline([
        ('feature_set_extractor', RMethodContextExtractor()),
        ('union', features),
        ('clf', sgd)])
    pipeline.fit(data_train, labels_train)
    joblib.dump(pipeline, output_file, compress='zlib')
    print('Model file {} saved.'.format(output_file))


def main(args):
    rmethod_ctx_train = json_from_file(args.input)
    rmethod_ctx_train = [d for d in rmethod_ctx_train
                         if d['title'] is not None]
    labels_train = [d['method'] for d in rmethod_ctx_train]
    rmethod_ctx_train, labels_train = shuffle(rmethod_ctx_train, labels_train,
                                              random_state=RAND_SEED)
    _generate_train_model(rmethod_ctx_train, labels_train, args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train classifier model for \
                                     research method recommendation.')
    parser.add_argument('--input', type=str, required=True,
                        help='Filename of input dataset to train the models.')
    parser.add_argument('--output', type=str, required=True,
                        help='Filename of model output')
    args = parser.parse_args()
    main(args)
