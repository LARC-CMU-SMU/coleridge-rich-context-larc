import argparse
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle
from sklearn.externals import joblib

from rcc_conf import RAND_SEED
from rcc_utils import load_rcc_train_dataset


class FeatureGroupExtractor(BaseEstimator, TransformerMixin):
    """Extract the reference titles and research methods fields.
       Takes a list of parsed publication and produces a dict of sequences.
       Keys are `ref_titles` and `r_methods`.
    """

    def fit(self, x, y=None):
        return self

    def _gen_ref_titles(self, references):
        titles = [r['title'] for r in references]
        return ' '.join(titles)

    def _gen_rmethods_tf_dict(self, rmethods, min_sup=3):
        if rmethods is None:
            return {}
        result = {}
        for method in rmethods:
            if method['count'] < min_sup:
                continue
            result[method['id']] = method['count']
        return result

    def transform(self, parsed_pubs):
        # construct object dtype array with two columns
        # first column = 'ref_titles' and second column = 'r_methods'
        features = np.empty(shape=(len(parsed_pubs), 2), dtype=object)
        for i, pub in enumerate(parsed_pubs):
            features[i, 0] = self._gen_ref_titles(pub['metadata']['references'])
            features[i, 1] = self._gen_rmethods_tf_dict(pub['rmethods'])
        return features


def _generate_train_model(data_train, labels_train,
                          output_file):
    print('Training dataset detection classification model...')
    features = ColumnTransformer(
        [
            ('ref_titles', TfidfVectorizer(ngram_range=(1, 2)), 0),
            ('rmethods', Pipeline([
                ('rm_vect', DictVectorizer()),
                ('rm_tfidf', TfidfTransformer())
            ]), 1),
        ],
        transformer_weights={
            'ref_titles': 1.0,
            'rmethods': 1.0
        }
    )
    svm = SGDClassifier(loss='hinge', penalty='l2',
                        alpha=1e-4,
                        random_state=RAND_SEED,
                        max_iter=1000, tol=1e-3)
    svm_clf = Pipeline([
        ('feature_set_extractor', FeatureGroupExtractor()),
        ('union', features),
        ('clf', svm)])
    svm_clf.fit(data_train, labels_train)
    joblib.dump(svm_clf, output_file, compress='zlib')
    print('Model file {} saved.'.format(output_file))


def main(args):
    """ This script generates training model dataset citation detection.
        We form the problem as binary classification: \
        negative class for publications with no dataset citation, \
        and positive class for publications with dataset citation.
        We combine tfidf reference titles features with research methods tfidf.
    """
    parsed_train = load_rcc_train_dataset(args.input_dir)
    labels_train = [p['citing'] for p in parsed_train]
    parsed_train, y_train = shuffle(parsed_train, labels_train,
                                    random_state=RAND_SEED)
    _generate_train_model(parsed_train, y_train, args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train classifier model for \
                                     dataset detection.')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Folder containing parsed publications \
                        to train the model.')
    parser.add_argument('--output', type=str, required=True,
                        help='Filename of model output')
    args = parser.parse_args()
    main(args)
