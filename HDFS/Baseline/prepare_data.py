import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline

from src.utils.model_selection import clean_split


class BagOfWords:
    def __init__(self, ngramm_range: int | None = None, rng_seed: int = 42):

        self.ngramm_range = ngramm_range
        self.rng_seed = rng_seed
        self.splits = [0.01, 0.19, 0.8]

    def fetch_data(self):

        logs = pd.read_csv("./data/parsed_logs.csv")
        labels = pd.read_csv("./data/labels.csv")
        # Split train/val/test, train-set contains no anomalies.
        train, val, test = clean_split(logs, labels, join_key="blk_id", label_encoding=(0, 1), sizes=[0.01, .19, .8])

        return train, val, test

    def preprocess(self, train, val, test, ngramm_range: tuple | None = None):

        if ngramm_range is not None:
            pipeline = Pipeline(steps=[
                ("CountVectorizer", CountVectorizer(ngram_range=ngramm_range, token_pattern="\d")),
                ("TF-IDF", TfidfTransformer())
            ])
            transformer = ColumnTransformer(
                [
                    ("bow", pipeline, "event_seq")
                ], sparse_threshold=0)

        else:
            pipeline = Pipeline(steps=[
                ("CountVectorizer", CountVectorizer(ngram_range=ngramm_range, token_pattern="\d")),
                ("TF-IDF", TfidfTransformer())
            ])
            transformer = ColumnTransformer(
                [
                    ("bow", pipeline, "event_seq")
                ], sparse_threshold=0)

        pipe_model = transformer.fit(train)

        train = pipe_model.transform(train)
        test = pipe_model.transform(test)
        val = pipe_model.transform(val)

        return train, val, test

    def prepare(self):

        train, val, test = self.fetch_data()

        self.y_train, self.y_val, self.y_test = train.label, val.label, test.label

        self.x_train, self.x_val, self.x_test = self.preprocess(train, val, test, ngramm_range=self.ngramm_range)
