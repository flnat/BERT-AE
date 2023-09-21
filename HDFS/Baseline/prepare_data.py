import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils.model_selection import random_split


# Dummy function to replace sklearn tokenizer and preprocesser which only get in the way

class BagOfWords:
    def __init__(self, features: str = "./HDFS/data/parsed_logs.csv", labels: str = "./HDFS/data/labels.csv",
                 ngramm_range: int | None = None, rng_seed: int = 42):
        self.ngramm_range = ngramm_range
        self.rng_seed = rng_seed
        self.splits = [0.01, 0.19, 0.8]
        self.features = features
        self.labels = labels

        self.x_train, self.x_val, self.x_test = None, None, None
        self.y_train, self.y_val, self.y_test = None, None, None


    def fetch_data(self):
        logs = pd.read_csv(self.features)
        labels = pd.read_csv(self.labels)
        # Split train/val/test, train-set contains no anomalies.
        return random_split(parsed_data=logs, labels=labels, join_key="blk_id",
                                label_encoding=(0, 1), mode="uncontaminated", sizes=[.01, 0.19, .8])


    def preprocess(self, train, val, test, ngramm_range: tuple | None = None):
        # Dummy function to replace sklearn tokenizer and preprocessor which only get in the way

        idf_params = {
            "input": "content",
            "norm": None,
            "analyzer": lambda x: x.split(),
            "min_df": 1,
            "max_df": 1.0,
        }
        if ngramm_range is not None:
            idf_params["ngramm_range"] = ngramm_range

        vectorizer = TfidfVectorizer(**idf_params)
        vectorizer.fit(train.event_seq)

        train = vectorizer.transform(train.event_seq)
        val = vectorizer.transform(val.event_seq)
        test = vectorizer.transform(test.event_seq)

        # Cast to dense memory representation
        return train.toarray(), val.toarray(), test.toarray()


    def prepare(self):
        train, val, test = self.fetch_data()

        self.y_train, self.y_val, self.y_test = train.label, val.label, test.label
        self.x_train, self.x_val, self.x_test = self.preprocess(train, val, test)
