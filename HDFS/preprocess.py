import logging
import os.path
import sys
import time

import pandas as pd
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig


def fetch_data() -> (pd.DataFrame, pd.DataFrame):
    logs = pd.read_csv("./data/HDFS.log", delimiter=";", header=None, names=["log_msg"])
    logs["blk_id"] = logs.log_msg.str.extract("(blk_-?\d+)")
    logs["blk_id"] = logs["blk_id"].str.strip()

    labels = pd.read_csv("./data/anomaly_label.csv")
    labels.rename(columns={"BlockId": "blk_id", "Label": "label"}, inplace=True)
    labels.loc[labels.label == "Normal", "label"] = 0
    labels.loc[labels.label == "Anomaly", "label"] = 1
    labels.label = labels.label.astype("int")

    return logs, labels


def extract_template(features: pd.DataFrame):
    logger = logging.getLogger("logparsing")
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')

    config = TemplateMinerConfig()
    config.load(os.path.join(os.getcwd(), "drain.ini"))
    log_parser = TemplateMiner(config=config)

    start_time = time.time()
    batch_start_time = start_time
    batch_size = 100000

    parsed_logs = []
    for idx, line in features.iterrows():

        line_count = idx + 1

        msg = line["log_msg"]
        msg = msg.rstrip()
        msg = msg.partition(": ")[2]
        result = log_parser.add_log_message(msg)
        parsed_logs.append({
            "blk_id": line["blk_id"],
            "event_id": result["cluster_id"]
        })
        if line_count % batch_size == 0:
            time_took = time.time() - batch_start_time
            rate = batch_size / time_took
            logger.info(f"Processing line: {line_count}, rate {rate:.1f} lines/sec, "
                        f"{len(log_parser.drain.clusters)} clusters so far.")
            batch_start_time = time.time()

    time_took = time.time() - start_time
    rate = line_count / time_took
    logger.info(
        f"--- Done processing file in {time_took:.2f} sec. Total of {line_count} lines, rate {rate:.1f} lines/sec, "
        f"{len(log_parser.drain.clusters)} clusters")

    parsed_logs = pd.DataFrame(parsed_logs)
    parsed_logs["event_id"] = parsed_logs["event_id"].astype("category").cat.codes
    return parsed_logs


def group_sequences(features: pd.DataFrame) -> pd.DataFrame:
    features = features.groupby("blk_id")["event_id"]. \
        apply(list). \
        reset_index(name="event_seq")

    features["event_seq"] = features["event_seq"].astype("string")

    # TorchText Vocab & Sklearn CountVectorizer except a Sequence represented as a string.
    # Therefore, we concatenate the events to a single string.
    features["event_seq"] = features["event_seq"].str.replace(r"[\[\],]+", "", regex=True)

    return features


if __name__ == "__main__":
    logs, labels = fetch_data()
    logs = extract_template(logs)
    logs = group_sequences(logs)
    logs.to_csv("./data/parsed_logs.csv", index=False)
    labels.to_csv("./data/labels.csv", index=False)
