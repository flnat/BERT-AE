from typing import Optional

import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split


def _normalize_sizes(sizes: list[float]) -> tuple[list[float], list[float]]:
    if sum(sizes) != 1:
        # Normalize split size if not already unit norm
        old_sizes = sizes
        sum_sizes = sum(sizes)

        sizes = [old_size / sum_sizes for old_size in old_sizes]

    val_test_split = [size / sum(sizes[1:]) for size in sizes[1:]]

    return sizes, val_test_split


def clean_split(parsed_data: DataFrame, labels: Optional[DataFrame], join_key: Optional[str],
                label_encoding: tuple,
                sizes: list[float], rng_seed: int = 42) -> tuple[DataFrame, ...]:
    """
    Perform train/val/test split with no anomalies in the train set
    :param parsed_data: DataFrame containing the feautures --> parsed logs.
    :param labels: DataFrame containing the labels.
    :param join_key: Key with which to join the features and labels.
    :param label_encoding: Encoding of the labels given as iterable of length 2. Label 0 is normal, Label 1 anomalous.
    :param sizes: Split ratio for the train/val/test split
    :param rng_seed: Seed for the random split
    :return: Train/Val/Test Set as iterable of DataFrames
    """

    sizes, val_test_split = _normalize_sizes(sizes)
    parsed_data = parsed_data. \
        merge(right=labels, left_on=join_key, right_on=join_key, how="inner")

    normal_data = parsed_data[parsed_data["label"] == label_encoding[0]]
    abnormal_data = parsed_data[parsed_data["label"] == label_encoding[1]]

    train, rest = train_test_split(normal_data, train_size=sizes[0], random_state=rng_seed)
    total_data = pd.concat([rest, abnormal_data])

    val, test = train_test_split(total_data, train_size=val_test_split[0], random_state=rng_seed)
    return train, val, test


def random_split(parsed_data: DataFrame, sizes: list[float],
                 labels: DataFrame | None = None, join_key: str | None = None,
                 label_encoding: tuple | None = None, mode: str | None = None, rng_seed: int = 42,
                 ) -> tuple[DataFrame, ...]:
    supervised_args = [join_key, label_encoding, mode]

    if labels is None:
        return random_split_unsupervised(parsed_data=parsed_data, sizes=sizes, rng_seed=rng_seed)
    # Check if supervised arguments are all coherent, i.e. if labels are given all necessary args are not null
    elif all(supervised_args):
        # Check if all required arguments for a supervised split have been provided, else throw an exception
        match mode:
            case "uncontaminated":
                return clean_split(parsed_data=parsed_data, sizes=sizes, labels=labels, join_key=join_key,
                                   label_encoding=label_encoding, rng_seed=rng_seed)
            case "stratified":
                return random_split_supervised(parsed_data=parsed_data, sizes=sizes, labels=labels, join_key=join_key,
                                               stratify=True, rng_seed=rng_seed)
            case _:
                return random_split_supervised(parsed_data=parsed_data, sizes=sizes, labels=labels, join_key=join_key,
                                               stratify=False, rng_seed=rng_seed)
    else:
        raise ValueError


def random_split_supervised(parsed_data: DataFrame, sizes: list[float],
                            labels: DataFrame | None = None, join_key: str | None = None,
                            stratify: bool = False, rng_seed: int = 42
                            ) -> tuple[DataFrame, ...]:
    sizes, val_test_split = _normalize_sizes(sizes)
    parsed_data = parsed_data. \
        merge(right=labels, left_on=join_key, right_on=join_key, how="inner")

    if stratify:
        train, rest = train_test_split(parsed_data, train_size=sizes[0], stratify=parsed_data["label"],
                                       random_state=rng_seed)
        val, test = train_test_split(rest, train_size=val_test_split[0], stratify=rest["label"],
                                     random_state=rng_seed)
    else:
        train, rest = train_test_split(parsed_data, train_size=sizes[0], random_state=rng_seed)
        val, test = train_test_split(rest, train_size=val_test_split[0], random_state=rng_seed)

    return train, val, test


def random_split_unsupervised(parsed_data: DataFrame, sizes: list[float], rng_seed: int = 42, ) -> tuple[
    DataFrame, ...]:
    sizes, val_test_split = _normalize_sizes(sizes)

    train, rest = train_test_split(parsed_data, train_size=sizes[0], random_state=rng_seed)
    val, test = train_test_split(rest, train_size=val_test_split[0], random_state=rng_seed)

    return train, val, test
