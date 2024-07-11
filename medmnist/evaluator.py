import os
import warnings
import numpy as np
import pandas as pd
from collections import namedtuple
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

from medmnist.info import INFO, DEFAULT_ROOT

Metrics = namedtuple("Metrics", ["AUC", "ACC"])


class Evaluator:
    def __init__(self, flag, split, size=None, root=DEFAULT_ROOT):
        self.flag = flag
        self.split = split

        if (size is None) or (size == 28):
            self.size = 28
            self.size_flag = ""
        else:
            self.size = size
            self.size_flag = f"_{size}"

        if root is not None and os.path.exists(root):
            self.root = root
        else:
            raise RuntimeError(
                "Failed to setup the default `root` directory. "
                + "Please specify and create the `root` directory manually."
            )

        npz_file = np.load(os.path.join(self.root, f"{self.flag}{self.size_flag}.npz"))

        self.info = INFO[self.flag]

        if self.split in ["train", "val", "test"]:
            self.labels = npz_file[f"{self.split}_labels"]
        else:
            raise ValueError

    def evaluate(self, y_score, save_folder=None, run=None):
        assert y_score.shape[0] == self.labels.shape[0]

        task = self.info["task"]
        auc = getAUC(self.labels, y_score, task)
        acc = getACC(self.labels, y_score, task)
        metrics = Metrics(auc, acc)

        if save_folder is not None:
            path = os.path.join(
                save_folder, self.get_standard_evaluation_filename(metrics, run)
            )
            pd.DataFrame(y_score).to_csv(path, header=None)
        return metrics

    def get_standard_evaluation_filename(self, metrics, run=None):
        eval_txt = "_".join([f"[{k}]{v:.3f}" for k, v in zip(metrics._fields, metrics)])

        if run is None:
            import time

            run = time.time()

        ret = f"{self.flag}{self.size_flag}_{self.split}_{eval_txt}@{run}.csv"
        return ret

    def get_dummy_prediction(self):
        """Return a dummy prediction of correct shape."""
        task = self.info["task"]
        if task == "multi-class" or task == "ordinal-regression":
            num_classes = self.labels.max()
            dummy = np.random.rand(self.labels.shape[0], num_classes)
            dummy = dummy / dummy.sum(axis=-1, keepdims=True)
        else:
            dummy = np.random.rand(*self.labels.shape)
        return dummy

    @classmethod
    def parse_and_evaluate(cls, path, run=None):
        """Parse and evaluate a standard result file.

        A standard result file is named as:
            {flag}{size_flag}_{split}|*|.csv (|*| can be anything)

        In a standard evaluation file, we also save the metrics in the filename:
            {flag}{size_flag}_{split}_[AUC]{auc:.3f}_[ACC]{acc:.3f}@{run}.csv

        Here, `size_flag` is blank for 28 images, and `_size` for larger images, e.g., "_64".

        In result/evaluation file, each line is (dataset index,float prediction).

        For instance,
        octmnist_test_[AUC]0.672_[ACC]0.892@3.csv
            0,0.125,0.275,0.5,0.2
            1,0.5,0.125,0.275,0.2
        """
        folder, filename = os.path.split(path)

        flag, split_, *_ = filename.split("_")
        size = None
        if split_.startswith("train"):
            split = "train"
        elif split_.startswith("val"):
            split = "val"
        elif split_.startswith("test"):
            split = "test"
        else:
            flag, size, split_, *_ = filename.split("_") # for size_flag
            if split_.startswith("train"):
                split = "train"
            elif split_.startswith("val"):
                split = "val"
            elif split_.startswith("test"):
                split = "test"
            else:
                raise ValueError

        if run is None:
            assert "@" in filename
            run = filename.split("@")[-1].split(".")[0]

        evaluator = cls(flag, split, size=size)

        df = pd.read_csv(path, index_col=0, header=None)
        y_score = df.sort_index().values

        metrics = evaluator.evaluate(y_score, folder, run)
        print(metrics)

        return metrics


def getAUC(y_true, y_score, task):
    """AUC metric.
    :param y_true: the ground truth labels, shape: (n_samples, n_labels) or (n_samples,) if n_labels==1
    :param y_score: the predicted score of each class,
    shape: (n_samples, n_labels) or (n_samples, n_classes) or (n_samples,) if n_labels==1 or n_classes==1
    :param task: the task of current dataset
    """
    y_true = y_true.squeeze()
    y_score = y_score.squeeze()

    if task == "multi-label, binary-class":
        auc = 0
        for i in range(y_score.shape[1]):
            label_auc = roc_auc_score(y_true[:, i], y_score[:, i])
            auc += label_auc
        ret = auc / y_score.shape[1]
    elif task == "binary-class":
        if y_score.ndim == 2:
            y_score = y_score[:, -1]
        else:
            assert y_score.ndim == 1
        ret = roc_auc_score(y_true, y_score)
    else:
        auc = 0
        for i in range(y_score.shape[1]):
            y_true_binary = (y_true == i).astype(float)
            y_score_binary = y_score[:, i]
            auc += roc_auc_score(y_true_binary, y_score_binary)
        ret = auc / y_score.shape[1]

    return ret


def getACC(y_true, y_score, task, threshold=0.5):
    """Accuracy metric.
    :param y_true: the ground truth labels, shape: (n_samples, n_labels) or (n_samples,) if n_labels==1
    :param y_score: the predicted score of each class,
    shape: (n_samples, n_labels) or (n_samples, n_classes) or (n_samples,) if n_labels==1 or n_classes==1
    :param task: the task of current dataset
    :param threshold: the threshold for multilabel and binary-class tasks
    """
    y_true = y_true.squeeze()
    y_score = y_score.squeeze()

    if task == "multi-label, binary-class":
        y_pre = y_score > threshold
        acc = 0
        for label in range(y_true.shape[1]):
            label_acc = accuracy_score(y_true[:, label], y_pre[:, label])
            acc += label_acc
        ret = acc / y_true.shape[1]
    elif task == "binary-class":
        if y_score.ndim == 2:
            y_score = y_score[:, -1]
        else:
            assert y_score.ndim == 1
        ret = accuracy_score(y_true, y_score > threshold)
    else:
        ret = accuracy_score(y_true, np.argmax(y_score, axis=-1))

    return ret


def save_results(y_true, y_score, outputpath):
    """Save ground truth and scores
    :param y_true: the ground truth labels, shape: (n_samples, n_classes) for multi-label, and (n_samples,) for other tasks
    :param y_score: the predicted score of each class, shape: (n_samples, n_classes)
    :param outputpath: path to save the result csv

    """

    warnings.DeprecationWarning(
        "Only kept for backward compatiblility."
        + "Please use `Evaluator` API instead. "
    )
    idx = []

    idx.append("id")

    for i in range(y_true.shape[1]):
        idx.append("true_%s" % (i))
    for i in range(y_score.shape[1]):
        idx.append("score_%s" % (i))

    df = pd.DataFrame(columns=idx)
    for id in range(y_score.shape[0]):
        dic = {}
        dic["id"] = id
        for i in range(y_true.shape[1]):
            dic["true_%s" % (i)] = y_true[id][i]
        for i in range(y_score.shape[1]):
            dic["score_%s" % (i)] = y_score[id][i]

        df_insert = pd.DataFrame(dic, index=[0])
        df = df.append(df_insert, ignore_index=True)

    df.to_csv(outputpath, sep=",", index=False, header=True, encoding="utf_8_sig")
