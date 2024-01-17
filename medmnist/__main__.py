import medmnist
from medmnist.info import INFO, DEFAULT_ROOT


def available():
    """List all available datasets."""
    print(f"MedMNIST v{medmnist.__version__} @ {medmnist.HOMEPAGE}")

    print("All available datasets:")
    for key in INFO.keys():
        if key.endswith("mnist"):
            print(
                f"\t{key:<15} | {INFO[key]['python_class']:<15} | Size: 28 (default), 64, 128, 224."
            )
        else:
            print(
                f"\t{key:<15} | {INFO[key]['python_class']:<15} | Size: 28 (default), 64."
            )


def download(size=None, root=DEFAULT_ROOT):
    """Download all available datasets."""

    if size is None:
        sizes = [28]
    elif size in [28, 64, 128, 224]:
        sizes = [size]
    elif size == "all":
        sizes = [28, 64, 128, 224]
    else:
        raise ValueError(f"Invalid size {size}.")
    
    for size in sizes:
        for key in INFO.keys():
            available_sizes = getattr(medmnist, INFO[key]["python_class"]).available_sizes
            if size in available_sizes:
                print(
                    f"Downloading {key:<15} | {INFO[key]['python_class']:<15} | Size: {size}"
                )
                _ = getattr(medmnist, INFO[key]["python_class"])(
                    split="train", download=True, root=root, size=size
                )
            else:
                print(
                    f"Size {size} not avaiable for {key:<15} | {INFO[key]['python_class']:<15}"
                )


def clean(root=DEFAULT_ROOT):
    """Delete all downloaded npz from root."""
    import os
    from glob import glob

    for path in glob(os.path.join(root, "*mnist*.npz")):
        os.remove(path)


def info(flag):
    """Print the dataset details given a subset flag."""

    import json

    print(json.dumps(INFO[flag], indent=4))


def save(flag, folder, postfix="png", root=DEFAULT_ROOT, download=False, size=None):
    """Save the dataset as standard figures, which could be used for AutoML tools, e.g., Google AutoML Vision."""

    for split in ["train", "val", "test"]:
        print(f"Saving {flag} {split}...")
        dataset = getattr(medmnist, INFO[flag]["python_class"])(
            split=split, download=download, root=root, size=size
        )
        dataset.save(folder, postfix)


def evaluate(path):
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
    medmnist.Evaluator.parse_and_evaluate(path)


def test(save_folder="tmp/", root=DEFAULT_ROOT):
    """For developmemnt only."""

    import os
    from glob import glob

    print("Testing `available()`...")
    available()

    print("Testing `download()`...")
    download(root=root)

    for key in INFO.keys():
        if key.endswith("mnist"):
            postfix = "jpg"
            # continue
        else:
            postfix = "gif"
            # continue

        print(f"Testing `info({key})`....")

        info(key)

        print(f"Testing `save({key})`....")

        save(key, save_folder, postfix=postfix, root=root)
        save(key, save_folder, postfix=postfix, root=root, download=True, size=64)

        for split in ["train", "val", "test"]:
            print(f"Testing `Evaluator({key}, {split})`....")

            # for 28
            evaluator = medmnist.Evaluator(key, split)
            dummy = evaluator.get_dummy_prediction()
            print("Evaluation using `evaluator.evaluate()` for 28")
            print(evaluator.evaluate(dummy, save_folder))

            dummy_evaluation_file = glob(
                os.path.join(save_folder, f"{key}_{split}*.csv")
            )[0]
            print("dummy @28", dummy_evaluation_file)

            print("Evaluation using `Evaluator.parse_and_evaluate()` for 28")
            medmnist.Evaluator.parse_and_evaluate(dummy_evaluation_file, run="dummy")

            # for 64
            evaluator = medmnist.Evaluator(key, split, size=64)
            dummy = evaluator.get_dummy_prediction()

            print("Evaluation using `evaluator.evaluate()` for 64")
            print(evaluator.evaluate(dummy, save_folder))

            dummy_evaluation_file = glob(
                os.path.join(save_folder, f"{key}_64_{split}*.csv")
            )[0]
            print("dummy @64", dummy_evaluation_file)

            print("Evaluation using `Evaluator.parse_and_evaluate()` for 64")
            medmnist.Evaluator.parse_and_evaluate(dummy_evaluation_file, run="dummy")

            print(f"Testing `montage()` for {key}....")
            dataset = getattr(medmnist, INFO[key]["python_class"])(
                split=split, root=root, size=64
            )
            assert len(dataset) == INFO[key]["n_samples"][split]

            n_channels = INFO[key]["n_channels"]

            _, *shape = dataset.imgs.shape
            if n_channels == 3:
                # assert shape == [28, 28, 3]
                assert shape == [64, 64, 3]
            else:
                assert n_channels == 1
                # assert shape == [28]*2 or shape == [28]*3
                assert shape == [64]*2 or shape == [64]*3

            dataset.montage(save_folder=save_folder, replace=True)

    # clean(root)


if __name__ == "__main__":
    import fire

    fire.Fire()
