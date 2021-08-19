import medmnist
from medmnist.info import INFO, DEFAULT_ROOT


def available():
    '''List all available datasets.'''
    print(f"MedMNIST v{medmnist.__version__} @ {medmnist.HOMEPAGE}")

    print("All available datasets:")
    for key in INFO.keys():
        print("\t"+key)


def download(root=DEFAULT_ROOT):
    '''Download all available datasets.'''
    for key in INFO.keys():
        print(f"Downloading {key}...")
        _ = getattr(medmnist, INFO[key]['python_class'])(
            split="train", root=root, download=True)


def clean(root=DEFAULT_ROOT):
    '''Delete all downloaded npz from root.'''
    import os
    from glob import glob

    for path in glob(os.path.join(root, "*mnist*.npz")):
        os.remove(path)


def info(flag):
    '''Print the dataset details given a subset flag.'''
    from pprint import pprint

    pprint(INFO[flag])


def save(flag, folder, postfix="png", root=DEFAULT_ROOT):
    '''Save the dataset as standard figures, which could be used for AutoML tools, e.g., Google AutoML Vision.'''
    print(f"Saving {flag} train...")
    train_dataset = getattr(medmnist, INFO[flag]['python_class'])(
        split="train", root=root)
    train_dataset.save(folder, postfix)

    print(f"Saving {flag} val...")
    val_dataset = getattr(medmnist, INFO[flag]['python_class'])(
        split="val", root=root)
    val_dataset.save(folder, postfix)

    print(f"Saving {flag} test...")
    test_dataset = getattr(medmnist, INFO[flag]['python_class'])(
        split="test", root=root)
    test_dataset.save(folder, postfix)


def test(save_folder="tmp/", root=DEFAULT_ROOT):
    '''For developmemnt only.'''

    available()

    download(root)

    for key in INFO.keys():
        print(f"Verifying {key}....")

        info(key)

        train_dataset = getattr(medmnist, INFO[key]['python_class'])(
            split="train", root=root)
        assert len(train_dataset) == INFO[key]["n_samples"]["train"]

        val_dataset = getattr(medmnist, INFO[key]['python_class'])(
            split="val", root=root)
        assert len(val_dataset) == INFO[key]["n_samples"]["val"]

        test_dataset = getattr(medmnist, INFO[key]['python_class'])(
            split="test", root=root)
        assert len(test_dataset) == INFO[key]["n_samples"]["test"]

        n_channels = INFO[key]["n_channels"]

        _, *shape = train_dataset.img.shape
        if n_channels == 3:
            assert shape == [28, 28, 3]
        else:
            assert n_channels == 1
            assert shape == [28, 28] or shape == [28, 28, 28]

        if save_folder != "null":
            try:
                train_dataset.montage(save_folder=save_folder)
            except NotImplementedError:
                print(f"{key} `montage` method not implemented.")
            
            try:
                save(key, save_folder, postfix=".jpg", root=root)
            except:
                print(f"{key} `save` method not implemented.")
            
    # clean(root)


if __name__ == "__main__":
    import fire
    fire.Fire()
