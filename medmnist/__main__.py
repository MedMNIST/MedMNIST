from medmnist.info import __version__, HOMEPAGE, INFO, DEFAULT_ROOT
import medmnist

def available():
    '''List all available datasets.'''
    print(f"MedMNIST v{__version__} @ {HOMEPAGE}")

    print("All available datasets:")
    for key in INFO.keys():
        print("\t"+key)


def download(root=DEFAULT_ROOT):
    '''Download all available datasets.'''
    for key in INFO.keys():
        print(f"Downloading {key}...")
        _ = getattr(medmnist, INFO[key]['python_class'])(
            split="train", root=root)


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


def test():
    '''For developmemnt only.'''

    available()

    download()

    clean()

    for key in INFO.keys():
        info(key)


if __name__ == "__main__":
    import fire
    fire.Fire()
