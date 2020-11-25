from setuptools import setup, find_packages

import medmnist


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


setup(
    name='MedMNIST',
    version=medmnist.__version__,
    url='https://github.com/MedMNIST/MedMNIST',
    license='Apache-2.0 License',
    author='Jiancheng Yang and Rui Shi and Bingbing Ni',
    author_email='jekyll4168@sjtu.edu.cn',
    description='MedMNIST Classification Decathlon: A Lightweight AutoML Benchmark for Medical Image Analysis',
    long_description=readme(),
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'pandas',
        'scikit-learn',
        'tqdm'
    ],
    zip_safe=True
)
