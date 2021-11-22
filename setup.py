from setuptools import setup, find_packages

import medmnist


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


README = readme()


setup(
    name='medmnist',
    version=medmnist.__version__,
    url=medmnist.HOMEPAGE,
    license='Apache-2.0 License',
    author='MedMNIST Team',
    author_email='jekyll4168@sjtu.edu.cn',
    python_requires=">=3.6.0",
    description='MedMNIST: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification',
    long_description=readme(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "scikit-image",
        "tqdm",
        "Pillow",
        "fire",
        "torch",
        "torchvision"
    ],
    zip_safe=True,
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3 :: Only"
    ]
)
