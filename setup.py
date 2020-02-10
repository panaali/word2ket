"""Setup for the word2ket package."""

import setuptools


with open('README.md') as f:
    README = f.read()

setuptools.setup(
    author="Ali Panahi",
    author_email="panaali@gmail.com",
    name='word2ket',
    license="BSD-3-Clause",
    description='word2ket is an effiecient embedding layer for PyTorch that is inspired by Quantum Entanglement.',
    version='0.0.2',
    long_description=README,
    long_description_content_type="text/markdown",
    url='https://github.com/panaali/word2ket',
    packages=setuptools.find_packages(),
    python_requires=">=3.5",
    install_requires=['torch', 'gpytorch'],
    classifiers=[
        # Trove classifiers
        # (https://pypi.python.org/pypi?%3Aaction=list_classifiers)
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
)