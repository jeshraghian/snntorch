#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.rst", encoding="utf-8") as readme_file:
    readme = readme_file.read()

# with open("HISTORY.rst") as history_file:
#     history = history_file.read()

# fmt: off
__version__ = '0.8.0'
# fmt: on

requirements = [
    "torch>=1.1.0",
    "pandas",
    "matplotlib",
    "numpy>=1.17",
    "nir",
    "nirtorch",
]


test_requirements = ["pytest>=6"]

version = __version__

setup(
    author="Jason K. Eshraghian",
    author_email="jeshragh@ucsc.edu",
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    description="Deep learning with spiking neural networks.",
    # entry_points={
    #     "console_scripts": [
    #         "snntorch=snntorch.cli:main",
    #     ],
    # },
    install_requires=requirements,
    license="MIT License",
    long_description=readme,
    include_package_data=True,
    keywords="snntorch",
    name="snntorch",
    packages=find_packages(include=["snntorch", "snntorch.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/jeshraghian/snntorch",
    version=__version__,
    zip_safe=False,
)
