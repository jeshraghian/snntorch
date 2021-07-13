#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

# fmt: off
__version__ = '0.4.0'
# fmt: on

requirements = [
    "torch>=1.2.0",
    "pandas",
    "matplotlib",
    "celluloid",
    "numpy>=1.17",
]

setup_requirements = [
    "pytest-runner",
]

test_requirements = ["pytest>=3"]

version = __version__

setup(
    author="Jason K. Eshraghian",
    author_email="jasonesh@umich.edu",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
    ],
    description="Deep learning with spiking neural networks.",
    # entry_points={
    #     "console_scripts": [
    #         "snntorch=snntorch.cli:main",
    #     ],
    # },
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="snntorch",
    name="snntorch",
    packages=find_packages(include=["snntorch", "snntorch.*"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/jeshraghian/snntorch",
    version=__version__,
    zip_safe=False,
)
