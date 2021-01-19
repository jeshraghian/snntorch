import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

VERSION = "0.0.1"

setuptools.setup(
    name="snntorch",
    version=VERSION,
    author="Jason K. Eshraghian",
    author_email="jasonesh@umich.edu",
    description="Deep learning with spiking neural networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jeshraghian/snntorch",
    download_url="https://github.com/jeshraghian/snntorch/tarball/{}".format(VERSION),
    license="GPL-3.0",
    packages=setuptools.find_packages(),
    install_requires=[
        "torch>=1.2.0",
        "pandas",
        "matplotlib",
        "math",
        "celluloid",
        "numpy>=1.17"
    ],
    extras_require={
        "dev": [
            "pytest>=3.7",
        ]
    },
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    keywords="snntorch pytorch machine learning",
    python_requires='>=3.6',
)
