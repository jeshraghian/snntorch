import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="snntorch",
    version="0.0.1",
    author="Jason K. Eshraghian",
    author_email="jasonesh@umich.edu",
    description="Deep learning with spiking neural networks",
    long_description=long_description, # loaded from README.md
    long_description_content_type="text/markdown",
    url="git@github.com:jeshraghian/snntorch.git",
        # "https://github.com/jeshraghian/snntorch",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU GPL v3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)