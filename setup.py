import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Basic_NN",
    version="0.0.1",
    author="Sridhar Adhikarla",
    author_email="adhikarla2sridhar@yahoo.com",
    description="A simple package to help you learn to implement NN",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sridhar1029/Basic_NN",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)