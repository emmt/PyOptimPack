import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="optimpack-emmt", # Replace with your own username
    version="0.0.1",
    author="Éric Thiébaut",
    author_email="eric.thiebaut@univ-lyon1.fr",
    description="Optimization methods for large scale problems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/emmt/PyOptimPack",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 2",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=2.7',
)
