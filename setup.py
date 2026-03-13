from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="simplistic-zipformer",
    version="0.1.0",
    author="Your Name",
    description="Implementation of Zipformer for automatic speech recognition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "einops>=0.6.0",
        "numpy<2.0.0",
    ],
)
