import logging

from setuptools import find_packages, setup

# Setup logging
logging.basicConfig(level=logging.INFO)

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="r3",
    version="1.0.0",
    author="David Anugraha",
    author_email="david.anugraha@gmail.com",
    description="R3: Robust Rubric-Agnostic Reward Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Apache 2.0 License",
    url="https://github.com/rubricreward/r3",
    project_urls={
        "Bug Tracker": "https://github.com/rubricreward/r3/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    install_requires=[
        "vllm>=0.8.5",
        "llama-factory>=0.9.2",
        "datasets",
        "scipy",
        "pandas",
    ],
    package_dir={"": "src"},
    packages = find_packages("src"),
    python_requires=">=3.10",
)
