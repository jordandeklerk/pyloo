import codecs
import os
import re

from setuptools import find_packages, setup

PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
REQUIREMENTS_FILE = os.path.join(PROJECT_ROOT, "requirements.txt")
REQUIREMENTS_DEV_FILE = os.path.join(PROJECT_ROOT, "requirements-dev.txt")
REQUIREMENTS_TEST_FILE = os.path.join(PROJECT_ROOT, "requirements-test.txt")
README_FILE = os.path.join(PROJECT_ROOT, "README.md")
VERSION_FILE = os.path.join(PROJECT_ROOT, "src", "pyloo", "__init__.py")


def get_requirements():
    with codecs.open(REQUIREMENTS_FILE) as buff:
        return buff.read().splitlines()


def get_requirements_dev():
    with codecs.open(REQUIREMENTS_DEV_FILE) as buff:
        return buff.read().splitlines()


def get_requirements_test():
    with codecs.open(REQUIREMENTS_TEST_FILE) as buff:
        return buff.read().splitlines()


def get_long_description():
    with codecs.open(README_FILE, "rt") as buff:
        return buff.read()


def get_version():
    lines = open(VERSION_FILE, "rt").readlines()
    version_regex = r"^__version__ = ['\"]([^'\"]*)['\"]"
    for line in lines:
        mo = re.search(version_regex, line, re.M)
        if mo:
            return mo[1]
    raise RuntimeError(f"Unable to find version in {VERSION_FILE}.")


setup(
    name="pyloo",
    license="MIT",
    version="0.1.0",  # Hardcoded for now until __init__.py is set up
    description="Python implementation of Leave-One-Out cross-validation",
    author="Jordan Deklerk",
    author_email="jordan.deklerk@gmail.com",
    url="https://github.com/your-username/pyloo",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=get_requirements(),
    extras_require={
        "dev": get_requirements_dev(),
        "test": get_requirements_test(),
    },
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    include_package_data=True,
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)