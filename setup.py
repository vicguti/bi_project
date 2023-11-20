from setuptools import find_packages, setup

from functions import __version__

setup(
    name="functions",
    packages=find_packages(exclude=["tests", "tests.*"]),
    setup_requires=["wheel"],
    version=__version__,
)
