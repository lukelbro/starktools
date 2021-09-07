from setuptools import setup, find_packages

setup(
    name="starktools",
    version="0.1",
    author="Luke Brown",
    packages=find_packages(),
    install_requires=['numpy', 'numba'],
)