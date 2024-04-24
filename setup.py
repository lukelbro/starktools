from setuptools import setup, find_packages

setup(
    name="starktools",
    version="0.1.16",
    author="Luke Brown",
    packages=find_packages(exclude=['*tests']),
    install_requires=['numpy', 'nested_dict']
)
