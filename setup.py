from setuptools import setup, find_packages

setup(
    name="minitorch",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
    ],
)