from setuptools import setup, find_packages

setup(
    name="coderipple",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "transformers>=4.38.0",
        "torch>=2.2.0",
        "networkx>=3.2.0",
        "gitpython>=3.1.40",
        "flask>=3.0.0",
        "flask-cors>=4.0.0",
    ],
)