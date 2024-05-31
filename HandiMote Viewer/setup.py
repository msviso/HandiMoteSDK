# setup.py
from setuptools import setup, find_packages

setup(
    name="HandiMoteSDK",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["bleak>=0.12.1", "asyncio"],
    author="Microsense Vision",
    author_email="info@msviso.com",
    description="An SDK for the HandiMote BLE device",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/msviso/HandiMoteSDK",
)
