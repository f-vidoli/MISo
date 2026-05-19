#!/usr/bin/env python3
"""Setup script for pytlc."""

from setuptools import setup, find_packages
from pathlib import Path

# Read long description from README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='pytlc',
    version='0.1.0',
    author='Based on work by Xingyi Du et al.',
    description='Python wrapper for Lifting Simplices to Find Injectivity (TLC energy)',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/duxingyi-charles/lifting_simplices_to_find_injectivity',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Visualization',
    ],
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.20.0',
        'scipy>=1.7.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'matplotlib>=3.4',
        ],
    },
)
