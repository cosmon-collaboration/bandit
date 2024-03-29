# -*- coding: utf-8 -*-
"""Setup file for callat_corr_fitter
"""
#from espressodb import __version__
__version__="0.1"

__author__ = "@walkloud"

from os import path

from setuptools import setup, find_packages

CWD = path.abspath(path.dirname(__file__))

with open(path.join(CWD, "README.md"), encoding="utf-8") as inp:
    LONG_DESCRIPTION = inp.read()

with open(path.join(CWD, "requirements.txt"), encoding="utf-8") as inp:
    REQUIREMENTS = [el.strip() for el in inp.read().split(",")]

#with open(path.join(CWD, "requirements-dev.txt"), encoding="utf-8") as inp:
#    REQUIREMENTS_DEV = [el.strip() for el in inp.read().split(",")]

setup(
    name="callat_corr_fitter",
    python_requires=">=3.8",
    version=__version__,
    description="Library to fit 2pt and 3pt lattice QCD correlation functions",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/callat-qcd/callat_corr_fitter",
    project_urls={
        "Bug Reports": "https://github.com/callat-qcd/callat_corr_fitter/issues",
        "Source": "https://github.com/callat-qcd/callat_corr_fitter",
        "Documentation": "https://callat_corr_fitter.readthedocs.io",
    },
    author=__author__,
    author_email="walkloud@lbl.gov",
    keywords=["Lattice QCD"],
    packages=find_packages(
        include=["fitter"],
        exclude=[]),
    install_requires=REQUIREMENTS,
    entry_points={"console_scripts": ["fit_corr=fitter.fit_corr:main"]},
    #extras_require={"dev": REQUIREMENTS_DEV},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Database :: Database Engines/Servers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: BSD License",
    ],
    include_package_data=True,
)
