#!/usr/bin/env python
from setuptools import setup

with open("requirements.txt","r+") as f:
    req_raw = f.readlines()
reqs = [x.rstrip("\n") for x in req_raw]

setup(name='repair',
      version='1.0',
      description='Python Distribution Utilities',
      author='Alma Andersson',
      author_email='alma.andersson@differentiable.net',
      install_requires=reqs,
      python_requires=">=3.7",
      entry_points={"console_scripts":["repair = repair.__main__:main"]},
     )
