#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="src",
    version="0.0.1",
    description="Slot Attention Lightning",
    author="Janghyuk Choi",
    author_email="janghyuk.ch@gmail.com",
    url="https://github.com/janghyuk-choi/slot-attention-lightning",
    install_requires=["pytorch-lightning", "hydra-core"],
    packages=find_packages(),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "train_command = src.train:main",
            "eval_command = src.eval:main",
        ]
    },
)
