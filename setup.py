#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup,find_namespace_packages
import os
from typing import List

VERSION_FILE = os.path.join(os.path.dirname(__file__), "VERSION.txt")


def _read_requirements() -> List[str]:
    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    requirements: List[str] = []
    try:
        with open(requirements_path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("-"):
                    continue
                if "#" in line:
                    line = line.split("#", 1)[0].strip()
                if line:
                    requirements.append(line)
    except FileNotFoundError:
        # Some build tools build wheels from an sdist in a temp dir.
        # Ensure the build doesn't fail if requirements.txt wasn't included.
        return []
    return requirements

INSTALL_REQUIRES = _read_requirements()
setup(
    name="mb_lang",
    description="RAG function file",
    author="Malav Bateriwala",
    packages=find_namespace_packages(include=["mb.*"]),
    scripts=[],
    install_requires=INSTALL_REQUIRES,
    setup_requires=["setuptools-git-versioning<2"],
    python_requires='>=3.8',
    setuptools_git_versioning={
        "enabled": True,
        "version_file": VERSION_FILE,
        "count_commits_from_version_file": True,
        "template": "{tag}",
        "dev_template": "{tag}.dev{ccount}+{branch}",
        "dirty_template": "{tag}.post{ccount}",
    },
)