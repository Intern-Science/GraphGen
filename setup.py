# pylint: skip-file
import os

from setuptools import find_packages, setup

pwd = os.path.dirname(__file__)
version_file = "graphgen/_version.py"


def readme():
    with open(os.path.join(pwd, "README.md"), encoding="utf-8") as f:
        content = f.read()
    return content


def get_version():
    with open(os.path.join(pwd, version_file), "r") as f:
        exec(compile(f.read(), version_file, "exec"))
    return locals()["__version__"]


def read_requirements():
    lines = []
    with open("requirements.txt", "r") as f:
        for line in f.readlines():
            if line.startswith("#"):
                continue
            if "textract" in line:
                continue
            if len(line) > 0:
                lines.append(line)
    return lines


install_packages = read_requirements()

if __name__ == "__main__":
    setup(
        name="graphg",
        version=get_version(),
        url="https://github.com/open-sciencelab/GraphGen",
        description="GraphGen: Enhancing Supervised Fine-Tuning for LLMs with Knowledge-Driven Synthetic Data Generation",
        long_description=readme(),
        long_description_content_type="text/markdown",
        author="open-sciencelab",
        author_email="open-sciencelab@pjlab.org.cn",
        packages=find_packages(exclude=["models"]),
        package_data={"GraphGen": ["configs/*"]},
        include_package_data=True,
        install_requires=install_packages,
        classifiers=[
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
        ],
        entry_points={"console_scripts": ["graphgen=graphgen.generate:main"]},
    )
