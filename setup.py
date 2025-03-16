from setuptools import find_namespace_packages, setup

setup(
    name="ghostwriter",
    packages=find_namespace_packages(where="src"),
    package_dir={"": "src"},
)
