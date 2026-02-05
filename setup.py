from setuptools import setup, find_packages

setup(
    name="rl-engine",
    version="0.1.0",
    packages=find_packages(include=["rl_engine", "rl_engine.*"]),
    install_requires=[
        "torch>=2.4.0",
        "tabulate",
        "numpy",
        "accelerate",
        "transformers",
    ],
    extras_require={
        "cuda": ["flashinfer"],
        "rocm": ["aiter"],
    },
    python_requires=">=3.10",
    include_package_data=True,
    zip_safe=False,
)