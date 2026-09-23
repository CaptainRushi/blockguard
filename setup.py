from setuptools import setup, find_packages

setup(
    name="blockguard",
    version="0.1.0",
    description="Hidden hallucination-filtering funnel for AI model outputs",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "spacy>=3.7.0",
        "lightgbm>=4.2.0",
        "transformers>=4.36.0",
        "torch>=2.1.0",
        "sentence-transformers>=2.5.0",
        "nltk>=3.8.1",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pydantic>=2.5.0",
        "httpx>=0.25.0",
        "click>=8.1.0",
    ],
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "blockguard=blockguard.cli:main",
        ],
    },
)
