from setuptools import setup, find_packages

setup(
    name="news-classifier",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "fastapi[standard]",
        "uvicorn[standard]",
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "scikit-learn",
        "numpy",
        "langdetect",
        "python-dotenv",
    ],
    python_requires=">=3.10",
)