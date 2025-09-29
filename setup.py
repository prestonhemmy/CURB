from setuptools import setup, find_packages

setup(
    name="news-classifier",
    version="0.0.1",
    pachages=find_packages(),
    install_requires=[
        "fastapi[standard]",
        "uvicorn[standard]",
        "torch >= 2.0.0",
        "transformers >= 4.35.0",
        "python-dotenv",
         # TODO: Additional dependencies
    ],
    python_requires=">=3.8",
)