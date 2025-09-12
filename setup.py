from setuptools import setup, find_packages

setup(
    name="spam-filtering-nlp",
    version="1.0.0",
    description="A spam filtering system using NLP",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "nltk>=3.7",
        "scikit-learn>=1.0.2",
        "pandas>=1.4.2",
        "numpy>=1.22.3",
        "streamlit>=1.11.0",
        "joblib>=1.1.0"
    ],
    python_requires=">=3.8",
)