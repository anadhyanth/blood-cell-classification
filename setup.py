from setuptools import setup, find_packages

setup(
    name="blood-classification",
    version="1.0.0",
    description="Blood Classification using Machine Learning",
    author="B. Anadhyanth",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "joblib"
    ]
)