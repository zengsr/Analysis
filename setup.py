from setuptools import setup, find_packages

setup(
    name="census-income-analysis",
    version="1.0.0",
    description="Census Income Classification & Customer Segmentation Pipeline",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/census-income-analysis",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "joblib>=1.3.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
