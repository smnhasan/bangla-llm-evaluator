from setuptools import setup, find_packages

setup(
    name="llm_bangla_evaluator",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "langchain",
        "langchain_community",
        "InstructorEmbedding==1.0.1",
        "sentence-transformers==2.2.2",
        "transformers>=4.20",
        "datasets>=2.20",
        "pyarrow>=17.0",
        "torch>=2.0",
        "huggingface-hub==0.24.0",
    ],
    author="S M Nahid Hasan",
    description="A Bangla LLM evaluation framework",
    url="https://github.com/smnhasan/bangla-llm-evaluator",
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)