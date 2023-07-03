# RAGBooster

RAGBooster improves the performance of **retrieval-based large language models** by learning which data sources are important to retrieve high quality data.

We provide an example notebook that shows how we boost [RedPajama-INCITE-Instruct-3B-v1](https://huggingface.co/togethercomputer/RedPajama-INCITE-Instruct-3B-v1), a small LLM with 3 billion parameters to be on par with [OpenAI's GPT3.5](https://platform.openai.com/docs/models/gpt-3-5) (175 billion parameters) in a question answering task by using Bing websearch and ragbooster:

 * [Demo notebook: Boosting RedPajama-INCITE-Instruct-3B-v1 for question answering](demo-boosting-a-small-llm.ipynb)

Furthermore, we have an additional example notebook, where we demonstrate how to boost a [tiny qa model](https://huggingface.co/deepset/minilm-uncased-squad2) to get within 5% accuracy on GPT3.5 on a data imputation task:

 * [Demo notebook: Boosting minilm-uncased-squad2 for data imputation](demo-boosting-a-small-qa-model.ipynb)

## Core classes

At the core of RAGBooster are [RetrievalAugmentedModels](https://github.com/amsterdata/ragbooster/blob/main/python/ragbooster/rag.py#L11), which fetch external data to improve prediction quality. Retrieval augmentation requires two components:

 * A **retriever**, which retrieves external data for a prediction sample. We currently only implement a [BingRetriever](https://github.com/amsterdata/ragbooster/blob/main/python/ragbooster/retriever.py#L12), which queries Microsoft's Bing Websearch API. 
 * A **generator**, which generates the final prediction from the prediction sample and the external data. This is typically a large language model. We provide the [Generator](https://github.com/amsterdata/ragbooster/blob/main/python/ragbooster/generator.py#L6) interface, which makes it very easy to leverage LLMs available via an API, for example from OpenAI.

Once you defined your retrieval-augmented model, you can leverage [RAGBooster](https://github.com/amsterdata/ragbooster/blob/main/python/ragbooster/rag.py#L36) to boost its performance by learning the data importance of retrieval sources (e.g., domains in the web). This often increases accuracy by a few percent.

## Background 

Have a look at our paper on __Improving Retrieval-Augmented Large Language Models with Data-Centric Refinement__ for detailed algorithms, proofs and experimental results. 

## Installation

RAGBooster is available as [pip package](https://pypi.org/project/ragbooster/), and can be installed as follows:

`pip install ragbooster`


## Installation for Development

 * Requires Python 3.9 and [Rust](https://www.rust-lang.org/tools/install) to be available
 
 1. Clone the repository: `git clone git@github.com:amsterdata/ragbooster.git`
 1. Change to the project directory: `cd ragbooster`
 1. Create a virtualenv: `python3.9 -m venv venv`
 1. Activate the virtualenv `source venv/bin/activate`
 1. Install the dev dependencies with `pip install ".[dev]"`
 1. Build the project `maturin develop --release`
 
 * Optional steps:
    * Run the tests with `cargo test --release`
    * Run the benchmarks with `RUSTFLAGS="-C target-cpu=native" cargo bench`
    * Run linting for the Python code with `flake8 python`
    * Start jupyter with `jupyter notebook` and run the example notebooks
