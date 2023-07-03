# RAGBooster

RAGBooster improves the performance of **retrieval-based large language models** by learning which data sources are important to retrieve high quality data.

We provide an example notebook that shows how we boost [RedPajama-INCITE-Instruct-3B-v1](https://huggingface.co/togethercomputer/RedPajama-INCITE-Instruct-3B-v1), a small LLM with 3 billion parameters to be on par with [OpenAI's GPT3.5](https://platform.openai.com/docs/models/gpt-3-5) (175 billion parameters) in a question answering task by using Bing websearch and ragbooster:

 * [Demo notebook: Boosting RedPajama-INCITE-Instruct-3B-v1 for question answering](demo-boosting-a-small-llm.ipynb)

Furthermore, we have an additional example notebook, where we demonstrate how to boost a [tiny qa model](https://huggingface.co/deepset/minilm-uncased-squad2) to get within 5% accuracy on GPT3.5 on a data imputation task:

 * [Demo notebook: Boosting minilm-uncased-squad2 for data imputation](demo-boosting-a-small-qa-model.ipynb)

## Background 

Have a look at our paper on __Improving Retrieval-Augmented Large Language Models with Data-Centric Refinement__ for detailed algorithms, proofs and experimental results. 

## Installation

RAGBooster is available as [pip package](https://pypi.org/project/ragbooster/), and can be installed as follows:

`pip install ragbooster`
