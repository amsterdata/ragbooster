from abc import ABC, abstractmethod
import shelve
from transformers import pipeline


class Generator(ABC):

    def __init__(self, llm, max_tokens):
        self.llm = llm
        self.max_tokens = max_tokens

    @abstractmethod
    def _create_prompt(self, question, params):
        pass

    def _generate(self, prompt):
        response = self.llm.run(prompt, max_tokens=self.max_tokens, return_response=True)
        return self._extract_answer(response)

    def generate(self, question, params=None):
        if params is None:
            params = {}
        return self._generate(self._create_prompt(question, params))

    @abstractmethod
    def _extract_answer(self, response):
        pass


class HuggingfaceQAGenerator(ABC):

    def __init__(self, model_name, cache_path):
        self.cache_path = cache_path
        self.pipeline = pipeline('question-answering', model=model_name, tokenizer=model_name)

    @abstractmethod
    def _create_prompt(self, question, params):
        pass

    @abstractmethod
    def _extract_answer(self, response):
        pass

    def generate(self, question, params=None):
        if params is None:
            params = {}
        return self._generate(self._create_prompt(question, params))

    def _generate(self, prompt):

        query_cache = None
        try:
            query_cache = shelve.open(self.cache_path)

            question_text = prompt['question']
            context = prompt['context']

            # shelve can't handle tuples...
            query = f'{question_text};;;{context}'

            if query in query_cache:
                return query_cache[query]

            response = self.pipeline({'question': question_text, 'context': context})

            answer = self._extract_answer(response)
            query_cache[query] = answer
            return answer

        finally:
            if query_cache is not None:
                query_cache.close()
