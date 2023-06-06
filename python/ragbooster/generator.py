from abc import ABC, abstractmethod
import shelve
from manifest import Manifest
from transformers import pipeline


class GPT35Generator(ABC):

    def __init__(self):
        self.manifest = Manifest(
            client_name="openai",
            engine="text-davinci-003",
            cache_name="sqlite",
            cache_connection="manifest-cache.sqlite",
        )

    def generate(self, question, snippet=None):
        prompt = self._create_prompt(question, snippet)
        response = self.manifest.run(prompt, max_tokens=10, return_response=True)
        return self._extract_answer(response)

    @abstractmethod
    def _create_prompt(self, question, snippet):
        pass

    @abstractmethod
    def _extract_answer(self, response):
        pass

# TODO Naming is weird here, as we have two different questions
class HuggingfaceQAGenerator(ABC):

    def __init__(self):
        self.pipeline = pipeline('question-answering', model=self._model_name(),
                                 tokenizer=self._model_name())

    @abstractmethod
    def _model_name(self):
        pass

    @abstractmethod
    def _qa_question(self):
        pass

    @abstractmethod
    def _create_context(self, question, snippet=None):
        pass

    @abstractmethod
    def _extract_answer(self, response):
        pass

    def generate(self, question, snippet=None):

        try:
            query_cache = shelve.open('__huggingface_cache.pkl')

            # I have no idea why this returns a tuple... must be related to abstract classes in Python
            question_text, = self._qa_question(),
            context = self._create_context(question, snippet)

            # shelve can't handle tuples...
            query = f'{question_text};;;{context}'

            if query in query_cache:
                return query_cache[query]

            response = self.pipeline({'question': question_text, 'context': context})



            answer = self._extract_answer(response)
            query_cache[query] = answer
            return answer

        finally:
            query_cache.close()
