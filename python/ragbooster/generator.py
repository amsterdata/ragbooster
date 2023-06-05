from manifest import Manifest
from abc import ABC, abstractmethod

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
