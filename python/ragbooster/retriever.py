import os
import requests
import tldextract
import shelve
from abc import ABC, abstractmethod

class BingRetriever(ABC):

    def __init__(self, from_cache_only=False, max_results_per_query=50):
        self.subscription_key = os.getenv('BING_SUBSCRIPTION_KEY')
        self.from_cache_only = from_cache_only
        self.max_results_per_query = max_results_per_query

    def group(self, source):
        url_parts = tldextract.extract(source)
        return f'{url_parts.domain}.{url_parts.suffix}'

    def _search(self, query):

        try:
            query_cache = shelve.open('__bing_cache.pkl')

            if query in query_cache:
                return query_cache[query]
            elif self.from_cache_only:
                raise Exception(f'Query [{query}] not in cache, but we run in cache-only mode!')

            search_url = "https://api.bing.microsoft.com/v7.0/search"
            headers = {"Ocp-Apim-Subscription-Key": self.subscription_key}
            params = {
                "q": query,
                'count': self.max_results_per_query,
                'mkt': 'en-US',
                'setLang': 'en',
                'responseFilter': 'Webpages',
            }

            response = requests.get(search_url, headers=headers, params=params)
            response.raise_for_status()

            result = response.json()

            query_cache[query] = result

            return result

        finally:
            query_cache.close()

    @abstractmethod
    def create_query(self, question):
        pass

    def retrieve(self, question):
        query = self.create_query(question)
        result = self._search(query)

        return [(page['snippet'], page['url']) for page in result['webPages']['value']]
