import os
import requests
import tldextract
import shelve
from abc import ABC, abstractmethod

import logging

logger = logging.getLogger(__name__)

# TODO: Integrate retries https://stackoverflow.com/questions/15431044/can-i-set-max-retries-for-requests-request
class BingRetriever(ABC):

    def __init__(self, cache_path, max_results_per_query=50, market='en-US', language='en'):

        if os.getenv('BING_SUBSCRIPTION_KEY') is None:
            raise ValueError("Bing API key not set. Set BING_SUBSCRIPTION_KEY environment variable.")

        self.subscription_key = os.getenv('BING_SUBSCRIPTION_KEY')
        self.cache_path = cache_path
        self.max_results_per_query = max_results_per_query
        self.market = market
        self.language = language

        logger.info(f"Setup Bing retriever with cache_path={self.cache_path}," +
                    f"max_results_per_query={self.max_results_per_query},market={self.market},language={self.language}")

    def group(self, source):
        url_parts = tldextract.extract(source)
        return f'{url_parts.domain}.{url_parts.suffix}'

    def _search(self, query):

        query_cache = None
        try:
            query_cache = shelve.open(self.cache_path)

            if query in query_cache:
                return query_cache[query]

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
            if query_cache is not None:
                query_cache.close()

    @abstractmethod
    def create_query(self, question):
        pass

    def retrieve(self, question):
        query = self.create_query(question)
        result = self._search(query)

        return [(page['snippet'], page['url']) for page in result['webPages']['value']]
