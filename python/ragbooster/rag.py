from tqdm.notebook import tqdm
from ragbooster import learn_importance
from .core import grouped_weights, encode_groups, encode_retrievals, mode
from .tuning import tune_pruning_threshold

import logging

logger = logging.getLogger(__name__)


class RetrievalAugmentedModel:

    def __init__(self, retriever, generator, k):
        self.retriever = retriever
        self.generator = generator
        self.k = k

    def generate(self, question, params=None):

        results = self.retriever.retrieve(question)

        predictions = []
        for context, source in results[:self.k]:
            params = {'retrieved_context': context, 'source': source}
            answer = self.generator.generate(question, params)
            predictions.append(answer)

        if len(predictions) > 0:
            answer = mode(predictions)
        else:
            answer = ''

        return answer


class RAGBooster:

    def __init__(self, rag_model, validation_questions, learning_rate=10, num_epochs=100, n_jobs=-1):
        self.rag_model = rag_model
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.n_jobs = n_jobs
        self._fit(validation_questions)

    def _utility(self, retrieved, prediction):
        if prediction in retrieved["correct_answers"]:
            return 1.0
        else:
            return 0.0

    def _fit(self, validation_questions):

        logger.info(f'Computing validation corpus from {len(validation_questions)} samples...')
        validation_corpus = []

        for question in tqdm(validation_questions, leave=False):

            retrieved_answers = []
            retrieved_websites = []

            for context, url in self.rag_model.retriever.retrieve(question):
                retrieved_websites.append(url)
                params = {'retrieved_context': context, 'source': url}
                answer = self.rag_model.generator.generate(question, params)
                retrieved_answers.append(answer)

            validation_corpus.append({
                'question': question.text,
                'correct_answers': question.correct_answers,
                'retrieved_answers': retrieved_answers,
                'retrieved_websites': retrieved_websites,
            })

        logger.info('Learning importance weights for data sources...')
        encoded_retrievals, mapping = encode_retrievals(validation_corpus, "retrieved_websites",
                                                        "retrieved_answers", self._utility)
        grouping, group_mapping = encode_groups(mapping, self.rag_model.retriever.group)

        weights = learn_importance(encoded_retrievals, k=self.rag_model.k, learning_rate=self.learning_rate,
                                   num_epochs=self.num_epochs, n_jobs=self.n_jobs, grouping=grouping)
        domain_weights = grouped_weights(weights, grouping, group_mapping)

        percentile_range = range(0, 100, 5)

        logger.info('Tuning threshold for corpus pruning...')
        # grouping could be used here as well
        tuning_result = tune_pruning_threshold(validation_corpus, domain_weights, percentile_range,
                                               self._utility, self.rag_model.retriever.group,
                                               self.rag_model.k, normalize=True)

        logger.info(f'Achieved accuracy of {tuning_result.best_utility:.3f} with a pruning threshold ' +
                    f'of {tuning_result.best_threshold:.5f} on the validation set.')

        self.weights = domain_weights
        self.tuning_result = tuning_result

    # TODO this could be nicer with a decorator over the retriever
    def generate(self, question, params=None):
        predictions = []
        for context, url in self.rag_model.retriever.retrieve(question):
            if len(predictions) < self.rag_model.k:
                domain = self.rag_model.retriever.group(url)
                if domain not in self.weights or \
                        self.weights[domain] >= self.tuning_result.best_threshold:
                    params = {'retrieved_context': context, 'source': url}
                    answer = self.rag_model.generator.generate(question, params)
                    predictions.append(answer)

        if len(predictions) > 0:
            answer = mode(predictions)
        else:
            answer = ''
        return answer

    def importance(self, source):
        source_group = self.rag_model.retriever.group(source)
        if source_group not in self.weights:
            return None
        else:
            return self.weights[source_group]
