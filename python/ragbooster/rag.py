from tqdm.notebook import tqdm
from ragbooster import learn_importance
from .core import grouped_weights, encode_groups, encode_retrievals, mode
from .tuning import tune_pruning_threshold

class RAGModel:

    def __init__(self, retriever, generator, k):
        self.retriever = retriever
        self.generator = generator
        self.k = k

    def generate(self, test_question):

        results = self.retriever.retrieve(test_question.text)

        predictions = []
        for snippet, _ in results[:self.k]:
            answer = self.generator.generate(test_question, snippet)
            predictions.append(answer)

        if len(predictions) > 0:
            answer = mode(predictions)
        else:
            answer = ''

        return answer


class RAGBooster:

    def __init__(self, rag_model, validation_questions):
        self.rag_model = rag_model
        self._fit(validation_questions)

    def _utility(self, retrieved, prediction):
        if prediction in retrieved["correct_answers"]:
            return 1.0
        else:
            return 0.0

    def _fit(self, validation_questions):

        print('Computing validation corpus...')
        validation_corpus = []

        for question in tqdm(validation_questions, leave=False):

            retrieved_answers = []
            retrieved_websites = []

            for snippet, url in self.rag_model.retriever.retrieve(question.text):
                retrieved_websites.append(url)
                answer = self.rag_model.generator.generate(question, snippet)
                retrieved_answers.append(answer)

            validation_corpus.append({
                'question': question.text,
                'correct_answers': question.correct_answers,
                'retrieved_answers': retrieved_answers,
                'retrieved_websites': retrieved_websites,
            })

        print('Learning importance weights for data sources...')
        encoded_retrievals, mapping = encode_retrievals(validation_corpus, "retrieved_websites",
                                                        "retrieved_answers", self._utility)
        grouping, group_mapping = encode_groups(mapping, self.rag_model.retriever.group)

        # TODO these need to be class params
        weights = learn_importance(encoded_retrievals, k=self.rag_model.k, learning_rate=10, num_steps=100,
                                   n_jobs=-1, grouping=grouping)
        domain_weights = grouped_weights(weights, grouping, group_mapping)

        percentile_range = range(0, 100, 5)

        print('Tuning threshold for corpus pruning...')
        # grouping could be used here as well
        tuning_result = tune_pruning_threshold(validation_corpus, domain_weights, percentile_range,
                                               self._utility, self.rag_model.retriever.group,
                                               self.rag_model.k, normalize=True)

        print(f'Achieved accuracy of {tuning_result.best_utility:.3f} with a pruning threshold ' + \
              f'of {tuning_result.best_threshold:.5f} on the validation set.')

        self.weights = domain_weights
        self.tuning_result = tuning_result

    # TODO this could be nicer with a decorator over the retriever
    def generate(self, question):
        predictions = []
        for snippet, url in self.rag_model.retriever.retrieve(question.text):
            if len(predictions) < self.rag_model.k:
                domain = self.rag_model.retriever.group(url)
                if domain not in self.weights or \
                        self.weights[domain] >= self.tuning_result.best_threshold:

                    answer = self.rag_model.generator.generate(question, snippet)
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