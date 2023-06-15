from collections import Counter
from dataclasses import dataclass, field
from tqdm.notebook import tqdm


@dataclass
class Question:
    text: str
    correct_answers: list[str]
    metadata: dict[str] = field(default_factory=dict)


def score(test_questions, model):
    num_correct = 0
    for test_question in tqdm(test_questions, leave=False):

        answer = model.generate(test_question)

        if answer in test_question.correct_answers:
            num_correct += 1

    accuracy = num_correct / len(test_questions)
    return accuracy


def mode(values):
    return Counter(values).most_common(n=1)[0][0]


def encode_retrievals(retrievals, retrieved_key, prediction_key, utility):

    encoded_retrievals = []

    all_retrieveds = set()

    for retrieval in retrievals:
        for retrieved in retrieval[retrieved_key]:
            all_retrieveds.add(retrieved)

    all_retrieveds = list(all_retrieveds)
    all_retrieveds.sort()

    mapping = {retrieved: index for index, retrieved in enumerate(all_retrieveds)}

    for retrieval in retrievals:
        retrieveds = [mapping[name] for name in retrieval[retrieved_key]]
        utilities = [utility(retrieval, prediction) for prediction in retrieval[prediction_key]]
        encoded_retrievals.append({
            "retrieved": retrieveds,
            "utility_contributions": utilities
        })

    return encoded_retrievals, mapping


def encode_groups(mapping, group):
    groups = set()

    for retrieved in mapping.keys():
        assigned_group = group(retrieved)
        groups.add(assigned_group)

    all_groups = list(groups)
    all_groups.sort()

    group_mapping = {name: index for index, name in enumerate(all_groups)}

    grouping = [0 for _ in range(0, len(mapping))]

    for retrieved in mapping.keys():
        assigned_group = group(retrieved)
        retrieved_index = mapping[retrieved]
        group_index = group_mapping[assigned_group]
        grouping[retrieved_index] = group_index

    return grouping, group_mapping


def grouped_weights(weights, grouping, group_mapping):

    w_grouped = {}

    retrieved_index_per_group = {}

    for retrieved_index, group_index in enumerate(grouping):
        retrieved_index_per_group[group_index] = retrieved_index
        # TODO add break

    for group, group_index in group_mapping.items():
        w_grouped[group] = weights[retrieved_index_per_group[group_index]]

    return w_grouped
