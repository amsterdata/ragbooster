import json
import pandas as pd
from .core import Question


def load_wikifact_questions(path):

    with open(path, 'r') as myfile:
        helm_json = json.loads(myfile.read())

    questions = []

    for request_state in helm_json['request_states']:
        text = request_state['instance']['input']
        correct_answers = []
        for reference in request_state['instance']['references']:
            correct_answers.append(reference['output'])
        questions.append(Question(text, correct_answers))

    return questions


def load_imputation_dataset(path, impute, based_on):
    table = pd.read_csv(path)
    questions = []
    for _, row in table.iterrows():

        text = '; '.join([f'{column}: {row[column]}' for column in based_on])
        answer = row[impute].lower()

        questions.append(Question(text=text, correct_answers=[answer]))

    return questions
