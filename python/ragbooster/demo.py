import json
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
