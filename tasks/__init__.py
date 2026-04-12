import json
import random
import os

class Task:
    def __init__(self, difficulty: str, max_steps: int):
        self.difficulty = difficulty
        self.max_steps = max_steps


        filepath = os.path.join(os.path.dirname(__file__), '..', 'data', f'{difficulty}_prs.json')
        with open(filepath, 'r') as f:
            self.fixtures = json.load(f)

    def get_initial_state(self):


        random.seed(42)

        fixture = random.choice(self.fixtures)
        return fixture

