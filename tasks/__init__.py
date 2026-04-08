import json
import random
import os

class Task:
    def __init__(self, difficulty: str, max_steps: int):
        self.difficulty = difficulty
        self.max_steps = max_steps
        
        # Load PRs
        filepath = os.path.join(os.path.dirname(__file__), '..', 'data', f'{difficulty}_prs.json')
        with open(filepath, 'r') as f:
            self.fixtures = json.load(f)
            
    def get_initial_state(self):
        # Pick a random fixture, set random seed if needed per PRD
        # "All random seeds in environment fixtures must be fixed (random.seed(42))"
        random.seed(42) # For reproducibility
        # Though the PRD says it should be deterministic-ish, we could just return the first one or random with seed 42
        fixture = random.choice(self.fixtures)
        return fixture
