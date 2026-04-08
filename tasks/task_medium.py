from . import Task
from graders import grader_medium

class MediumTask(Task):
    def __init__(self):
        super().__init__('medium', max_steps=5)
        self.grader = grader_medium
