from . import Task
from graders import grader_easy

class EasyTask(Task):
    def __init__(self):
        super().__init__('easy', max_steps=3)
        self.grader = grader_easy
