from . import Task
from graders import grader_hard

class HardTask(Task):
    def __init__(self):
        super().__init__('hard', max_steps=7)
        self.grader = grader_hard

