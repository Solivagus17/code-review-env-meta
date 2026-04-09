from typing import List
from models import Observation, Action, Reward, ReviewHistoryEntry, ReviewVerdict
from tasks.task_easy import EasyTask
from tasks.task_medium import MediumTask
from tasks.task_hard import HardTask

def load_task(task_id: str):
    if task_id == 'easy':
        return EasyTask()
    elif task_id == 'medium':
        return MediumTask()
    elif task_id == 'hard':
        return HardTask()
    else:
        raise ValueError(f"Unknown task: {task_id}")

class CodeReviewEnv:
    def __init__(self, task_id: str = 'easy'):
        self.task_id    = task_id
        self.task       = load_task(task_id)       # Returns task fixture
        self.state_data = {}
        self.step_count = 0
        self.done       = False
        self.history: List[ReviewHistoryEntry] = []
        self.cumulative_reward = 0.0

    def _build_observation(self) -> Observation:
        obs = Observation(
            task_id=self.task_id,
            pr_id=self.state_data.get('pr_id', ''),
            pr_title=self.state_data.get('pr_title', ''),
            pr_description=self.state_data.get('pr_description', ''),
            code_diff=self.state_data.get('code_diff', ''),
            step_number=self.step_count,
            max_steps=self.task.max_steps,
            review_history=self.history,
            task_instructions=f"Complete the {self.task_id} code review task.",
            done=self.done,
            cumulative_reward=self.cumulative_reward
        )
        return obs

    def reset(self) -> Observation:
        """Reset environment to initial state. Returns initial observation."""
        self.step_count        = 0
        self.done              = False
        self.history           = []
        self.cumulative_reward = 0.0
        self.state_data        = self.task.get_initial_state()
        return self._build_observation()

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        """Execute one review action. Returns (obs, reward, done, info)."""
        if self.done:
            raise ValueError('Environment is done. Call reset() first.')
            
        # Add history to state_data for grader to use (loop penalty checking)
        self.state_data['history'] = [{'action_type': h.action_type} for h in self.history]
        
        reward = self.task.grader.grade(action, self.state_data, self.step_count)
        
        # 1. Calculate Target Absolute Score (Clamped strictly)
        # We clamp the target absolute score to [0.01, 0.99] to give us room for nudges
        target_abs = max(0.01, min(0.99, reward.total))
        
        # Max steps exceeded penalty
        max_steps_exceeded = (self.step_count + 1) >= self.task.max_steps
        if max_steps_exceeded and not reward.is_terminal and action.verdict not in [ReviewVerdict.APPROVE, ReviewVerdict.REQUEST_CHANGES]:
            reward.is_terminal = True
            target_abs = max(0.01, target_abs - 0.10)
            reward.message += " (Max steps exceeded)"

        # 2. Calculate ideal delta
        # reward.total in the response must be the delta that makes the SUM equal target_abs
        ideal_delta = target_abs - self.cumulative_reward

        # 3. Nudge logic (The "Anti-Zero" & "Anti-One" Guard)
        # If the delta is exactly 0, provide a tiny positive signal (0.0001) 
        # unless it would push the total sum to exactly 1.0 (impossible due to 0.99 clamp).
        # This ensures the validator never sees a "0.0" score for any step.
        if abs(ideal_delta) < 0.0001:
            ideal_delta = 0.0001
        
        # 4. Final safety check for the sum
        projected_sum = self.cumulative_reward + ideal_delta
        if projected_sum >= 1.0:
            ideal_delta = 0.999 - self.cumulative_reward
        elif projected_sum <= 0.0:
            ideal_delta = 0.001 - self.cumulative_reward

        ideal_delta = max(0.0001, min(ideal_delta, 0.999))
        reward.total = round(ideal_delta, 6)
        self.cumulative_reward = round(self.cumulative_reward + reward.total, 6)
        self.step_count       += 1
        
        self.done = (
            reward.is_terminal or
            self.step_count >= self.task.max_steps or
            action.verdict in [ReviewVerdict.APPROVE, ReviewVerdict.REQUEST_CHANGES]
        )
        
        entry = ReviewHistoryEntry(
            step=self.step_count, action_type=action.verdict,
            comment=action.overall_comment, reward=reward.total
        )
        self.history.append(entry)
        
        obs  = self._build_observation()
        info = {'step': self.step_count, 'cumulative_reward': self.cumulative_reward}
        return obs, reward, self.done, info

    def state(self) -> dict:
        """Return current internal state as a serialisable dict."""
        return {
            'task_id':          self.task_id,
            'step_count':       self.step_count,
            'done':             self.done,
            'cumulative_reward':self.cumulative_reward,
            'history':          [h.model_dump() for h in self.history],
            'state_data':       self.state_data,
        }
