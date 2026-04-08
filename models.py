from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum

class DifficultyLevel(str, Enum):
    EASY   = 'easy'
    MEDIUM = 'medium'
    HARD   = 'hard'

class ReviewHistoryEntry(BaseModel):
    step:        int
    action_type: str
    comment:     Optional[str] = None
    verdict:     Optional[str] = None
    reward:      float

class Observation(BaseModel):
    task_id:          str                     # e.g. 'easy', 'medium', 'hard'
    pr_id:            str                     # Unique PR identifier
    pr_title:         str
    pr_description:   str
    code_diff:        str                     # Unified diff string
    language:         str = 'python'
    step_number:      int
    max_steps:        int
    review_history:   List[ReviewHistoryEntry] = []
    task_instructions:str                     # Injected task-specific prompt
    metadata:         Dict[str, Any] = {}     # author, lines_changed, etc.
    done:             bool = False
    cumulative_reward:float = 0.0

class SeverityLevel(str, Enum):
    CRITICAL = 'critical'
    HIGH     = 'high'
    MEDIUM   = 'medium'
    LOW      = 'low'
    INFO     = 'info'

class ReviewVerdict(str, Enum):
    APPROVE           = 'approve'
    REQUEST_CHANGES   = 'request_changes'
    COMMENT           = 'comment'
    NEEDS_MORE_INFO   = 'needs_more_info'

class LineComment(BaseModel):
    line_number: int
    comment:     str
    severity:    SeverityLevel
    category:    str    # 'bug'|'security'|'style'|'performance'|'logic'

class Action(BaseModel):
    verdict:          ReviewVerdict
    overall_comment:  str = Field(min_length=10, max_length=2000)
    line_comments:    List[LineComment] = []
    suggested_fixes:  List[str] = []
    confidence_score: float = Field(ge=0.0, le=1.0, default=0.5)
    reasoning:        Optional[str] = None

class RewardBreakdown(BaseModel):
    verdict_accuracy:     float = 0.0   # Was approve/reject correct?
    bug_detection:        float = 0.0   # Fraction of bugs found
    security_detection:   float = 0.0   # Fraction of security issues found
    comment_quality:      float = 0.0   # Relevance + actionability
    false_positive_penalty:float = 0.0  # Penalizes hallucinated bugs
    loop_penalty:         float = 0.0   # Penalizes repeated identical actions
    efficiency_bonus:     float = 0.0   # Bonus for finishing before max_steps

class Reward(BaseModel):
    total:     float = Field(ge=0.0, le=1.0)
    breakdown: RewardBreakdown
    message:   str
    is_terminal:bool = False
