from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum



class DifficultyLevel(str, Enum):
    EASY   = 'easy'
    MEDIUM = 'medium'
    HARD   = 'hard'

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



class ReviewHistoryEntry(BaseModel):
    step:         int
    action_type:  str
    comment:      Optional[str] = None
    verdict:      Optional[str] = None
    reward:       float

class Observation(BaseModel):
    task_id:           str
    pr_id:             str
    pr_title:          str
    pr_description:    str
    code_diff:         str
    language:          str = 'python'
    step_number:       int
    max_steps:         int
    review_history:    List[ReviewHistoryEntry] = []
    task_instructions: str
    metadata:          Dict[str, Any] = {}
    done:              bool = False
    cumulative_reward: float = 0.0



class LineComment(BaseModel):
    line_number: int
    comment:     str
    severity:    SeverityLevel
    category:    str

class Action(BaseModel):
    verdict:           ReviewVerdict
    overall_comment:   str = Field(min_length=10, max_length=2000)
    line_comments:     List[LineComment] = []
    suggested_fixes:   List[str] = []
    confidence_score:  float = Field(ge=0.0, le=1.0, default=0.5)
    reasoning:         Optional[str] = None



class RewardBreakdown(BaseModel):
    verdict_accuracy:       float = 0.0
    bug_detection:          float = 0.0
    security_detection:     float = 0.0
    comment_quality:        float = 0.0
    false_positive_penalty: float = 0.0
    loop_penalty:           float = 0.0
    efficiency_bonus:       float = 0.0

class Reward(BaseModel):
    total:       float
    breakdown:   RewardBreakdown
    message:     str
    is_terminal: bool = False
