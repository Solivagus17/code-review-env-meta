---
title: CodeReviewEnv
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
tags:
  - openenv
  - code-review
  - reinforcement-learning
  - agent-environment
app_port: 7860
---

# CodeReviewEnv

## Environment Overview
CodeReviewEnv is an OpenEnv-compliant environment that simulates real-world software code review tasks. Code review is highly complex and consequential, evaluating an engineer's (or agent's) ability to detect logic flaws, catch security bugs, analyze intent, and provide constructive feedback. It evaluates whether an agent can properly analyze a pull request and decide an approval verdict.

## Observation Space
The `Observation` model tracks the state for the agent:
- `task_id` (str): 'easy', 'medium', or 'hard'
- `pr_id` (str): Unique PR identifier
- `pr_title` (str): Title of the Pull Request
- `pr_description` (str): Description of changes
- `code_diff` (str): Unified code diff string
- `language` (str): Defaults to 'python'
- `step_number` (int): Current action step in the environment
- `max_steps` (int): Max steps available for current task
- `review_history` (List[ReviewHistoryEntry]): Complete log of previous steps
- `task_instructions` (str): Specific details of the current task
- `done` (bool): True if the terminal condition has been reached
- `cumulative_reward` (float): Tracks accumulated score

## Action Space
The `Action` model details the agent's response to the environment:
- `verdict` (ReviewVerdict): 'approve' | 'request_changes' | 'comment' | 'needs_more_info'
- `overall_comment` (str): Overall feedback text (min_length=10)
- `line_comments` (List[LineComment]): Specific feedback for lines
  - `line_number` (int)
  - `comment` (str)
  - `severity` (SeverityLevel): 'critical' | 'high' | 'medium' | 'low' | 'info'
  - `category` (str): 'bug' | 'security' | 'style' | 'performance' | 'logic'
- `suggested_fixes` (List[str]): List of fixes provided
- `confidence_score` (float): Probability score 0.0 to 1.0
- `reasoning` (str): Optional thinking rationale

## Reward Function
Rewards are dense and provided at every step, avoiding purely binary outcomes:
- **Verdict Accuracy**: Correct verdict vs GT verdict (weighted by task).
- **Bug/Issue Detection**: Finding exact lines with matching severity.
- **Security Detection**: 2x multiplier for OWASP vulnerabilities (Hard task).
- **Comment Quality**: Word counts + presence of structured review terminology.
- **Fix Suggestions**: Points for proposing code remediation.
- **Efficiency Bonus**: Fewer steps yield more points.
- **False Positive Penalty**: Detects hallucinated bounds and subtracts points (-0.15 for critical FP).
- **Loop Penalty**: Triggers -0.10 to -0.15 if an identical verdict is submitted repetitively.

## Tasks
| ID | Difficulty | Description | Success Criteria |
|----|------------|-------------|------------------|
| `easy` | Easy | Classify bugs by severity in short Python snippets | ≥80% bugs found + accurate verdict |
| `medium` | Medium | Full code review with written feedback on a 50-line PR | ≥70% issues found + low FP |
| `hard` | Hard | Identify and remediate security vulnerabilities | Find all OWASP risks + high comment quality |

## Setup
Clone and set up using Docker:
```bash
git clone <repository>
cd codereviewenv
docker build -t codereviewenv .
docker run -p 7860:7860 codereviewenv
```

## Usage
Interact via the HTTP endpoints:
```bash
# Reset state for an easy task
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task_id": "easy"}'

# Submit an action
curl -X POST http://localhost:7860/step -H "Content-Type: application/json" -d '{
  "verdict": "request_changes",
  "overall_comment": "Found issues in the logic and authentication layer.",
  "confidence_score": 0.95
}'

# Check current state
curl http://localhost:7860/state
```

## Running Inference
```bash
export HF_TOKEN="your_hf_token_here"
python inference.py
```

## Baseline Scores
| Task | Min Score | Max Score | Notes |
|------|-----------|-----------|-------|
| easy — Bug Triage | 0.70 | 0.90 | Llama-3.3-70B baseline |
| medium — Deep Review | 0.50 | 0.75 | Llama-3.3-70B baseline |
| hard — Security Audit | 0.25 | 0.55 | Llama-3.3-70B baseline |

## OpenEnv Validation
```
pip install openenv
openenv validate .

✅ openenv.yaml found and valid
✅ Observation model: typed Pydantic
✅ Action model: typed Pydantic
✅ Reward model: typed Pydantic
✅ reset() endpoint: HTTP 200
✅ step() endpoint: HTTP 200
✅ state() endpoint: HTTP 200
✅ 3 tasks found with graders
✅ All reward values in [0.0, 1.0]
```
