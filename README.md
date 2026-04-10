---
title: CodeReviewEnv
emoji: 🐐
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

# CodeReviewEnv 🔍

**CodeReviewEnv** is an OpenEnv-compatible reinforcement learning environment that challenges AI agents to perform real-world software code review. Agents analyze Python pull requests, detect bugs and security vulnerabilities, classify issues by severity, and deliver structured review verdicts — just like a senior engineer would.

---

## 🧠 What Does an Agent Do?

At each step, the agent receives a pull request observation containing a code diff, PR title, description, and review history. The agent must produce a structured review action including a verdict, line-level annotations, suggested fixes, and a confidence score. The environment evaluates the response and returns a reward.

---

## 📦 Observation Space

| Field | Type | Description |
|---|---|---|
| `task_id` | string | Task difficulty: `easy`, `medium`, or `hard` |
| `pr_id` | string | Unique pull request identifier |
| `pr_title` | string | Title of the pull request |
| `pr_description` | string | Description of the proposed changes |
| `code_diff` | string | Unified diff of the code to review |
| `step_number` | int | Current step in the episode |
| `max_steps` | int | Maximum allowed steps for this task |
| `review_history` | list | Log of all previous actions this episode |
| `task_instructions` | string | Specific guidance for the current task |
| `done` | bool | Whether the episode has ended |
| `cumulative_reward` | float | Accumulated score so far |

---

## 🎮 Action Space

| Field | Type | Description |
|---|---|---|
| `verdict` | enum | `approve` \| `request_changes` \| `comment` \| `needs_more_info` |
| `overall_comment` | string | Full review text (min 10 characters) |
| `line_comments` | list | Per-line annotations: `line_number`, `comment`, `severity`, `category` |
| `suggested_fixes` | list | Concrete fix recommendations |
| `confidence_score` | float | Agent confidence from 0.0 to 1.0 |
| `reasoning` | string | Optional chain-of-thought |

**Severity levels:** `critical` · `high` · `medium` · `low` · `info`
**Categories:** `bug` · `security` · `style` · `performance` · `logic`

---

## 🏆 Tasks

| Task | Difficulty | Max Steps | Focus |
|---|---|---|---|
| `easy` | 🟢 Easy | 3 | Bug triage in short Python snippets |
| `medium` | 🟡 Medium | 5 | Full review of a 50-line pull request |
| `hard` | 🔴 Hard | 7 | Security vulnerability identification & remediation |

---

## 💰 Reward Signal

Rewards are **dense** — the agent receives feedback at every step. The score is a float strictly within `(0, 1)` and accounts for:

| Component | Effect |
|---|---|
| Verdict accuracy | ✅ Correct approval decision |
| Issue detection | ✅ Real bugs and vulnerabilities found |
| Comment quality | ✅ Clear, detailed, actionable feedback |
| Fix suggestions | ✅ Concrete remediation steps provided |
| Efficiency bonus | ✅ Fewer steps to conclusive verdict |
| False positive penalty | ❌ Hallucinated bugs reduce the score |
| Loop penalty | ❌ Repeating verdicts without progress |

---

## 🌐 API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/reset` | Start a new episode. Body: `{"task_id": "easy"}` |
| `POST` | `/step` | Submit a review action. Returns `reward` (float) and `done`. |
| `GET` | `/state` | Current internal environment state |
| `GET` | `/health` | Liveness check — returns `200 OK` when ready |
| `GET` | `/schema` | JSON schemas for `Action` and `Observation` models |

### Example: POST /step

**Request:**
```json
{
  "verdict": "request_changes",
  "overall_comment": "Found a critical SQL injection vulnerability on line 7.",
  "line_comments": [
    { "line_number": 7, "comment": "Unsanitized input passed to SQL.", "severity": "critical", "category": "security" }
  ],
  "suggested_fixes": ["Use parameterized queries."],
  "confidence_score": 0.95
}
```

**Response:**
```json
{ "observation": { "done": true, "cumulative_reward": 0.82 }, "reward": 0.82, "done": true }
```

---

## 🚀 Running Locally

```bash
git clone https://github.com/Solivagus17/code-review-env-meta
cd code-review-env-meta
docker build -t codereviewenv .
docker run -p 7860:7860 codereviewenv
```

---

## 🤖 Running the Baseline Agent

```bash
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export API_KEY="your_api_key_here"
export ENV_BASE_URL="http://localhost:7860"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"

python inference.py
```

**Structured output format:**
```
[START] task=easy
[STEP] task=easy step=1 reward=0.7842
[END] task=easy score=0.7842 steps=1
{"type": "SUMMARY", "scores": {"easy": 0.78, "medium": 0.61, "hard": 0.44}}
```

---

## 📋 Requirements

```
fastapi
uvicorn
pydantic
openai
requests
```

---

## 📄 License

MIT
