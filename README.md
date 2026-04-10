# 🔍 CodeReviewEnv

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)
![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-success)
![License](https://img.shields.io/badge/License-MIT-green)

> An [OpenEnv](https://github.com/openenv)-compliant reinforcement learning environment for training and evaluating agents on real-world software code review tasks.

-----

## Table of Contents

- [Overview](#overview)
- [Tasks](#tasks)
- [Reward Function](#reward-function)
- [Baseline Scores](#baseline-scores)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Usage](#usage)
- [Running Inference](#running-inference)
- [Observation Space](#observation-space)
- [Action Space](#action-space)
- [OpenEnv Validation](#openenv-validation)
- [Contributing](#contributing)
- [License](#license)

-----

## Overview

**CodeReviewEnv** simulates real-world pull request (PR) review scenarios. It evaluates whether an agent can:

- Detect logic flaws, security bugs, and code quality issues
- Provide constructive, structured feedback
- Return a correct approval verdict (`approve`, `request_changes`, `comment`, or `needs_more_info`)

The environment spans three difficulty tiers — from simple bug triage on short snippets to full OWASP security audits on realistic PRs. It is designed for use in RL training pipelines and agent evaluation benchmarks.

-----

## Tasks

Three escalating challenges that push agents from basic classification to expert-level security reasoning:

|ID      |Difficulty|Description                                            |Success Criteria                            |
|--------|----------|-------------------------------------------------------|--------------------------------------------|
|`easy`  |🟢 Easy    |Classify bugs by severity in short Python snippets     |≥ 80% bugs found + accurate verdict         |
|`medium`|🟡 Medium  |Full code review with written feedback on a ~50-line PR|≥ 70% issues found + low false positive rate|
|`hard`  |🔴 Hard    |Identify and remediate security vulnerabilities        |Find all OWASP risks + high comment quality |

-----

## Reward Function

Rewards are **dense** — provided at every step rather than only at episode end. This avoids sparse-reward training instability and gives agents a richer learning signal throughout the episode.

|Component                   |Description                                                         |
|----------------------------|--------------------------------------------------------------------|
|✅ **Verdict Accuracy**      |Correct verdict vs. ground truth, weighted by task difficulty       |
|🐛 **Bug/Issue Detection**   |Points for flagging exact lines with matching severity              |
|🔐 **Security Detection**    |2× multiplier for identifying OWASP vulnerabilities (hard task only)|
|💬 **Comment Quality**       |Scored on word count and use of structured review terminology       |
|🔧 **Fix Suggestions**       |Points awarded for proposing concrete code remediations             |
|⚡ **Efficiency Bonus**      |Higher reward for completing the task in fewer steps                |
|❌ **False Positive Penalty**|`-0.15` per hallucinated critical issue                             |
|🔁 **Loop Penalty**          |`-0.10` to `-0.15` for submitting an identical verdict repeatedly   |

-----

## Baseline Scores

Scores achieved by **Llama-3.3-70B** on each task — a useful reference point for evaluating your own agent:

|Task                   |Min Score|Max Score|
|-----------------------|---------|---------|
|`easy` — Bug Triage    |0.70     |0.90     |
|`medium` — Deep Review |0.50     |0.75     |
|`hard` — Security Audit|0.25     |0.55     |

Can your agent beat these? The hard tier leaves significant headroom.

-----

## Prerequisites

- **Python** 3.10+
- **Docker** (recommended for quickstart)
- **uv** or **pip** for local Python dependency management
- A [HuggingFace](https://huggingface.co) account and token (for running inference with hosted models)

-----

## Setup

### Option 1: Docker (recommended)

```bash
git clone https://github.com/Solivagus17/code-review-env-meta.git
cd code-review-env-meta
docker build -t codereviewenv .
docker run -p 7860:7860 codereviewenv
```

### Option 2: Local (uv)

```bash
git clone https://github.com/Solivagus17/code-review-env-meta.git
cd code-review-env-meta
uv sync
uv run environment.py
```

### Option 3: Local (pip)

```bash
git clone https://github.com/Solivagus17/code-review-env-meta.git
cd code-review-env-meta
pip install -r requirements.txt
python environment.py
```

The server will start on `http://localhost:7860`.

-----

## Usage

The environment exposes three HTTP endpoints once running.

### Reset the environment

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy"}'
```

### Submit an action (step)

```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "verdict": "request_changes",
    "overall_comment": "Found issues in the logic and authentication layer.",
    "confidence_score": 0.95
  }'
```

### Inspect current state

```bash
curl http://localhost:7860/state
```

-----

## Running Inference

Set your HuggingFace token and run the inference script:

```bash
export HF_TOKEN="your_hf_token_here"
python inference.py
```

This will run the default model against all three task difficulties and print per-step rewards and a final episode summary.

-----

## Observation Space

The `Observation` model tracks the full state presented to the agent at each step:

|Field              |Type                      |Description                                       |
|-------------------|--------------------------|--------------------------------------------------|
|`task_id`          |`str`                     |Difficulty tier: `'easy'`, `'medium'`, or `'hard'`|
|`pr_id`            |`str`                     |Unique PR identifier                              |
|`pr_title`         |`str`                     |Title of the pull request                         |
|`pr_description`   |`str`                     |Description of the proposed changes               |
|`code_diff`        |`str`                     |Unified diff string of the PR                     |
|`language`         |`str`                     |Programming language (defaults to `'python'`)     |
|`step_number`      |`int`                     |Current action step within the episode            |
|`max_steps`        |`int`                     |Maximum steps allowed for the task                |
|`review_history`   |`List[ReviewHistoryEntry]`|Full log of all previous actions and rewards      |
|`task_instructions`|`str`                     |Task-specific instructions for the agent          |
|`done`             |`bool`                    |`True` when the terminal condition is reached     |
|`cumulative_reward`|`float`                   |Accumulated reward across all steps               |

-----

## Action Space

The `Action` model defines what the agent can submit at each step:

|Field             |Type               |Description                                                                  |
|------------------|-------------------|-----------------------------------------------------------------------------|
|`verdict`         |`ReviewVerdict`    |One of: `'approve'` | `'request_changes'` | `'comment'` | `'needs_more_info'`|
|`overall_comment` |`str`              |Overall feedback on the PR (min. 10 characters)                              |
|`line_comments`   |`List[LineComment]`|Inline comments on specific lines (see below)                                |
|`suggested_fixes` |`List[str]`        |Proposed code-level remediation steps                                        |
|`confidence_score`|`float`            |Agent’s confidence in its verdict (`0.0` – `1.0`)                            |
|`reasoning`       |`str`              |Optional chain-of-thought rationale                                          |

**LineComment fields:**

|Field        |Type           |Values                                                          |
|-------------|---------------|----------------------------------------------------------------|
|`line_number`|`int`          |Line number in the diff                                         |
|`comment`    |`str`          |Feedback text                                                   |
|`severity`   |`SeverityLevel`|`'critical'` | `'high'` | `'medium'` | `'low'` | `'info'`       |
|`category`   |`str`          |`'bug'` | `'security'` | `'style'` | `'performance'` | `'logic'`|

-----

## OpenEnv Validation

This environment is fully compliant with the [OpenEnv](https://github.com/openenv) specification. To validate:

```bash
pip install openenv
openenv validate .
```

Expected output:

```
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

-----

## Contributing

Contributions are welcome! To get started:

1. Fork the repository
1. Create a feature branch: `git checkout -b feature/your-feature`
1. Commit your changes: `git commit -m 'Add your feature'`
1. Push to the branch: `git push origin feature/your-feature`
1. Open a Pull Request

Please open an [issue](https://github.com/Solivagus17/code-review-env-meta/issues) first for major changes so we can discuss the approach.

-----

## License

This project is licensed under the [MIT License](LICENSE).