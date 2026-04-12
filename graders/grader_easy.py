from models import Action, Reward, RewardBreakdown


SCORE_MIN = 0.001
SCORE_MAX = 0.999

def _clamp(value: float) -> float:
    """Clamp to strictly (0, 1)."""
    return max(SCORE_MIN, min(SCORE_MAX, value))

def grade(action: Action, state: dict, step: int) -> Reward:
    gt   = state['ground_truth']
    bugs = gt['bugs']
    breakdown = RewardBreakdown()


    breakdown.verdict_accuracy = 0.4 if action.verdict == gt['verdict'] else 0.01


    found = 0
    for true_bug in bugs:
        for lc in action.line_comments:
            if (abs(lc.line_number - true_bug['line']) <= 2 and
                lc.category == true_bug['category'] and
                lc.severity == true_bug['severity']):
                found += 1
                break
    if len(bugs) > 0:
        breakdown.bug_detection = (found / len(bugs)) * 0.4
    else:
        breakdown.bug_detection = 0.01


    fp = max(0, len(action.line_comments) - len(bugs))
    if fp > 0:
        breakdown.false_positive_penalty = -min(fp * 0.05, 0.1)
    else:
        breakdown.false_positive_penalty = 0.0


    breakdown.efficiency_bonus = max(0.01, 0.2 - (step * 0.05))

    total = sum([breakdown.verdict_accuracy, breakdown.bug_detection,
                 breakdown.false_positive_penalty, breakdown.efficiency_bonus])


    total = _clamp(total)

    return Reward(total=total, breakdown=breakdown,
                  message=f'Found {found}/{len(bugs)} bugs',
                  is_terminal=(action.verdict in ['approve','request_changes']))
