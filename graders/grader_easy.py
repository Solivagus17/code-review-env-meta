from models import Action, Reward, RewardBreakdown

# Small epsilon to ensure scores are strictly between 0 and 1
EPSILON = 1e-6

def grade(action: Action, state: dict, step: int) -> Reward:
    gt   = state['ground_truth']
    bugs = gt['bugs']
    breakdown = RewardBreakdown()

    # 1. Verdict accuracy (0.0 or 0.4)
    breakdown.verdict_accuracy = 0.4 if action.verdict == gt['verdict'] else EPSILON

    # 2. Bug detection (up to 0.4): partial credit per bug found
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
        breakdown.bug_detection = EPSILON

    # 3. False positive penalty (-0.05 per false positive, min 0)
    fp = max(0, len(action.line_comments) - len(bugs))
    if fp > 0:
        breakdown.false_positive_penalty = -min(fp * 0.05, 0.1)
    else:
        breakdown.false_positive_penalty = EPSILON  # No penalty, but avoid exact 0

    # 4. Efficiency bonus (0.0-0.2): fewer steps = higher bonus
    breakdown.efficiency_bonus = max(EPSILON, 0.2 - (step * 0.05))

    total = sum([breakdown.verdict_accuracy, breakdown.bug_detection,
                 breakdown.false_positive_penalty, breakdown.efficiency_bonus])
    
    # Clamp to strictly (0, 1) with safety margin
    total = max(EPSILON, min(1.0 - EPSILON, total))
    
    return Reward(total=total, breakdown=breakdown,
                  message=f'Found {found}/{len(bugs)} bugs',
                  is_terminal=(action.verdict in ['approve','request_changes']))