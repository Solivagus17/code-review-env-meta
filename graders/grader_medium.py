from models import Action, Reward, RewardBreakdown

# Strict bounds — scores must be strictly inside (0, 1)
SCORE_MIN = 0.001
SCORE_MAX = 0.999

def _clamp(value: float) -> float:
    """Clamp to strictly (0, 1)."""
    return max(SCORE_MIN, min(SCORE_MAX, value))

def grade(action: Action, state: dict, step: int) -> Reward:
    gt        = state['ground_truth']
    issues    = gt['issues']   # All seeded issues (bugs + logic + style)
    breakdown = RewardBreakdown()

    # 1. Verdict accuracy (0.25)
    breakdown.verdict_accuracy = 0.25 if action.verdict == gt['verdict'] else 0.01

    # 2. Issue detection (0.35): weighted by severity
    SEVERITY_WEIGHT = {'critical':0.3,'high':0.25,'medium':0.2,'low':0.15,'info':0.1}
    total_weight = sum(SEVERITY_WEIGHT.get(i['severity'],0.1) for i in issues)
    found_weight = 0
    for issue in issues:
        for lc in action.line_comments:
            if (abs(lc.line_number - issue['line']) <= 3 and
                lc.category == issue['category']):
                found_weight += SEVERITY_WEIGHT.get(issue['severity'], 0.1)
                break
    
    if total_weight > 0:
        breakdown.bug_detection = (found_weight / total_weight) * 0.35
    else:
        breakdown.bug_detection = 0.01

    # 3. Comment quality (0.20): word count + keyword presence
    word_count = len(action.overall_comment.split())
    wc_score   = min(word_count / 80, 1.0) * 0.10  # full score at 80 words
    kw_score   = sum(1 for kw in ['because','should','instead','recommend','suggest']
                     if kw in action.overall_comment.lower()) / 5 * 0.10
    breakdown.comment_quality = max(0.01, wc_score + kw_score)

    # 4. Fix suggestions (0.10): at least 1 concrete fix required
    breakdown.efficiency_bonus = max(0.01, min(len(action.suggested_fixes) * 0.05, 0.10))

    # 5. False positive penalty
    fp = max(0, len(action.line_comments) - len(issues))
    if fp > 0:
        breakdown.false_positive_penalty = -min(fp * 0.05, 0.15)
    else:
        breakdown.false_positive_penalty = 0.0

    # 6. Loop penalty: if last 2 actions had same verdict
    breakdown.loop_penalty = 0.0
    if len(state.get('history', [])) >= 2:
        last_two = [h['action_type'] for h in state['history'][-2:]]
        if last_two[0] == last_two[1] == action.verdict:
            breakdown.loop_penalty = -0.10

    total = sum(vars(breakdown).values())
    
    # Clamp to strictly (0, 1)
    total = _clamp(total)
    
    return Reward(total=total, breakdown=breakdown, message='Medium review graded',
                  is_terminal=(action.verdict in ['approve','request_changes']))