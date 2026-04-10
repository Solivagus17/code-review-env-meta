from models import Action, Reward, RewardBreakdown

# Strict bounds — scores must be strictly inside (0, 1)
SCORE_MIN = 0.001
SCORE_MAX = 0.999

def _clamp(value: float) -> float:
    """Clamp to strictly (0, 1)."""
    return max(SCORE_MIN, min(SCORE_MAX, value))

def grade(action: Action, state: dict, step: int) -> Reward:
    gt        = state['ground_truth']
    issues    = gt['issues']
    breakdown = RewardBreakdown()

    # 1. Verdict accuracy (0.20 for hard)
    breakdown.verdict_accuracy = 0.20 if action.verdict == gt['verdict'] else 0.01

    # 2. Security issue detection (0.40 detection + 0.20 specific security weighting)
    # Security issues carry 2x multiplier for `category == 'security'`.
    # Exact severity = 1x, adjacent = 0.5x. Must have >=1 suggested_fix for full credit.
    total_weight = 0
    found_weight = 0
    
    SEVERITY_LEVELS = ['info', 'low', 'medium', 'high', 'critical']
    
    for issue in issues:
        weight = 2.0 if issue['category'] == 'security' else 1.0
        total_weight += weight
        
        best_match_score = 0
        for lc in action.line_comments:
            if abs(lc.line_number - issue['line']) <= 3 and lc.category == issue['category']:
                # Calculate severity match
                if lc.severity == issue['severity']:
                    sev_score = 0.99
                else:
                    try:
                        idx_true = SEVERITY_LEVELS.index(issue['severity'])
                        idx_pred = SEVERITY_LEVELS.index(lc.severity)
                        sev_score = 0.5 if abs(idx_true - idx_pred) == 1 else 0.01
                    except ValueError:
                        sev_score = 0.01
                
                # Full credit implies fix is provided
                has_fix = len(action.suggested_fixes) > 0
                if not has_fix:
                    sev_score *= 0.5  # Penalty if no fixes suggested overall
                
                if sev_score > best_match_score:
                    best_match_score = sev_score
                    
        found_weight += (best_match_score * weight)

    # Detection ratio
    if total_weight > 0:
        detection_ratio = found_weight / total_weight
    else:
        detection_ratio = 0.01
    
    breakdown.bug_detection = max(0.01, detection_ratio * 0.40)
    breakdown.security_detection = max(0.01, detection_ratio * 0.20)

    # 3. Comment quality / Structured Report Bonus (+0.10)
    word_count = len(action.overall_comment.split())
    base_quality = min(word_count / 100, 1.0) * 0.10
    
    # Structured report bonus
    comment_lower = action.overall_comment.lower()
    has_struct = all(kw in comment_lower for kw in ['vulnerability', 'line', 'exploit', 'fix'])
    if has_struct:
        base_quality += 0.10
    
    breakdown.comment_quality = max(0.01, base_quality)

    # 4. Fix suggestions (0.10)
    fix_score = max(0.01, min(len(action.suggested_fixes) * 0.05, 0.10))

    # Efficiency bonus (0.05)
    eff_score = max(0.01, 0.05 - (step * 0.01))
    breakdown.efficiency_bonus = fix_score + eff_score

    # 5. False positive penalty (-0.15 for critical FP)
    fp_penalty = 0.0
    for lc in action.line_comments:
        matched = False
        for issue in issues:
            if abs(lc.line_number - issue['line']) <= 5:
                matched = True
                break
        if not matched:
            if lc.severity == 'critical':
                fp_penalty -= 0.15
            else:
                fp_penalty -= 0.05
    
    breakdown.false_positive_penalty = max(fp_penalty, -0.4)

    # 6. Loop penalty (-0.15)
    breakdown.loop_penalty = 0.0
    if len(state.get('history', [])) >= 2:
        last_two = [h['action_type'] for h in state['history'][-2:]]
        if last_two[0] == last_two[1] == action.verdict:
            breakdown.loop_penalty = -0.15

    total = sum([
        breakdown.verdict_accuracy, 
        breakdown.bug_detection,
        breakdown.security_detection,
        breakdown.comment_quality,
        breakdown.efficiency_bonus,
        breakdown.false_positive_penalty,
        breakdown.loop_penalty
    ])
    
    # Clamp to strictly (0, 1)
    total = _clamp(total)
    
    return Reward(total=total, breakdown=breakdown, message='Hard review graded',
                  is_terminal=(action.verdict in ['approve','request_changes']))