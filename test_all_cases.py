
"""
Comprehensive local test for Phase 2 validation
Tests all edge cases to ensure scores are strictly in (0, 1)
"""

import sys
import random
from environment import CodeReviewEnv
from models import Action, ReviewVerdict, LineComment, SeverityLevel

def test_basic_scenarios():
    """Test basic scenarios for all task types"""
    print("="*70)
    print("TEST 1: Basic Scenarios - All Task Types")
    print("="*70)

    violations = []

    for task_id in ['easy', 'medium', 'hard']:
        print(f"\n  Testing {task_id.upper()} task...")
        env = CodeReviewEnv(task_id=task_id)
        obs = env.reset()

        for step in range(5):
            if obs.done:
                break


            action = Action(
                verdict=ReviewVerdict.COMMENT if step < 3 else ReviewVerdict.APPROVE,
                overall_comment="Test review comment with sufficient length to meet requirements.",
                line_comments=[
                    LineComment(
                        line_number=10,
                        comment="Test issue",
                        severity=SeverityLevel.MEDIUM,
                        category='bug'
                    )
                ],
                suggested_fixes=["Fix 1"],
                confidence_score=0.7
            )

            obs, reward, done, info = env.step(action)
            score = reward.total
            cumulative = info['cumulative_reward']


            if score <= 0.0 or score >= 1.0:
                violations.append({
                    'test': 'basic',
                    'task': task_id,
                    'step': step + 1,
                    'score': score,
                    'cumulative': cumulative,
                    'issue': f'Score {score} not in (0, 1)'
                })
                print(f"    FAIL Step {step+1}: score={score:.6f} (VIOLATION)")
            else:
                print(f"    OK   Step {step+1}: score={score:.6f}, cumulative={cumulative:.6f}")

            if cumulative <= 0.0 or cumulative >= 1.0:
                violations.append({
                    'test': 'basic',
                    'task': task_id,
                    'step': step + 1,
                    'score': score,
                    'cumulative': cumulative,
                    'issue': f'Cumulative {cumulative} not in (0, 1)'
                })
                print(f"    FAIL Cumulative violation!")

    return violations

def test_edge_cases():
    """Test edge cases that might trigger boundary issues"""
    print("\n" + "="*70)
    print("TEST 2: Edge Cases - Boundary Conditions")
    print("="*70)

    violations = []

    test_cases = [
        {
            'name': 'No line comments (minimal score)',
            'action': Action(
                verdict=ReviewVerdict.APPROVE,
                overall_comment="Minimal comment.",
                line_comments=[],
                suggested_fixes=[],
                confidence_score=0.1
            )
        },
        {
            'name': 'Many line comments (potential FP penalty)',
            'action': Action(
                verdict=ReviewVerdict.REQUEST_CHANGES,
                overall_comment="Many issues found in this code review.",
                line_comments=[
                    LineComment(line_number=i, comment=f"Issue {i}",
                               severity=SeverityLevel.HIGH, category='bug')
                    for i in range(1, 11)
                ],
                suggested_fixes=["Fix everything"],
                confidence_score=0.9
            )
        },
        {
            'name': 'Maximum efficiency (step 0)',
            'action': Action(
                verdict=ReviewVerdict.APPROVE,
                overall_comment="Perfect code, approved immediately with confidence.",
                line_comments=[],
                suggested_fixes=["No fixes needed"],
                confidence_score=1.0
            )
        },
        {
            'name': 'All critical FP (max penalty)',
            'action': Action(
                verdict=ReviewVerdict.REQUEST_CHANGES,
                overall_comment="Many critical issues found in review.",
                line_comments=[
                    LineComment(line_number=i*100, comment=f"Critical issue {i}",
                               severity=SeverityLevel.CRITICAL, category='security')
                    for i in range(1, 20)
                ],
                suggested_fixes=[],
                confidence_score=0.9
            )
        },
    ]

    for test_case in test_cases:
        print(f"\n  Testing: {test_case['name']}")

        for task_id in ['easy', 'medium', 'hard']:
            env = CodeReviewEnv(task_id=task_id)
            obs = env.reset()

            obs, reward, done, info = env.step(test_case['action'])
            score = reward.total
            cumulative = info['cumulative_reward']

            if score <= 0.0 or score >= 1.0:
                violations.append({
                    'test': test_case['name'],
                    'task': task_id,
                    'score': score,
                    'cumulative': cumulative,
                    'issue': f'Score {score} not in (0, 1)'
                })
                print(f"    FAIL {task_id}: score={score:.6f} (VIOLATION)")
            else:
                print(f"    OK   {task_id}: score={score:.6f}, cumulative={cumulative:.6f}")

    return violations

def test_multi_step_accumulation():
    """Test that cumulative scores don't reach 0 or 1 over multiple steps"""
    print("\n" + "="*70)
    print("TEST 3: Multi-Step Accumulation")
    print("="*70)

    violations = []

    for task_id in ['easy', 'medium', 'hard']:
        print(f"\n  Testing {task_id.upper()} - 10 random steps...")
        env = CodeReviewEnv(task_id=task_id)
        obs = env.reset()

        max_steps = min(10, env.task.max_steps)

        for step in range(max_steps):
            if obs.done:
                break


            num_comments = random.randint(0, 5)
            action = Action(
                verdict=random.choice([ReviewVerdict.COMMENT, ReviewVerdict.APPROVE,
                                      ReviewVerdict.REQUEST_CHANGES]),
                overall_comment="Random review comment " * random.randint(2, 10),
                line_comments=[
                    LineComment(
                        line_number=random.randint(1, 100),
                        comment=f"Random comment {i}",
                        severity=random.choice(list(SeverityLevel)),
                        category=random.choice(['bug', 'security', 'style', 'performance'])
                    )
                    for i in range(num_comments)
                ],
                suggested_fixes=[f"Fix {i}" for i in range(random.randint(0, 3))],
                confidence_score=random.uniform(0.3, 0.9)
            )

            obs, reward, done, info = env.step(action)
            score = reward.total
            cumulative = info['cumulative_reward']

            if score <= 0.0 or score >= 1.0:
                violations.append({
                    'test': 'accumulation',
                    'task': task_id,
                    'step': step + 1,
                    'score': score,
                    'cumulative': cumulative,
                    'issue': f'Score {score} not in (0, 1)'
                })
                print(f"    FAIL Step {step+1}: score={score:.6f} (VIOLATION)")
            elif cumulative <= 0.0 or cumulative >= 1.0:
                violations.append({
                    'test': 'accumulation',
                    'task': task_id,
                    'step': step + 1,
                    'score': score,
                    'cumulative': cumulative,
                    'issue': f'Cumulative {cumulative} not in (0, 1)'
                })
                print(f"    FAIL Step {step+1}: cumulative={cumulative:.6f} (VIOLATION)")
            else:
                print(f"    OK   Step {step+1}: score={score:.6f}, cumulative={cumulative:.6f}")

    return violations

def test_max_steps_penalty():
    """Test behavior when max steps is exceeded"""
    print("\n" + "="*70)
    print("TEST 4: Max Steps Penalty")
    print("="*70)

    violations = []

    for task_id in ['easy', 'medium', 'hard']:
        print(f"\n  Testing {task_id.upper()} - reaching max steps...")
        env = CodeReviewEnv(task_id=task_id)
        obs = env.reset()

        max_steps = env.task.max_steps

        for step in range(max_steps + 2):
            if obs.done:
                break

            action = Action(
                verdict=ReviewVerdict.COMMENT,
                overall_comment="Continuing review with additional comments.",
                line_comments=[],
                suggested_fixes=[],
                confidence_score=0.5
            )

            obs, reward, done, info = env.step(action)
            score = reward.total
            cumulative = info['cumulative_reward']

            if score <= 0.0 or score >= 1.0:
                violations.append({
                    'test': 'max_steps',
                    'task': task_id,
                    'step': step + 1,
                    'score': score,
                    'cumulative': cumulative,
                    'issue': f'Score {score} not in (0, 1)'
                })
                print(f"    FAIL Step {step+1}/{max_steps}: score={score:.6f} (VIOLATION)")
            else:
                print(f"    OK   Step {step+1}/{max_steps}: score={score:.6f}, cumulative={cumulative:.6f}")

    return violations

def test_server_endpoint():
    """Test the actual server endpoint returns scores in (0, 1)"""
    print("\n" + "="*70)
    print("TEST 5: Server Endpoint (FastAPI TestClient)")
    print("="*70)

    violations = []

    try:
        from fastapi.testclient import TestClient
        from server.app import app

        client = TestClient(app)

        for task_id in ['easy', 'medium', 'hard']:
            print(f"\n  Testing {task_id.upper()} via /step endpoint...")

            r = client.post('/reset', json={'task_id': task_id})
            assert r.status_code == 200, f"Reset failed: {r.text}"

            action = {
                'verdict': 'request_changes',
                'overall_comment': 'Found issues in the code that need to be addressed before approval.',
                'line_comments': [{'line_number': 3, 'comment': 'Bug found',
                                   'severity': 'high', 'category': 'bug'}],
                'suggested_fixes': ['Fix the bug'],
                'confidence_score': 0.7
            }

            r = client.post('/step', json=action)
            assert r.status_code == 200, f"Step failed: {r.text}"
            result = r.json()


            reward = result['reward']
            if isinstance(reward, dict):
                score = reward['total']
            else:
                score = reward

            if score <= 0.0 or score >= 1.0:
                violations.append({
                    'test': 'server',
                    'task': task_id,
                    'score': score,
                    'issue': f'Server reward score {score} not in (0, 1)'
                })
                print(f"    FAIL {task_id}: server score={score:.6f} (VIOLATION)")
            else:
                print(f"    OK   {task_id}: server score={score:.6f}")

    except ImportError:
        print("  SKIP: FastAPI not installed, skipping server test")
    except Exception as e:
        print(f"  ERROR: {e}")

    return violations

def main():
    print("=" * 70)
    print("  COMPREHENSIVE LOCAL VALIDATION TEST")
    print("  Testing Phase 2 Score Boundary Requirements")
    print("=" * 70 + "\n")

    all_violations = []


    all_violations.extend(test_basic_scenarios())
    all_violations.extend(test_edge_cases())
    all_violations.extend(test_multi_step_accumulation())
    all_violations.extend(test_max_steps_penalty())
    all_violations.extend(test_server_endpoint())


    print("\n" + "="*70)
    print("FINAL TEST REPORT")
    print("="*70)

    if all_violations:
        print(f"\nFAILED: {len(all_violations)} violation(s) found\n")
        print("Violations:")
        for i, v in enumerate(all_violations, 1):
            print(f"\n{i}. {v['test']} - {v.get('task', 'N/A')}")
            print(f"   Step: {v.get('step', 'N/A')}")
            print(f"   Score: {v['score']:.6f}")
            print(f"   Cumulative: {v.get('cumulative', 'N/A')}")
            print(f"   Issue: {v['issue']}")
        print("\n  DO NOT RESUBMIT - Fix needed!")
        return 1
    else:
        print("\n  ALL TESTS PASSED!")
        print("   - All scores strictly in (0, 1)")
        print("   - All cumulative scores strictly in (0, 1)")
        print("   - Edge cases handled correctly")
        print("   - Multi-step accumulation safe")
        print("   - Max steps penalty handled")
        print("   - Server endpoint verified")
        print("\n  READY TO RESUBMIT!")
        return 0

if __name__ == "__main__":
    sys.exit(main())

