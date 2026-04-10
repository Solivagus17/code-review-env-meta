from fastapi.testclient import TestClient
from server.app import app
import pytest

client = TestClient(app)

def test_health():
    r = client.get('/health')
    assert r.status_code == 200

def test_root():
    r = client.get('/')
    assert r.status_code == 200
    assert r.json()['env'] == 'CodeReviewEnv'

def test_reset_easy():
    r = client.post('/reset', json={'task_id': 'easy'})
    assert r.status_code == 200
    data = r.json()
    # Response follows OpenEnv protocol: {"observation": {...}, "reward": ..., "done": ...}
    obs = data['observation']
    assert 'code_diff' in obs
    assert obs['task_id'] == 'easy'
    assert data['reward'] is None  # No reward at reset
    assert data['done'] == False

def test_step_easy():
    client.post('/reset', json={'task_id': 'easy'})
    action = {
        'verdict': 'request_changes',
        'overall_comment': 'Found a SQL injection vulnerability on line 3.',
        'line_comments': [{'line_number': 3, 'comment': 'SQL injection',
                           'severity': 'critical', 'category': 'security'}],
        'suggested_fixes': ['Use parameterized queries'],
        'confidence_score': 0.9
    }
    r = client.post('/step', json=action)
    assert r.status_code == 200
    result = r.json()
    # reward must be a float strictly in (0, 1)
    assert isinstance(result['reward'], float), f"reward should be float, got {type(result['reward'])}"
    assert 0.0 < result['reward'] < 1.0, f"Score {result['reward']} not strictly in (0, 1)"
    # observation should also have a reward field
    assert 'reward' in result['observation']

def test_step_all_tasks():
    """Test that all 3 tasks return scores strictly in (0, 1)."""
    for task_id in ['easy', 'medium', 'hard']:
        client.post('/reset', json={'task_id': task_id})
        action = {
            'verdict': 'request_changes',
            'overall_comment': 'Found issues that need to be addressed before approval.',
            'line_comments': [{'line_number': 5, 'comment': 'Bug here',
                               'severity': 'high', 'category': 'bug'}],
            'suggested_fixes': ['Fix the bug'],
            'confidence_score': 0.7
        }
        r = client.post('/step', json=action)
        assert r.status_code == 200
        result = r.json()
        score = result['reward']
        assert isinstance(score, float), f"{task_id}: reward should be float, got {type(score)}"
        assert 0.0 < score < 1.0, f"{task_id}: Score {score} not strictly in (0, 1)"

def test_state():
    client.post('/reset', json={'task_id': 'medium'})
    r = client.get('/state')
    assert r.status_code == 200
    state = r.json()
    assert state['task_id'] == 'medium'
    assert 'done' in state
