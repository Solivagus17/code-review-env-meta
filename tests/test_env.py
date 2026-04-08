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
    obs = r.json()
    assert 'code_diff' in obs
    assert obs['task_id'] == 'easy'

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
    assert 0.0 < result['reward']['total'] < 1.0

def test_state():
    client.post('/reset', json={'task_id': 'medium'})
    r = client.get('/state')
    assert r.status_code == 200
    state = r.json()
    assert state['task_id'] == 'medium'
    assert 'done' in state
