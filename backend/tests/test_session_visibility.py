import uuid

from fastapi.testclient import TestClient

from sqlalchemy import select

from app.main import Assignment, SessionLocal, app

client = TestClient(app)

def auth(email, password):
    response = client.post('/auth/login', json={'email': email, 'password': password})
    assert response.status_code == 200, response.text
    return {'Authorization': f"Bearer {response.json()['token']}"}

def test_published_session_is_visible_to_every_registered_student():
    faculty = auth('faculty@rvu.edu.in', 'faculty-demo')
    title = f"Visibility test {uuid.uuid4().hex}"
    created = client.post('/lessons', headers=faculty, json={
        'title': title,
        'subject': 'Business Forecasting',
        'reference_text': 'Forecast demand using historical evidence.',
        'concepts': ['historical evidence'],
    })
    assert created.status_code == 200, created.text
    payload = created.json()
    assert payload['id'] and payload['publication_status'] == 'published'
    with SessionLocal() as session:
        assert session.scalar(select(Assignment).where(Assignment.lesson_id == payload['id'])) is None
    for student in [('student@rvu.edu.in', 'student-demo'), ('student-after-publication-' + uuid.uuid4().hex + '@rvu.edu.in', 'student-pass')]:
        if student[0].startswith('student-after'):
            assert client.post('/auth/register', json={'name': 'Late Student', 'email': student[0], 'password': student[1], 'confirm_password': student[1], 'role': 'student'}).status_code == 201
        visible = next(item for item in client.get('/lessons', headers=auth(*student)).json() if item['id'] == payload['id'])
        assert visible['title'] == title
        assert visible['subject'] == 'Business Forecasting'
        submitted = client.post(f"/lessons/{payload['id']}/submissions", headers=auth(*student), json={'text': f"Reflection from {student[0]}"})
        assert submitted.status_code == 201

def test_published_session_is_visible_without_assignment():
    faculty = auth('faculty@rvu.edu.in', 'faculty-demo')
    student = auth('student@rvu.edu.in', 'student-demo')
    created = client.post('/lessons', headers=faculty, json={
        'title': f"Private visibility test {uuid.uuid4().hex}",
        'reference_text': 'Private reference material.',
        'concepts': [],
    })
    assert created.status_code == 200
    ids = {item['id'] for item in client.get('/lessons', headers=student).json()}
    assert created.json()['id'] in ids
