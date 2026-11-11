import uuid

from fastapi.testclient import TestClient

from app import main

client = TestClient(main.app)

def login(email, password):
    response = client.post('/auth/login', json={'email': email, 'password': password})
    assert response.status_code == 200, response.text
    return {'Authorization': f"Bearer {response.json()['token']}"}

def create_assigned_lesson(faculty, student_email):
    response = client.post('/lessons', headers=faculty, json={
        'title': f'Assessment test {uuid.uuid4().hex}',
        'subject': 'Business Forecasting',
        'reference_text': 'Demand forecasting uses historical demand and uncertainty.',
        'concepts': ['historical demand', 'uncertainty'],
    })
    assert response.status_code == 200, response.text
    return response.json()['id']

def test_500_character_boundary_and_count():
    faculty = login('faculty@rvu.edu.in', 'faculty-demo')
    student = login('student@rvu.edu.in', 'student-demo')
    lesson_id = create_assigned_lesson(faculty, 'student@rvu.edu.in')
    accepted = client.post(f'/lessons/{lesson_id}/submissions', headers=student, json={'text': 'a' * 498 + '!?'} )
    assert accepted.status_code == 201
    assert accepted.json()['character_count'] == 500
    rejected = client.post(f'/lessons/{lesson_id}/submissions', headers=student, json={'text': 'b' * 501})
    assert rejected.status_code == 422
    assert '500 characters' in rejected.json()['detail'][0]['msg']

def test_assessment_uses_correct_master_and_saves_structured_result(monkeypatch):
    faculty = login('faculty@rvu.edu.in', 'faculty-demo')
    student = login('student@rvu.edu.in', 'student-demo')
    lesson_id = create_assigned_lesson(faculty, 'student@rvu.edu.in')
    submitted = client.post(f'/lessons/{lesson_id}/submissions', headers=student, json={'text': 'Historical demand and uncertainty guide a forecast.'})
    submission_id = submitted.json()['id']
    captured = {}
    def fake_assessment(lesson, submission):
        captured['reference'] = lesson.reference_text
        captured['reflection'] = submission.text
        return main.NvidiaAssessment(
            recall_percentage=90,
            concept_results=[main.ConceptResult(concept='historical demand', status='recalled', evidence='Historical demand', explanation='The reflection names the historical demand driver.')],
            missing_concepts=['uncertainty'],
            feedback='Good recall of the main forecasting driver.',
        )
    monkeypatch.setattr(main, 'nvidia_assess', fake_assessment)
    result = client.post(f'/submissions/{submission_id}/assess', headers=faculty)
    assert result.status_code == 200
    body = result.json()
    assert captured['reference'] == 'Demand forecasting uses historical demand and uncertainty.'
    assert captured['reflection'] == 'Historical demand and uncertainty guide a forecast.'
    assert body['recall_percentage'] == 90
    assert body['rating'] == 4
    assert body['similarity'] is not None
    assert body['assessment_details']['concept_results'][0]['status'] == 'recalled'

def test_nvidia_failure_never_creates_completed_score(monkeypatch):
    faculty = login('faculty@rvu.edu.in', 'faculty-demo')
    student = login('student@rvu.edu.in', 'student-demo')
    lesson_id = create_assigned_lesson(faculty, 'student@rvu.edu.in')
    submission = client.post(f'/lessons/{lesson_id}/submissions', headers=student, json={'text': 'A short reflection.'})
    submission_id = submission.json()['id']
    def failed_assessment(lesson, item):
        raise RuntimeError('NVIDIA unavailable')
    monkeypatch.setattr(main, 'nvidia_assess', failed_assessment)
    failed = client.post(f'/submissions/{submission_id}/assess', headers=faculty)
    assert failed.status_code == 502
    rows = client.get('/submissions', headers=student).json()
    saved = next(row for row in rows if row['id'] == submission_id)
    assert saved['assessment_status'] == 'failed'
    assert saved['rating'] is None
    assert saved['recall_percentage'] is None
