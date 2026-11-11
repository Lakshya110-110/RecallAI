from io import BytesIO
import uuid

from fastapi.testclient import TestClient
from openpyxl import load_workbook

from app.main import SUPPORTED_SUBJECT, app

client = TestClient(app)

def login(email, password):
    response = client.post('/auth/login', json={'email': email, 'password': password})
    assert response.status_code == 200, response.text
    return {'Authorization': f"Bearer {response.json()['token']}"}

def test_student_registration_duplicate_and_login():
    email = f"student-{uuid.uuid4().hex}@rvu.edu.in"
    payload = {'name': 'Forecast Student', 'email': email, 'password': 'student-pass', 'confirm_password': 'student-pass', 'role': 'student'}
    created = client.post('/auth/register', json=payload)
    assert created.status_code == 201
    assert client.post('/auth/register', json=payload).status_code == 409
    logged_in = client.post('/auth/login', json={'email': email, 'password': 'student-pass'})
    assert logged_in.status_code == 200
    assert logged_in.json()['user']['role'] == 'student'

def test_rvu_edu_in_email_domain_enforcement():
    bad_reg = {
        'name': 'Outside Domain',
        'email': f"user-{uuid.uuid4().hex}@gmail.com",
        'password': 'student-pass',
        'confirm_password': 'student-pass',
        'role': 'student',
    }
    res = client.post('/auth/register', json=bad_reg)
    assert res.status_code == 422
    assert '@rvu.edu.in' in str(res.json())

    bad_login = client.post('/auth/login', json={'email': 'someone@gmail.com', 'password': 'student-pass'})
    assert bad_login.status_code == 422
    assert '@rvu.edu.in' in str(bad_login.json())

def test_faculty_and_contributor_require_same_signup_code(monkeypatch):
    monkeypatch.setenv('FACULTY_SIGNUP_CODE', 'rvu-shared-secret')
    fac_email = f"faculty-{uuid.uuid4().hex}@rvu.edu.in"
    fac_payload = {'name': 'Forecast Faculty', 'email': fac_email, 'password': 'faculty-pass', 'confirm_password': 'faculty-pass', 'role': 'faculty'}
    assert client.post('/auth/register', json=fac_payload).status_code == 403
    fac_payload['signup_code'] = 'wrong-code'
    assert client.post('/auth/register', json=fac_payload).status_code == 403
    fac_payload['signup_code'] = 'rvu-shared-secret'
    fac_created = client.post('/auth/register', json=fac_payload)
    assert fac_created.status_code == 201
    assert fac_created.json()['user']['role'] == 'faculty'

    contrib_email = f"contrib-{uuid.uuid4().hex}@rvu.edu.in"
    contrib_payload = {'name': 'Forecast Contributor', 'email': contrib_email, 'password': 'contrib-pass', 'confirm_password': 'contrib-pass', 'role': 'contributor'}
    assert client.post('/auth/register', json=contrib_payload).status_code == 403
    contrib_payload['signup_code'] = 'rvu-shared-secret'
    contrib_created = client.post('/auth/register', json=contrib_payload)
    assert contrib_created.status_code == 201
    assert contrib_created.json()['user']['role'] == 'contributor'

def test_student_cannot_access_faculty_endpoints():
    headers = login('student@rvu.edu.in', 'student-demo')
    assert client.get('/faculty/export.xlsx', headers=headers).status_code == 403
    assert client.post('/lessons', headers=headers, json={'title': 'Nope', 'reference_text': 'x'}).status_code == 403

def test_business_forecasting_is_only_supported_subject():
    faculty_headers = login('faculty@rvu.edu.in', 'faculty-demo')
    lessons = client.get('/lessons', headers=faculty_headers)
    assert lessons.status_code == 200
    assert lessons.json()
    assert all(item['subject'] == SUPPORTED_SUBJECT for item in lessons.json())
    invalid = {'title': 'Unsupported', 'subject': 'Accounting', 'reference_text': 'reference'}
    assert client.post('/lessons', headers=faculty_headers, json=invalid).status_code == 422
    valid = {'title': 'Forecasting Test', 'subject': SUPPORTED_SUBJECT, 'reference_text': 'A forecast estimates future business demand.'}
    created = client.post('/lessons', headers=faculty_headers, json=valid)
    assert created.status_code == 200
    assert created.json()['subject'] == SUPPORTED_SUBJECT

def test_export_identifies_subject():
    faculty_headers = login('faculty@rvu.edu.in', 'faculty-demo')
    response = client.get('/faculty/export.xlsx', headers=faculty_headers)
    assert response.status_code == 200
    workbook = load_workbook(BytesIO(response.content), read_only=True)
    assert 'Class Summary' in workbook.sheetnames
    assert 'Assessment Log' in workbook.sheetnames
    values = list(workbook['Assessment Log'].values)
    assert any(SUPPORTED_SUBJECT in row for row in values if row)
