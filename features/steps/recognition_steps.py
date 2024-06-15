from behave import given, when, then
from app import app, SessionLocal, User, Base, engine
from flask import json

@given('a user with DNI "{dni}" and name "{name}" is registered with image "{image_path}"')
def step_given_user_registered(context, dni, name, image_path):
    with open(image_path, 'rb') as img:
        response = context.client.post('/register', data={
            'dni': dni,
            'name': name,
            'image': img
        })
        response_data = json.loads(response.data)
        assert response_data['success'] == True

@when('I upload the recognition image "{image_path}"')
def step_when_upload_recognition_image(context, image_path):
    with open(image_path, 'rb') as img:
        context.response = context.client.post('/recognize', data={'image': img})
        context.response_data = json.loads(context.response.data)

@then('the user with DNI "{dni}" should be recognized')
def step_then_user_should_be_recognized(context, dni):
    assert context.response_data['user']['dni'] == dni

@then('the recognized user name should be "{name}"')
def step_then_recognized_user_name_should_be(context, name):
    assert context.response_data['user']['name'] == name

@then('no user should be recognized')
def step_then_no_user_should_be_recognized(context):
    assert context.response_data['user'] is None
