"""
This module contains the unit tests for the Flask application.
"""
import unittest
from flask import json
from app import app, SessionLocal, User, Base, engine  # pylint: disable=E0401


class FaceRecognitionTestCase(unittest.TestCase):
    """
    Unit tests suite for the Flask application.
    """

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

        Base.metadata.create_all(bind=engine)
        self.session = SessionLocal()

    def tearDown(self):
        self.session.close()
        Base.metadata.drop_all(bind=engine)

    def test_register_user(self):
        """
        Test that a user can be registered in the database.
        """
        with open('Fotos/SebastianCotrina.png', 'rb') as img:
            response = self.app.post('/register', data={
                'dni': '12345678',
                'name': 'John Doe',
                'image': img
            })

        data = json.loads(response.data)
        self.assertTrue(data['success'])

        user = self.session.query(User).filter_by(dni='12345678').first()
        self.assertIsNotNone(user)
        self.assertEqual(user.name, 'John Doe')

    def test_recognize_user(self):
        """
        Test that a registered user can be recognized.
        """
        with open('Fotos/Keanu.webp', 'rb') as img:
            self.app.post('/register', data={
                'dni': '12345678',
                'name': 'John Doe',
                'image': img
            })

        with open('Fotos/Keanu2.webp', 'rb') as img:
            response = self.app.post('/recognize', data={'image': img})

        data = json.loads(response.data)
        self.assertIsNotNone(data['user'])
        self.assertEqual(data['user']['dni'], '12345678')
        self.assertEqual(data['user']['name'], 'John Doe')

    def test_recognize_unknown_user(self):
        """
        Test that an unknown user is not recognized.
        """
        with open('Fotos/Adam.jpg', 'rb') as img:
            response = self.app.post('/recognize', data={'image': img})

        data = json.loads(response.data)
        self.assertIsNone(data['user'])


if __name__ == '__main__':
    unittest.main()

