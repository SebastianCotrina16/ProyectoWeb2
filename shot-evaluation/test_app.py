"""
This module contains the unit tests for the Flask bullet detection application.
"""
import unittest
import os
import io
from PIL import Image
from flask import json
from app import app


class BulletDetectionTestCase(unittest.TestCase):
    """
    Unit test suite for the Flask bullet detection application.
    """

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    def tearDown(self):
        files = os.listdir(app.config['UPLOAD_FOLDER'])
        for file in files:
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], file))
        os.rmdir(app.config['UPLOAD_FOLDER'])

    def test_no_image_uploaded(self):
        """
        Test the error response when no image is uploaded.
        """
        response = self.app.post('/detect', data={})
        data = json.loads(response.data)

        self.assertEqual(response.status_code, 400)
        self.assertEqual(data['status'], 'error')
        self.assertEqual(data['message'], 'El campo de imagen es obligatorio')

    def test_detection_with_invalid_image(self):
        """
        Test error handling when uploading a non-image file.
        """
        response = self.app.post('/detect', data={
            'image': (io.BytesIO(b"Not an image"), 'test.txt')
        }, content_type='multipart/form-data')

        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 400)
        self.assertEqual(data['status'], 'error')
        self.assertEqual(data['message'], 'El archivo no es una imagen válida')
    def test_no_bullet_detected(self):
        """
        Test the behavior when no bullets are detected in the image.
        """
        img = Image.new('RGB', (1335, 1920), color=(73, 109, 137))
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        response = self.app.post('/detect', data={
            'image': (img_byte_arr, 'no_bullets_image.png')
        }, content_type='multipart/form-data')

        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'error')
        self.assertEqual(data['message'], 'No se detectaron balas')

if __name__ == '__main__':
    unittest.main()
