import axios from 'axios';
import React, { useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Webcam from 'react-webcam';

function FacialRecognition() {
  const webcamRef = useRef(null);
  const navigate = useNavigate();
  const [dni, setDni] = useState('');
  const [name, setName] = useState('');
  const [askForDni, setAskForDni] = useState(false);
  const [capturedImage, setCapturedImage] = useState(null);

  const capture = async () => {
    const imageSrc = webcamRef.current?.getScreenshot();
    if (imageSrc) {
      setCapturedImage(imageSrc);
      const blob = await fetch(imageSrc).then(res => res.blob());
      try {
        const formData = new FormData();
        formData.append('image', blob);

        const response = await axios.post('https://facialrecognitionsistemabalas.azurewebsites.net/recognize', formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
        });
        console.log('Response:', response.data.user);
        if (response.data.user) {
          navigate('/dashboard', { state: { user: response.data.user.name } });
        } else {
          setAskForDni(true);
        }
      } catch (error) {
        console.error('Error:', error);
        setAskForDni(true);
      }
    } else {
      console.error('Webcam not mounted or unable to capture image');
    }
  };

  const handleDniSubmit = async (event) => {
    event.preventDefault();
    if (capturedImage) {
      const blob = await fetch(capturedImage).then(res => res.blob());
      const formData = new FormData();
      formData.append('image', blob);
      formData.append('dni', dni);
      formData.append('name', name);

      try {
        await axios.post('https://facialrecognitionsistemabalas.azurewebsites.net/register', formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
        });
        setAskForDni(false);
        setDni('');
        setName('');
        setCapturedImage(null);
        capture(); // Reinicia el proceso de captura
      } catch (error) {
        console.error('Error:', error);
      }
    } else {
      console.error('No captured image available');
    }
  };

  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-800">
      <div className="text-center">
        <h1 className="text-3xl font-bold text-white mb-4">Facial Recognition in Progress...</h1>
        {askForDni ? (
          <form onSubmit={handleDniSubmit}>
            <label htmlFor="dni" className="text-white">Ingrese su DNI:</label>
            <input
              type="text"
              id="dni"
              value={dni}
              onChange={e => setDni(e.target.value)}
              required
              className="block w-full p-2 border border-gray-300 rounded mt-2 mb-4"
            />
            <label htmlFor="name" className="text-white">Ingrese su Nombre:</label>
            <input
              type="text"
              id="name"
              value={name}
              onChange={e => setName(e.target.value)}
              required
              className="block w-full p-2 border border-gray-300 rounded mt-2 mb-4"
            />
            <button type="submit" className="bg-blue-500 text-white px-4 py-2 rounded">Guardar</button>
          </form>
        ) : (
          <>
            <Webcam
              audio={false}
              ref={webcamRef}
              screenshotFormat="image/jpeg"
              width={1280}
              height={720}
              videoConstraints={{ width: 1280, height: 720, facingMode: "user" }}
              className="rounded-lg shadow-xl"
            />
            <button onClick={capture} className="mt-4 px-4 py-2 bg-blue-500 text-white rounded-lg">Capture</button>
          </>
        )}
      </div>
    </div>
  );
}

export default FacialRecognition;
