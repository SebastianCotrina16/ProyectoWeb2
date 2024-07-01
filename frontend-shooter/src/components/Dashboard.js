import React, { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import axios from 'axios';

function Dashboard() {
  const [shots, setShots] = useState(0);
  const [maxShots, setMaxShots] = useState(5);
  const [selectedFile, setSelectedFile] = useState(null);
  const navigate = useNavigate();
  const location = useLocation();
  const user = location.state?.user;

  useEffect(() => {
    axios.get('http://localhost:5003/settings')
      .then(response => {
        setMaxShots(response.data.settings.numShots);
      })
      .catch(error => {
        console.error('Error fetching settings:', error);
      });
  }, []);

  useEffect(() => {
    if (shots >= maxShots) {
      navigate('/results', { state: { user } });
    }
  }, [shots, maxShots, navigate, user]);

  const handleUpload = () => {
    if (!selectedFile) {
      alert("Please select an image first!");
      return;
    }

    const formData = new FormData();
    formData.append('imageName', selectedFile);
    formData.append('idUsuario', 1);  
    formData.append('idReserva', 1);

    axios.post('http://localhost:5008/detect', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })
      .then(response => {
        console.log('Image processed:', response.data);
        navigate('/results', { state: { user } });
      })
      .catch(error => {
        console.error('Error processing image:', error);
      });
  };

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-900 text-white p-4">
      <div className="text-center">
        <h1>Examen de {user}</h1>
        <input type="file" onChange={handleFileChange} accept="image/*" />
        <button onClick={handleUpload} className="mt-4 bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">
          Upload
        </button>
        <p>Maximum shots allowed: {maxShots}</p>
      </div>
    </div>
  );
}

export default Dashboard;
