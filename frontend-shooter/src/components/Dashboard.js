import axios from 'axios';
import React, { useEffect, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

function Dashboard() {
  const [maxShots, setMaxShots] = useState(5);
  const [selectedFile, setSelectedFile] = useState(null);
  const [loading, setLoading] = useState(false);
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

  const handleUpload = () => {
    if (!selectedFile) {
      alert("Por favor selecciona una imagen primero.");
      return;
    }

    setLoading(true);

    const formData = new FormData();
    formData.append('image', selectedFile); 
    formData.append('idUsuario', 1);      

    axios.post('http://localhost:5008/detect', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })
      .then(response => {
        const disparosDetectados = response.data.disparos.length;

        if (disparosDetectados === maxShots) {
          navigate('/results', { state: { user, results: response.data } });
        } else if (disparosDetectados < maxShots) {
          alert(`Faltan disparos. Se detectaron ${disparosDetectados} disparos de ${maxShots} permitidos.`);
        } else {
          alert(`Se detectaron más disparos de los permitidos. Se detectaron ${disparosDetectados} de ${maxShots}.`);
        }
      })
      .catch(error => {
        console.error('Error procesando la imagen:', error);
        alert('Error procesando la imagen, por favor intenta nuevamente.');
      })
      .finally(() => {
        setLoading(false);
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
          {loading ? 'Subiendo...' : 'Subir Imagen'}
        </button>
        <p>Máximo de disparos permitidos: {maxShots}</p>
      </div>
    </div>
  );
}

export default Dashboard;


