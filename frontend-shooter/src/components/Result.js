import React from 'react';
import { useNavigate } from 'react-router-dom';

function StartExam() {
  const navigate = useNavigate();

  const handleStart = () => {
    navigate('/facial-recognition');
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen w-full">
      <div className="text-center p-12 bg-gray-800 bg-opacity-80 rounded-lg shadow-2xl">
        <h1 className="text-4xl font-bold text-white mb-6">Bienvenido a la Evaluacion de Disparo</h1>
        <p className="text-gray-300 mb-8">Preparate para rendir tu examen, dispondras de un tiempo para rendir la evaluacion.</p>
        <button
          onClick={handleStart}
          className="px-8 py-4 bg-blue-500 text-white text-lg font-bold rounded-full hover:bg-blue-600 transition-colors"
        >
          Iniciar Examen
        </button>
      </div>
    </div>
  );
}

export default StartExam;

