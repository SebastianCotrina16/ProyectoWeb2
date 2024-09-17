import React from 'react';
import { useLocation } from 'react-router-dom';

function Results() {
  const location = useLocation();
  const user = location.state?.user;
  const results = location.state?.results;

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-900 text-white p-4">
      <div className="text-center">
        <h1 className="text-3xl font-bold mb-2">Resultados del examen de {user}</h1>
        {results ? (
          <div>
            <p className="text-xl">Precisión final: <span className="font-bold text-green-400">{results.final_accuracy}%</span></p>
            <h3 className="mt-4 text-lg">Detalles de los disparos:</h3>
            <ul className="mt-2">
              {results.disparos.map((shot, index) => (
                <li key={index}>Impacto en {shot.ubicacion}: Precisión {shot.precision}%</li>
              ))}
            </ul>
          </div>
        ) : (
          <p>No se han encontrado resultados. Intenta nuevamente.</p>
        )}
      </div>
    </div>
  );
}

export default Results;
