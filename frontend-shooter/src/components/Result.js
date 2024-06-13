import React, { useEffect, useState } from 'react';
import { useLocation } from 'react-router-dom';
import axios from 'axios';

function Results() {
  const [score, setScore] = useState(0);
  const location = useLocation();
  const user = location.state?.user;

  useEffect(() => {
    axios.post('http://localhost:5003/results', { idUsuario: 1 })
      .then(response => {
        setScore(response.data.score);
      })
      .catch(error => {
        console.error('Failed to fetch results:', error);
      });
  }, []);

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-900 text-white p-4">
      <div className="text-center">
        <h1 className="text-3xl font-bold mb-2">Exam Results</h1>
        <p className="text-xl">{user}, your score is: <span className="font-bold text-green-400">{score}</span></p>
      </div>
    </div>
  );
}

export default Results;
