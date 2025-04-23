import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [images, setImages] = useState([]);
  const [predictions, setPredictions] = useState([]);

  const handleImageChange = (e) => {
    setImages([...e.target.files]);
    setPredictions([]);
  };

  const handleUpload = async () => {
    const formData = new FormData();
    images.forEach(img => formData.append('images', img));

    try {
      const response = await axios.post('http://localhost:5000/predict', formData);
      setPredictions(response.data);
    } catch (error) {
      console.error("Error during prediction:", error);
    }
  };

  return (
    <div className="App" style={{ padding: 40, fontFamily: 'sans-serif' }}>
      <h1>Spine Pose Classification</h1>
      <input type="file" multiple onChange={handleImageChange} />
      <button onClick={handleUpload} style={{ marginTop: 10 }}>Upload & Predict</button>

      <div style={{ marginTop: 30 }}>
        {predictions.length > 0 && (
          <div>
            <h3>Results:</h3>
            {predictions.map((pred, index) => (
              <div key={index} style={{ marginBottom: 15 }}>
                <strong>{pred.filename}</strong><br />
                {pred.error ? (
                  <span style={{ color: 'red' }}>{pred.error}</span>
                ) : (
                  <span>
                    Prediction: <b>{pred.prediction === 1 ? 'Scoliosis' : 'Normal'}</b><br />
                    Confidence: {pred.confidence.map(c => c.toFixed(2)).join(" / ")}
                  </span>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
