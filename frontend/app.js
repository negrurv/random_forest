// frontend/app.js

document.getElementById('prediction-form').addEventListener('submit', async function(event) {
    event.preventDefault();

    const homeScored = parseFloat(document.getElementById('home-scored').value);
    const homeConceded = parseFloat(document.getElementById('home-conceded').value);
    const awayScored = parseFloat(document.getElementById('away-scored').value);
    const awayConceded = parseFloat(document.getElementById('away-conceded').value);

    const requestData = {
        samples: [
            [homeScored, homeConceded, awayScored, awayConceded]
        ]
    };

    try {
        // Note: DigitalOcean live deployment is currently archived to save cloud costs.
        // Fetching from local Uvicorn/FastAPI server instead.
        const response = await fetch('http://localhost:8000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        
        const prediction = data.predictions[0];
        
        document.getElementById('result-container').classList.remove('hidden');
        document.getElementById('prediction-value').innerText = prediction.toFixed(2);

    } catch (error) {
        console.error("Error: ", error);
        alert("Failed to reach the API. Is your Uvicorn server running?");
    }
});
