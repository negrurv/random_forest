// frontend/app.js

document.getElementById('prediction-form').addEventListener('submit', async function(event) {
    // Prevent the page from refreshing when you click submit
    event.preventDefault();

    // 1. Grab the values from the input fields
    const homeScored = parseFloat(document.getElementById('home-scored').value);
    const homeConceded = parseFloat(document.getElementById('home-conceded').value);
    const awayScored = parseFloat(document.getElementById('away-scored').value);
    const awayConceded = parseFloat(document.getElementById('away-conceded').value);

    // 2. Format them exactly how our Python FastAPI expects them
    const requestData = {
        samples: [
            [homeScored, homeConceded, awayScored, awayConceded]
        ]
    };

    try {
        // 3. Send the POST request to your local server
        const response = await fetch('https://lionfish-app-jcks5.ondigitalocean.app/predict', {
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
        
        // 4. Update the UI with the result!
        const prediction = data.predictions[0];
        
        document.getElementById('result-container').classList.remove('hidden');
        // Round to 2 decimal places for a cleaner look
        document.getElementById('prediction-value').innerText = prediction.toFixed(2);

    } catch (error) {
        console.error("Error fetching prediction:", error);
        alert("Failed to reach the API. Is your Uvicorn server running?");
    }
});
