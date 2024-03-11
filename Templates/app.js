const express = require('express');
const app = express();
const bodyParser = require('body-parser');
const path = require('path');

app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

app.post('/predict', (req, res) => {
    // Perform prediction logic here and return JSON data
    const data = {
        transcription: 'Transcription text',
        result: ['Symptom 1', 'Symptom 2'],
        accurcy: ['80%', '85%'],
        reason: 'Reason text',
        medicine: 'Medicine text',
        diet: 'Diet text',
        workout: ['Exercise 1', 'Exercise 2']
    };
    res.json(data);
});

app.listen(3000, () => {
    console.log('Server is running on port 3000');
});
