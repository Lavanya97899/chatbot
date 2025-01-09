CHATBOT USING NLP

```markdown
# Intent-based Chatbot with NLP and Logistic Regression

## Overview
This project implements a chatbot capable of recognizing user intents and providing dynamic responses. Using NLP techniques and Logistic Regression, the chatbot also includes additional features like weather analysis and sentiment analysis. It is deployed through a user-friendly interface built with Streamlit.

## Features
- **Intent Recognition:** The chatbot identifies user intent based on input text and responds accordingly.
- **Dynamic Responses:** Based on the recognized intent, the chatbot provides contextually relevant replies.
- **Weather Analysis:** Fetches real-time weather data using an external API.
- **Sentiment Analysis:** Analyzes the sentiment of user input and adjusts responses based on emotional tone.
- **Streamlit Interface:** A responsive interface for seamless interaction and real-time responses.

## Installation
To run the chatbot locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/chatbot-repository.git
   ```

2. Navigate to the project directory:
   ```bash
   cd chatbot-repository
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## Requirements
- Python 3.x
- NLTK
- Scikit-learn
- Streamlit
- Requests (for API integration)

## Dataset
The chatbot is trained on a labeled dataset of intents, patterns, and responses stored in a JSON file. You can modify the dataset to add new intents or responses.

## License
This project is licensed under the MIT License 

## Acknowledgements
- **NLTK:** For natural language processing and text preprocessing.
- **Scikit-learn:** For Logistic Regression and machine learning functionality.
- **Streamlit:** For deploying the chatbot with an interactive web interface.


