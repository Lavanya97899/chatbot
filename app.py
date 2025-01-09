import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import requests
from textblob import TextBlob

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load intents from the JSON file
file_path = os.path.abspath(r'C:\Users\mmbl6\OneDrive\Desktop\bot\intents (2).json')
with open(file_path, "r") as file:
    intents = json.load(file)

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Training the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

# Function to classify the chatbot response based on the intent
def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response
    return "Sorry, I didn't understand that."

# Sentiment Analysis Function
def get_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return "You're positive! How can I assist you?"
    elif polarity < 0:
        return "You're negative. How can I help you?"
    else:
        return "You're neutral. How can I assist you?"

# Weather Prediction Function
def get_weather(city):
    api_key = "2a52ecd4384424e7cb926c7dab6dd5e0"  # Replace with your actual weather API key
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    data = response.json()

    if data.get("cod") != 200:
        return "Error: City not found or issue with API."

    main_data = data.get("main", {})
    weather_data = data.get("weather", [{}])[0]

    temperature = main_data.get("temp", "N/A")
    weather_description = weather_data.get("description", "N/A")
    humidity = main_data.get("humidity", "N/A")

    return f"Temperature: {temperature}°C\nWeather: {weather_description}\nHumidity: {humidity}%"

counter = 0

def main():
    global counter
    st.title("Chatbot with Sentiment Analysis and Weather Prediction")

    # Create a sidebar menu with options
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Home Menu
    if choice == "Home":
        st.write("Welcome to the chatbot. Please type a message and press Enter to start the conversation.")

        # Check if the chat_log.csv file exists, and if not, create it with column names
        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        counter += 1
        user_input = st.text_input("You:", key=f"user_input_{counter}")
        response = ""
        sentiment = ""
        weather_info = ""

        if user_input:
            # Convert the user input to a string
            user_input_str = str(user_input)

            # Check for sentiment first (before intent response)
            sentiment = get_sentiment(user_input)

            # Check for weather query
            if "weather" in user_input.lower():
                city = user_input.split("in")[-1].strip()  # Extract city name after "in"
                weather_info = get_weather(city)
            else:
                # Get Chatbot Response based on the trained intents
                response = chatbot(user_input)

            # Display the results
            st.text_area("Chatbot:", value=response, height=120, max_chars=None, key=f"chatbot_response_{counter}")
            st.write(f"Sentiment Analysis: {sentiment}")

            if weather_info:
                st.write("Weather Report:")
                st.text(weather_info)

            # Get the current timestamp
            timestamp = datetime.datetime.now().strftime(f"%Y-%m-%d %H:%M:%S")

            # Save the user input and chatbot response to the chat_log.csv file
            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input_str, response or weather_info or sentiment, timestamp])

            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting with me. Have a great day!")
                st.stop()

    # Conversation History Menu
    elif choice == "Conversation History":
        st.header("Conversation History")
        with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)  # Skip the header row
            for row in csv_reader:
                st.text(f"User: {row[0]}")
                st.text(f"Chatbot: {row[1]}")
                st.text(f"Timestamp: {row[2]}")
                st.markdown("---")

    # About Menu
    elif choice == "About":
        st.write("The goal of this project is to create a chatbot that can understand and respond to user input based on intents. The chatbot is built using Natural Language Processing (NLP) and Logistic Regression to classify intents and respond accordingly. The project also includes sentiment analysis and weather prediction features.")
        st.subheader("Project Overview:")
        st.write("""
        The project is divided into three parts:
        1. NLP techniques using Logistic Regression to train the chatbot on labeled intents.
        2. Sentiment analysis to detect the mood of the user (positive, negative, neutral).
        3. Weather prediction by querying a weather API for the city mentioned by the user.
        """)
        st.subheader("Conclusion:")
        st.write("In this project, we built a chatbot that understands user inputs using NLP, performs sentiment analysis, and provides weather information. It was implemented using Streamlit for the user interface.")

if __name__ == '__main__':
    main()
