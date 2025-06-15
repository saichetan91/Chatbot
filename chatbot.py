from flask import Flask, request, jsonify, render_template
import json
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Load intents from JSON file
with open("intents.json", "r") as file:
    intents = json.load(file)

# Prepare training data
tags = []
patterns = []
responses = {}

for intent in intents["intents"]:
    tag = intent["tag"]
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        tags.append(tag)
    responses[tag] = intent["responses"]

# Train a Naive Bayes model
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(patterns, tags)

# Initialize Flask app
app = Flask(__name__)

# Home route to render index.html
@app.route("/")
def index():
    return render_template("index.html")

# Chatbot API endpoint
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message", "")
    try:
        tag = model.predict([user_input])[0]
        response = random.choice(responses[tag])
        return jsonify({"response": response})
    except Exception:
        return jsonify({"response": "Sorry, I didn't understand that. Can you rephrase?"})

if __name__ == "__main__":
    app.run(debug=True)