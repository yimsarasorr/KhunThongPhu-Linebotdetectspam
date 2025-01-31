from flask import Flask, request, jsonify
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the Naive Bayes model and TFIDF vectorizer
nb_model = joblib.load('thai_spam_naive_bayes_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Function to predict whether a message is spam
def predict_spam(text):
    # Transform the input text using the TFIDF vectorizer
    text_vector = tfidf_vectorizer.transform([text])
    # Use the Naive Bayes model to make a prediction
    prediction = nb_model.predict(text_vector)
    # Return the prediction result
    return "Spam" if prediction == 1 else "Not Spam"

# Define a POST endpoint for predicting spam
@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    data = request.get_json()

    # Ensure 'message' key is in the request data
    if 'message' not in data:
        return jsonify({"error": "No message provided"}), 400

    # Get the message text
    message = data['message']

    # Make a prediction
    result = predict_spam(message)

    # Return the prediction result as JSON
    return jsonify({"message": message, "prediction": result})

@app.route('/')
def hello_world():
    return 'เข้ามาทำไม'

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
