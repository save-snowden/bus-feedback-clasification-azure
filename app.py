from flask import Flask, request, jsonify
import joblib
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords

app = Flask(__name__)

# Load the saved model and fitted vectorizer
loaded_model = joblib.load('feedback_model.sav')
loaded_vectorizer = joblib.load('tfidf_vectorizer.sav')

# Preprocessing setup
stemmer = SnowballStemmer('english')
stop_words = set(stopwords.words('english'))

# Map the predicted numerical class back to its label
class_mapping = {'good': 0, 'bad': 1, 'emergency': 2}
class_mapping_inverse = {v: k for k, v in class_mapping.items()}

# API endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input text from the request
        input_text = request.json['feedback']

        # Preprocess the input text
        def preprocess_text(text):
            words = text.split()
            processed_words = [stemmer.stem(word) for word in words if word.lower() not in stop_words]
            return ' '.join(processed_words)

        input_text_processed = preprocess_text(input_text)

        # Vectorize the input text using the loaded vectorizer
        input_text_tfidf = loaded_vectorizer.transform([input_text_processed])

        # Make predictions using the loaded model
        predicted_class = loaded_model.predict(input_text_tfidf)

        # Map the predicted numerical class back to its label
        predicted_class_label = class_mapping_inverse[predicted_class[0]]

        # Return the result as JSON
        result = {'feedback': input_text, 'predicted_class': predicted_class_label}
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
