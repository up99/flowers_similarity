from flask import Flask, request, jsonify
import numpy as np
import pickle
import os
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.resnet import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
from collections import OrderedDict

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
app.json.sort_keys = False
# --- Load model, embeddings and image paths ---
MODEL_PATH = 'models/feature_extractor.h5'
EMBEDDINGS_PATH = 'embeddings/embeddings.npy' # поиск должен производиться по тестовой выборке датасета (имитирующего библиотеку)
IMAGE_PATHS_PATH = 'image_paths/image_paths.pkl' # поиск должен производиться по тестовой выборке датасета (имитирующего библиотеку)

print("Loading feature extractor model...")
feature_extractor = load_model(MODEL_PATH)

print("Loading embeddings and image paths...")
features_array = np.load(EMBEDDINGS_PATH)
with open(IMAGE_PATHS_PATH, 'rb') as f:
    image_paths = pickle.load(f)


def extract_features(img_path):
    """Extract embedding from input image."""
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    return feature_extractor.predict(img_data).flatten()


def find_similar_images(input_img_path, top_n=5):
    """
    Find top-N most similar images (excluding self).
    Returns dict: {image_path: similarity_score}
    """
    input_features = extract_features(input_img_path)
    input_features = input_features.reshape(1, -1)

    similarities = cosine_similarity(input_features, features_array).flatten()
    matches = [(img_path, round(float(similarities[i]),2)) for i, img_path in enumerate(image_paths)]

    # Sort descending by score
    matches.sort(key=lambda x: x[1], reverse=True)

    # Exclude input image itself if present
    input_abs = os.path.abspath(input_img_path)
    filtered_matches = [m for m in matches if os.path.abspath(m[0]) != input_abs]

    # Take top N
    top_matches = filtered_matches[:top_n]
    result_dict  = OrderedDict(top_matches)
    return result_dict


# --- Flask Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        # Save uploaded image temporarily
        temp_path = os.path.join('uploads', 'temp.jpg')
        os.makedirs('uploads', exist_ok=True)
        file.save(temp_path)

        # Get results
        result = find_similar_images(temp_path)

        # Clean up
        os.remove(temp_path)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# --- Run App ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)