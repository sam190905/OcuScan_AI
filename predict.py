import os
import sys
import json
import numpy as np
from PIL import Image

# Suppress verbose TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# Default paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_resnet50_model.keras")
CLASS_MAP_PATH = os.path.join(BASE_DIR, "class_indices.json")

# Load class mapping
with open(CLASS_MAP_PATH, "r") as f:
    CLASS_NAMES = json.load(f)

# Load model lazily
_MODEL = None

def get_model():
    global _MODEL
    if _MODEL is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
        print(f"Loading model from {MODEL_PATH} ...")
        _MODEL = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully!")
    return _MODEL

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess retinal fundus image for ResNet50."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 3)
    return img_array

def predict_disease(image_path, top_k=3):
    """
    Predict ocular disease for a given fundus image.
    Returns:
        dict: top prediction, confidence, and sorted probability breakdown.
    """
    model = get_model()
    processed_img = preprocess_image(image_path)
    predictions = model.predict(processed_img, verbose=0)[0]

    # Map probabilities to classes
    results = []
    for idx, prob in enumerate(predictions):
        class_name = CLASS_NAMES.get(str(idx), f"Class {idx}")
        results.append({
            "class_id": idx,
            "disease": class_name,
            "confidence": float(prob)
        })

    # Sort descending by probability
    results.sort(key=lambda x: x["confidence"], reverse=True)
    top_prediction = results[0]

    return {
        "image": image_path,
        "prediction": top_prediction["disease"],
        "confidence": top_prediction["confidence"],
        "all_probabilities": results
    }

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_img_path = sys.argv[1]
    else:
        # Default test image if available
        test_img_path = r"C:\Users\SAMARTH\Downloads\archive (2)\datasets\normal\image0.png"
        if not os.path.exists(test_img_path):
            print("Usage: python predict.py <path_to_fundus_image>")
            sys.exit(1)

    print(f"\nAnalyzing image: {test_img_path}")
    result = predict_disease(test_img_path)

    print("\n" + "=" * 50)
    print(f" PRIMARY DIAGNOSIS: {result['prediction']}")
    print(f" CONFIDENCE:        {result['confidence'] * 100:.2f}%")
    print("=" * 50)
    print("\nDetailed Probabilities:")
    for item in result["all_probabilities"]:
        bar = "#" * int(item["confidence"] * 30)
        print(f" - {item['disease']:<38} : {item['confidence']*100:6.2f}%  {bar}")
    print("=" * 50)
