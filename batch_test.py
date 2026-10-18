import os
import sys
import json
import numpy as np
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

from predict import get_model, preprocess_image, CLASS_NAMES

# Samples to test across different categories
DATASET_DIR = r"C:\Users\SAMARTH\Downloads\archive (2)\datasets"

test_samples = [
    ("Normal (Sample 1)", os.path.join(DATASET_DIR, "normal", "image1.png")),
    ("Normal (Sample 2)", os.path.join(DATASET_DIR, "normal", "image10.png")),
    ("Diabetes (Sample 1)", os.path.join(DATASET_DIR, "diabetes", "image0.png")),
    ("Diabetes (Sample 2)", os.path.join(DATASET_DIR, "diabetes", "image5.png")),
    ("Glaucoma (Sample 1)", os.path.join(DATASET_DIR, "glaucoma", "image1.png")),
    ("Cataract (Sample 1)", os.path.join(DATASET_DIR, "cataract", "image1.png")),
    ("Myopia (Sample 1)", os.path.join(DATASET_DIR, "myopia", "image1.png")),
    ("AMD (Sample 1)", os.path.join(DATASET_DIR, "ageDegeneration", "image1.png")),
]

model = get_model()

print("\n" + "=" * 80)
print(f"{'IMAGE LABEL':<25} | {'PREDICTED DIAGNOSIS':<35} | {'CONFIDENCE':<10}")
print("=" * 80)

results = []
for label, path in test_samples:
    if not os.path.exists(path):
        continue
    img_array = preprocess_image(path)
    probs = model.predict(img_array, verbose=0)[0]
    top_idx = int(np.argmax(probs))
    top_pred = CLASS_NAMES.get(str(top_idx), f"Class {top_idx}")
    top_conf = float(probs[top_idx])
    
    # Store top 2
    sorted_indices = np.argsort(probs)[::-1]
    second_idx = sorted_indices[1]
    second_pred = CLASS_NAMES.get(str(second_idx), "")
    second_conf = float(probs[second_idx])

    print(f"{label:<25} | {top_pred:<35} | {top_conf*100:6.2f}%")
    results.append({
        "label": label,
        "top_pred": top_pred,
        "top_conf": top_conf,
        "second_pred": second_pred,
        "second_conf": second_conf
    })

print("=" * 80)
print("\nTop 2 Predictions Breakdown:")
for r in results:
    print(f"[{r['label']}]")
    print(f"   1st: {r['top_pred']} ({r['top_conf']*100:.2f}%)")
    print(f"   2nd: {r['second_pred']} ({r['second_conf']*100:.2f}%)")
    print()
