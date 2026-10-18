import os
import io
import json
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Suppress verbose TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

app = FastAPI(
    title="OccuScan AI - Ocular Disease Recognition Platform",
    description="Deep Learning Multi-Disease Screening for Retinal Fundus Images",
    version="1.0.0"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_resnet50_model.keras")
CLASS_MAP_PATH = os.path.join(BASE_DIR, "class_indices.json")
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Load Class Mapping
with open(CLASS_MAP_PATH, "r") as f:
    CLASS_NAMES = json.load(f)

# Medical Insights Database
DISEASE_INSIGHTS = {
    "Normal": {
        "tag": "Healthy",
        "color": "#10b981", # Emerald
        "urgency": "Routine",
        "description": "Retinal structures, optic disc, macula, and vascular architecture appear normal with no significant pathological lesions.",
        "action": "Annual routine eye examination recommended to maintain optimal visual health."
    },
    "Cataract": {
        "tag": "Lens Opacity",
        "color": "#f59e0b", # Amber
        "urgency": "Consultation Advised",
        "description": "Clouding or opacity detected in ocular optical media, scattering light transmission to retina.",
        "action": "Slit-lamp examination recommended with an ophthalmologist for visual acuity assessment and surgical evaluation."
    },
    "Diabetes": {
        "tag": "Diabetic Retinopathy",
        "color": "#ef4444", # Red
        "urgency": "High Priority",
        "description": "Retinal microvascular indicators detected (microaneurysms, hemorrhages, or lipid exudates).",
        "action": "Urgent dilated eye exam by retinal specialist. Strict glycemic and blood pressure monitoring advised."
    },
    "Glaucoma": {
        "tag": "Optic Neuropathy",
        "color": "#8b5cf6", # Purple
        "urgency": "High Priority",
        "description": "Increased cup-to-disc ratio and structural neuroretinal rim thinning suspicious for glaucomatous damage.",
        "action": "Tonometry (IOP check), visual field testing (perimetry), and OCT optic nerve scan strongly recommended."
    },
    "Age-related Macular Degeneration (AMD)": {
        "tag": "Maculopathy",
        "color": "#ec4899", # Pink
        "urgency": "Specialist Review",
        "description": "Macular changes, drusen deposits, or pigmentary irregularities detected in the central retina.",
        "action": "Macular Optical Coherence Tomography (OCT) and Amsler grid self-monitoring recommended."
    },
    "Hypertension": {
        "tag": "Vascular Retinopathy",
        "color": "#f97316", # Orange
        "urgency": "Clinical Review",
        "description": "Arteriolar narrowing, focal constriction, or vascular tortuosity consistent with hypertensive changes.",
        "action": "Systemic cardiovascular checkup and blood pressure management recommended."
    },
    "Myopia": {
        "tag": "High Myopia",
        "color": "#06b6d4", # Cyan
        "urgency": "Routine Care",
        "description": "Chorioretinal atrophy, peripapillary crescent, or myopic stretching patterns observed.",
        "action": "Refraction test and periodic peripheral retina inspection to monitor for retinal detachment risk."
    },
    "Others": {
        "tag": "Non-Specific Finding",
        "color": "#64748b", # Slate
        "urgency": "Investigate",
        "description": "Unclassified fundus anomalies, scarring, or non-conforming retinal lesions observed.",
        "action": "Comprehensive clinical evaluation by an eye care specialist for diagnostic confirmation."
    }
}

# Global cached model
_MODEL = None

def get_model():
    global _MODEL
    if _MODEL is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(f"Trained model not found at: {MODEL_PATH}")
        print(f"Loading ResNet50 model from: {MODEL_PATH}")
        _MODEL = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully into memory.")
    return _MODEL

def preprocess_pil_image(image: Image.Image, target_size=(224, 224)):
    """Convert PIL image to preprocessed float32 tensor."""
    img = image.convert("RGB")
    img = img.resize(target_size)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

import base64
import cv2

def generate_gradcam(pil_image: Image.Image, img_tensor: np.ndarray, top_class_idx: int) -> str:
    """Generate Grad-CAM activation heatmap overlay as base64 JPEG data URL."""
    try:
        model = get_model()
        resnet = model.layers[0]
        head_layers = model.layers[1:]
        conv_layer = resnet.get_layer('conv5_block3_out')
        grad_model = tf.keras.models.Model(inputs=resnet.inputs, outputs=[conv_layer.output, resnet.output])

        with tf.GradientTape() as tape:
            conv_outs, resnet_feats = grad_model(img_tensor)
            tape.watch(conv_outs)
            out = resnet_feats
            for layer in head_layers:
                out = layer(out)
            loss = out[:, top_class_idx]

        grads = tape.gradient(loss, conv_outs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        cam = conv_outs[0] @ pooled_grads[..., tf.newaxis]
        cam = tf.squeeze(cam).numpy()
        cam = np.maximum(cam, 0)
        max_val = np.max(cam)
        if max_val > 0:
            cam = cam / max_val

        # Resize heatmap to match image dimensions
        orig_w, orig_h = pil_image.size
        cam_resized = cv2.resize(cam, (orig_w, orig_h))
        heatmap = np.uint8(255 * cam_resized)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        orig_bgr = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(orig_bgr, 0.6, heatmap_color, 0.4, 0)

        _, buffer = cv2.imencode(".jpg", overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        b64_str = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/jpeg;base64,{b64_str}"
    except Exception as e:
        print(f"Grad-CAM generation error: {e}")
        return ""

def compute_diagnosis(img_array: np.ndarray, pil_img: Image.Image = None):
    """Run model inference and generate clinical diagnostic payload with Grad-CAM."""
    model = get_model()
    raw_probs = model.predict(img_array, verbose=0)[0]
    
    breakdown = []
    for idx, prob in enumerate(raw_probs):
        name = CLASS_NAMES.get(str(idx), f"Class {idx}")
        meta = DISEASE_INSIGHTS.get(name, {
            "tag": "General",
            "color": "#3b82f6",
            "urgency": "Review",
            "description": "",
            "action": ""
        })
        breakdown.append({
            "class_id": idx,
            "disease": name,
            "confidence": float(prob),
            "percentage": round(float(prob) * 100, 2),
            "tag": meta["tag"],
            "color": meta["color"]
        })

    # Sort descending by confidence
    breakdown.sort(key=lambda x: x["confidence"], reverse=True)
    top = breakdown[0]
    meta = DISEASE_INSIGHTS.get(top["disease"], {})

    # Generate Grad-CAM for the top predicted disease
    gradcam_url = ""
    if pil_img is not None:
        gradcam_url = generate_gradcam(pil_img, img_array, top["class_id"])

    return {
        "primary_diagnosis": top["disease"],
        "confidence": top["confidence"],
        "confidence_pct": top["percentage"],
        "urgency": meta.get("urgency", "Review Required"),
        "color": meta.get("color", "#10b981"),
        "description": meta.get("description", ""),
        "clinical_action": meta.get("action", ""),
        "gradcam_url": gradcam_url,
        "breakdown": breakdown
    }

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = os.path.join(TEMPLATES_DIR, "index.html")
    with open(index_file, "r", encoding="utf-8") as f:
        return f.read()

@app.get("/api/info")
async def get_system_info():
    return {
        "model_architecture": "ResNet50 + Deep Dense Head (3x512)",
        "input_resolution": "224x224 RGB",
        "classes_count": len(CLASS_NAMES),
        "classes": list(CLASS_NAMES.values()),
        "status": "Online and Operational"
    }

@app.post("/api/predict")
async def predict_uploaded_image(file: UploadFile = File(...)):
    """Handle user uploaded image file and return clinical prediction."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid image (.jpg, .jpeg, .png).")
    
    try:
        image_bytes = await file.read()
        pil_img = Image.open(io.BytesIO(image_bytes))
        img_tensor = preprocess_pil_image(pil_img)
        diagnosis = compute_diagnosis(img_tensor, pil_img)
        diagnosis["filename"] = file.filename
        return JSONResponse(content=diagnosis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

class SampleRequest(BaseModel):
    sample_name: str

@app.post("/api/predict-sample")
async def predict_sample_image(req: SampleRequest):
    """Predict on one of the pre-bundled clinical sample images."""
    sample_file = os.path.join(STATIC_DIR, "samples", f"{req.sample_name}.png")
    if not os.path.exists(sample_file):
        raise HTTPException(status_code=404, detail=f"Sample '{req.sample_name}' not found.")
    
    try:
        pil_img = Image.open(sample_file)
        img_tensor = preprocess_pil_image(pil_img)
        diagnosis = compute_diagnosis(img_tensor, pil_img)
        diagnosis["filename"] = f"{req.sample_name}.png"
        diagnosis["sample_url"] = f"/static/samples/{req.sample_name}.png"
        return JSONResponse(content=diagnosis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
