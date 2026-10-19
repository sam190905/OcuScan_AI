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

try:
    import gradio as gr
    HAS_GRADIO = True
except ImportError:
    HAS_GRADIO = False
    gr = None

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

def ensure_model_weights():
    """Ensure best_resnet50_model.keras exists on disk, downloading if deployed to cloud."""
    if os.path.exists(MODEL_PATH):
        return
    
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model_url = os.environ.get("MODEL_DOWNLOAD_URL")
    hf_repo = os.environ.get("HF_MODEL_REPO", "sam190905/OcuScan-ResNet50")
    
    print(f"Model not found at {MODEL_PATH}. Checking cloud storage...")
    
    # Priority 1: Direct URL (GitHub Release asset, Cloudflare R2, S3, etc.)
    if model_url:
        print(f"Downloading weights from MODEL_DOWNLOAD_URL: {model_url}...")
        import urllib.request
        urllib.request.urlretrieve(model_url, MODEL_PATH)
        print("Download complete.")
        return
    
    # Priority 2: Hugging Face Model Hub (100% free unlimited storage)
    try:
        from huggingface_hub import hf_hub_download
        print(f"Downloading weights from Hugging Face Hub repository '{hf_repo}'...")
        hf_hub_download(
            repo_id=hf_repo,
            filename="best_resnet50_model.keras",
            local_dir=os.path.dirname(MODEL_PATH),
            local_dir_use_symlinks=False
        )
        print("Model downloaded successfully from Hugging Face Hub.")
    except Exception as e:
        raise RuntimeError(
            f"Model file missing at {MODEL_PATH} and automatic cloud download failed: {e}. "
            "Please set MODEL_DOWNLOAD_URL in Render environment variables or upload the model."
        )

def get_model():
    global _MODEL
    if _MODEL is None:
        ensure_model_weights()
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

# Calibrated Clinical Decision Thresholds from 10% Holdout Validation
CLINICAL_THRESHOLDS = {
    "Age-related Macular Degeneration (AMD)": 0.10,
    "Cataract": 0.18,
    "Diabetes": 0.20,
    "Glaucoma": 0.12,
    "Hypertension": 0.50,
    "Myopia": 0.20,
    "Normal": 0.35,
    "Others": 0.10
}

def compute_diagnosis(img_array: np.ndarray, pil_img: Image.Image = None):
    """Run model inference and generate clinical diagnostic payload with Grad-CAM and multi-label thresholds."""
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
        threshold = CLINICAL_THRESHOLDS.get(name, 0.20)
        is_detected = bool(float(prob) >= threshold)
        breakdown.append({
            "class_id": idx,
            "disease": name,
            "confidence": float(prob),
            "percentage": round(float(prob) * 100, 2),
            "threshold_pct": round(threshold * 100, 1),
            "is_detected": is_detected,
            "status": "DETECTED" if is_detected else "CLEAR",
            "tag": meta["tag"],
            "color": meta["color"]
        })

    # Sort descending by confidence
    breakdown.sort(key=lambda x: x["confidence"], reverse=True)
    
    # Mutual Exclusivity Rule: Normal vs Pathological Disease
    normal_item = next((item for item in breakdown if item["disease"] == "Normal"), None)
    disease_items = [item for item in breakdown if item["disease"] != "Normal"]
    
    is_normal_detected = bool(normal_item and normal_item["confidence"] >= (normal_item["threshold_pct"] / 100.0))
    top_disease = max(disease_items, key=lambda x: x["confidence"]) if disease_items else None
    
    if is_normal_detected and (top_disease is None or normal_item["confidence"] >= top_disease["confidence"]):
        # If Normal is detected: suppress all disease detections to CLEAR
        top = normal_item
        top["is_detected"] = True
        top["status"] = "DETECTED"
        for d in disease_items:
            d["is_detected"] = False
            d["status"] = "CLEAR"
        detected = [normal_item]
        co_occurring = []
    else:
        # A pathological condition takes precedence
        detected_diseases = [d for d in disease_items if d["is_detected"]]
        if detected_diseases:
            top = detected_diseases[0]
            if normal_item:
                normal_item["is_detected"] = False
                normal_item["status"] = "CLEAR"
            detected = detected_diseases
            co_occurring = [
                {
                    "disease": d["disease"],
                    "percentage": d["percentage"],
                    "threshold_pct": d["threshold_pct"],
                    "color": d["color"]
                }
                for d in detected_diseases if d["disease"] != top["disease"]
            ]
        else:
            # Fallback if no category passed threshold
            top = breakdown[0]
            detected = []
            co_occurring = []

    # Generate Grad-CAM for the leading detected disease
    gradcam_url = ""
    if pil_img is not None:
        gradcam_url = generate_gradcam(pil_img, img_array, top["class_id"])

    return {
        "primary_diagnosis": top["disease"],
        "confidence": top["confidence"],
        "confidence_pct": top["percentage"],
        "threshold_pct": top["threshold_pct"],
        "is_detected": top["is_detected"],
        "co_occurring": co_occurring,
        "total_detected_count": len(detected),
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
        "model_type": "Multi-Label Fine-Tuned (Independent Sigmoid)",
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

# ---------------------------------------------------------------------------
# Hugging Face Space Gradio Compatibility Wrapper (Optional)
# If gradio is installed, mounts fallback at /gradio; otherwise pure FastAPI
# ---------------------------------------------------------------------------
if HAS_GRADIO:
    def _gradio_analyze(pil_img):
        if pil_img is None:
            return {"error": "Please upload a valid retinal fundus image."}
        try:
            tensor = preprocess_pil_image(pil_img)
            res = compute_diagnosis(tensor, pil_img)
            return {k: v for k, v in res.items() if k not in ("gradcam_heatmap", "original_image")}
        except Exception as err:
            return {"error": str(err)}

    with gr.Blocks(title="OcuScan AI - Retinal Diagnostics") as demo:
        gr.HTML("""
        <div style="text-align: center; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; padding: 24px; background: linear-gradient(135deg, #0f172a, #1e293b); color: white; border-radius: 12px; margin-bottom: 20px;">
            <h1 style="color: #60a5fa; margin: 0 0 8px 0; font-size: 26px;">👁️ OcuScan AI - Retinal Disease Screening</h1>
            <p style="color: #94a3b8; font-size: 14px; margin: 0 0 16px 0;">Multi-Label Residual Network with Grad-CAM Explainability Heatmaps</p>
            <a href="/" target="_self" style="display: inline-block; padding: 10px 24px; background: linear-gradient(135deg, #2563eb, #7c3aed); color: white; text-decoration: none; font-weight: 600; border-radius: 8px; font-size: 14px; box-shadow: 0 4px 14px rgba(37,99,235,0.4);">
                Open Interactive Clinical Dashboard & Grad-CAM →
            </a>
        </div>
        """)
        with gr.Row():
            with gr.Column():
                gr_img = gr.Image(type="pil", label="Upload Retinal Fundus Photograph")
                gr_btn = gr.Button("Analyze Fundus Image", variant="primary")
            with gr.Column():
                gr_out = gr.JSON(label="Multi-Label Diagnostic Findings")
        
        gr_btn.click(fn=_gradio_analyze, inputs=gr_img, outputs=gr_out)

    app = gr.mount_gradio_app(app, demo, path="/gradio")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)

