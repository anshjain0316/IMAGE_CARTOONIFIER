import cv2
import numpy as np
import os
import uuid
from flask import Flask, request, send_file, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename

# Import enhanced pipeline + presets (alias main function to avoid name clashes)
from cartoonify import (
    cartoonify as apply_cartoonify,
    preset_classic,
    preset_clean_quantized,
    preset_stylization,
    preset_pencil,
)

app = Flask(__name__)

# ----------------------
# App Config
# ----------------------
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['CARTOON_FOLDER'] = 'cartoonized_images'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'webp'}
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5 MB

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['CARTOON_FOLDER'], exist_ok=True)

# ----------------------
# Helpers
# ----------------------
def allowed_file(filename: str) -> bool:
    return (
        '.' in filename
        and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']
    )

# (Kept) Legacy simple cartoon function, not used by default
def cartoonize_image(image, k=8):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    edges = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 8
    )
    data = np.float32(image).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, label, center = cv2.kmeans(
        data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )
    center = np.uint8(center)
    result = center[label.flatten()].reshape(image.shape)
    blurred = cv2.medianBlur(result, 3)
    cartoon = cv2.bitwise_and(blurred, blurred, mask=edges)
    return cartoon

# ----------------------
# Routes
# ----------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/cartoonify', methods=['POST'])
def cartoonify_route():
    # Validate file
    if 'image' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['image']
    if not (file and allowed_file(file.filename)):
        return jsonify({"error": "Invalid file"}), 400

    # Secure + unique filename
    safe_name = secure_filename(file.filename)
    name_root, ext = os.path.splitext(safe_name)
    unique_name = f"{name_root}_{uuid.uuid4().hex}{ext.lower()}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
    file.save(filepath)

    # Read image
    img = cv2.imread(filepath)
    if img is None:
        return jsonify({"error": "Could not read image"}), 400

    # ------------- Read options from form -------------
    style = (request.form.get('style') or 'classic').strip().lower()
    edge_mode = (request.form.get('edge_mode') or 'adaptive').strip().lower()
    try:
        edge_thicken = int(request.form.get('edge_thicken') or 3)
    except ValueError:
        edge_thicken = 3
    try:
        edge_opacity = float(request.form.get('edge_opacity') or 1.0)
    except ValueError:
        edge_opacity = 1.0
    try:
        kmeans_k = int(request.form.get('kmeans_k') or 8)
    except ValueError:
        kmeans_k = 8

    # ------------- Choose preset -------------
    if style == 'stylization':
        cfg = preset_stylization()
    elif style == 'pencil':
        cfg = preset_pencil()
    elif style == 'quantized':
        cfg = preset_clean_quantized()
        cfg.color.kmeans_k = max(4, min(12, kmeans_k))
    else:
        cfg = preset_classic()

    # ------------- Strengthen edges to avoid faint outlines -------------
    cfg.edges.mode = edge_mode  # 'adaptive' | 'canny' | 'dog'
    cfg.edges.thicken = max(0, min(5, edge_thicken))
    cfg.combine.edge_opacity = float(max(0.0, min(1.0, edge_opacity)))

    # Slightly crisper default for adaptive edges
    if cfg.edges.mode == 'adaptive' and style in ('classic', 'quantized'):
        cfg.edges.block_size = 7  # crisper
        cfg.edges.C = 5
        cfg.edges.blur_ksize = 5

    # ------------- Process -------------
    out_bgr = apply_cartoonify(img, cfg)

    # ------------- Save & return -------------
    out_name = f"cartoon_{os.path.splitext(unique_name)[0]}.jpg"
    out_path = os.path.join(app.config['CARTOON_FOLDER'], out_name)
    cv2.imwrite(out_path, out_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

    return send_file(out_path, mimetype='image/jpeg')

@app.route('/cartoonized_images/<filename>')
def send_cartoonized_image(filename):
    return send_from_directory(app.config['CARTOON_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
