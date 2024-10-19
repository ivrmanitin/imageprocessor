from flask import Flask, request, jsonify, send_file, render_template
import torch
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Serve the HTML page
@app.route('/')
def index():
    return render_template('index.html')  # Make sure this HTML file is in a templates folder

# Path to store uploaded and generated images
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the MiDaS model
model_type = "DPT_Large"  # or "MiDaS_small" for a smaller model
model = torch.hub.load("intel-isl/MiDaS", model_type)
model.eval()

# Load transformation function
if "DPT" in model_type:
    transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
else:
    transform = torch.hub.load("intel-isl/MiDaS", "transforms").midas_transform

# Function to predict depth from an image
def predict_depth(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (640, 480))
    input_image = transform(img)
    
    with torch.no_grad():
        prediction = model(input_image)

    depth_map = prediction.squeeze().cpu().numpy()
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, norm_type=cv2.NORM_MINMAX)
    depth_map_normalized = depth_map_normalized.astype(np.uint8)
    
    # Save depth map
    depth_map_path = os.path.join(UPLOAD_FOLDER, 'depth_map.png')
    cv2.imwrite(depth_map_path, depth_map_normalized)
    
    return depth_map_path

# Function to generate a pencil sketch
def generate_pencil_sketch(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inverted_gray = 255 - gray
    blurred = cv2.GaussianBlur(inverted_gray, (21, 21), 0)
    inverted_blurred = 255 - blurred
    sketch = cv2.divide(gray, inverted_blurred, scale=256.0)
    
    # Save pencil sketch
    sketch_path = os.path.join(UPLOAD_FOLDER, 'pencil_sketch.png')
    cv2.imwrite(sketch_path, sketch)
    
    return sketch_path

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(image_path)

    # Get selected option
    selected_option = request.form.get('option')

    # Generate depth map or pencil sketch based on selected option
    if selected_option == "depth_map":
        depth_map_path = predict_depth(image_path)
        download_filename = 'depth_map.png'
    elif selected_option == "pencil_sketch":
        depth_map_path = generate_pencil_sketch(image_path)
        download_filename = 'pencil_sketch.png'
    else:
        return jsonify({"error": "Invalid option selected"}), 400

    return jsonify({
        "original_image_url": f"/uploads/{filename}",
        "depth_map_url": f"/uploads/{download_filename}",
        "download_url": f"/download/{download_filename}"  # Add download URL to the response
    })

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join(UPLOAD_FOLDER, filename))

# Route for downloading the processed image
@app.route('/download/<filename>', methods=['GET'])
def download_image(filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return jsonify({"error": "File not found"}), 404

if __name__ == '__main__':
    app.run(debug=True)
