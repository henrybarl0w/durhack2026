import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration for file uploads
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit uploads to 16MB

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/encode', methods=['POST'])
def encode():
    text_message = request.form.get('secret_message', '')
    uploaded_file = request.files.get('cover_image')
    
    if uploaded_file and uploaded_file.filename:
        filename = secure_filename(uploaded_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        uploaded_file.save(filepath)  # Saved to server's hard drive
    
    # Placeholder response for encoding (simulate output image)
    return render_template('index.html', active_tab='encode', 
                           encoded_result=f"Success! Hidden message: '{text_message}'. Image saved and processed.")

@app.route('/decode', methods=['POST'])
def decode():
    uploaded_file = request.files.get('encoded_image')
    
    if uploaded_file and uploaded_file.filename:
        filename = secure_filename(uploaded_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        uploaded_file.save(filepath)  # Stored on the server's hard drive for decoding
        
        # Placeholder decoded text output
        decoded_text = "Hello! This is placeholder decoded text extracted from your uploaded image."
    else:
        decoded_text = "No valid image uploaded."
        
    return render_template('index.html', active_tab='decode', decoded_result=decoded_text)

if __name__ == '__main__':
    app.run(debug=True)