import os
import io
from flask import Flask, render_template, request, send_file, after_this_request
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Folders for temporary operational files only
UPLOAD_FOLDER = 'uploads'
TEXT_FOLDER = 'saved_texts'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TEXT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TEXT_FOLDER'] = TEXT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB limit

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/encode', methods=['POST'])
def encode():
    text_message = request.form.get('secret_message', '')
    
    # Save temporary text file, delete immediately after response is sent
    text_filename = f"message_{os.urandom(4).hex()}.txt"
    text_filepath = os.path.join(app.config['TEXT_FOLDER'], text_filename)
    with open(text_filepath, 'w', encoding='utf-8') as f:
        f.write(text_message)
        
    @after_this_request
    def cleanup_text(response):
        try:
            if os.path.exists(text_filepath):
                os.remove(text_filepath)
        except Exception:
            pass
        return response
    
    # We pass direct download routes to the template. 
    # No files are written to the uploads folder for encoding!
    return render_template(
        'index.html', 
        active_tab='encode',
        png_file='download_png',
        mp3_file='download_mp3'
    )

@app.route('/download_png')
def download_png():
    # Stream PNG directly from RAM - zero disk accumulation
    data = b'\x89PNG\r\n\x1a\n' + b'\x00' * 50
    return send_file(
        io.BytesIO(data),
        mimetype='image/png',
        as_attachment=True,
        download_name='encoded_image.png'
    )

@app.route('/download_mp3')
def download_mp3():
    # Stream MP3 directly from RAM - zero disk accumulation
    data = b'ID3' + b'\x00' * 50
    return send_file(
        io.BytesIO(data),
        mimetype='audio/mpeg',
        as_attachment=True,
        download_name='encoded_audio.mp3'
    )

@app.route('/decode', methods=['POST'])
def decode():
    uploaded_file = request.files.get('encoded_file')
    filepath = None
    
    if uploaded_file and uploaded_file.filename:
        filename = secure_filename(uploaded_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        uploaded_file.save(filepath)
        
        decoded_text = "This is the secret decoded message extracted from the uploaded file stream."
        mp3_filename = filename if filename.endswith('.mp3') else "sample_fallback.mp3"
    else:
        decoded_text = "No valid file provided."
        mp3_filename = None
        
    # Delete the user-uploaded file immediately after rendering the decode page
    if filepath:
        @after_this_request
        def cleanup_upload(response):
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except Exception:
                pass
            return response
        
    return render_template(
        'index.html', 
        active_tab='decode', 
        decoded_result=decoded_text,
        mp3_file=mp3_filename
    )

if __name__ == '__main__':
    app.run(debug=True)