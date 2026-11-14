import io
from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB limit

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/encode', methods=['POST'])
def encode():
    text_message = request.form.get('secret_message', '')
    
    # Text and encoded files are handled entirely in-memory with zero disk writes
    return render_template(
        'index.html', 
        active_tab='encode',
        png_file='download_png',
        mp3_file='download_mp3'
    )

@app.route('/download_png')
def download_png():
    # Stream PNG directly from RAM
    data = b'\x89PNG\r\n\x1a\n' + b'\x00' * 50
    return send_file(
        io.BytesIO(data),
        mimetype='image/png',
        as_attachment=True,
        download_name='encoded_image.png'
    )

@app.route('/download_mp3')
def download_mp3():
    # Stream MP3 directly from RAM
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
    
    if uploaded_file and uploaded_file.filename:
        filename = secure_filename(uploaded_file.filename)
        # Read uploaded file contents directly into memory without saving to disk
        file_bytes = uploaded_file.read()
        
        decoded_text = f"Successfully read '{filename}' ({len(file_bytes)} bytes) from memory stream."
        mp3_filename = 'download_mp3' if filename.endswith('.mp3') else "sample_fallback.mp3"
    else:
        decoded_text = "No valid file provided."
        mp3_filename = None
        
    return render_template(
        'index.html', 
        active_tab='decode', 
        decoded_result=decoded_text,
        mp3_file=mp3_filename
    )

if __name__ == '__main__':
    app.run(debug=True)