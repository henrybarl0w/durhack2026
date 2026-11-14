import io
from flask import Flask, Response, jsonify, redirect, render_template, request, send_file
from werkzeug.utils import secure_filename

from chorales import ChoraleError, encodeText, maxBytes, pngToDataUri
from scanner import Scanner

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB limit

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/encode', methods=['GET', 'POST'])
def encode():
    #a refresh, back button or typed address asks with GET: show the page instead of an error
    if request.method == 'GET':
        return redirect('/')
    text = request.form.get('secret_message', '')
    try:
        png, chorale = encodeText(text)
    except ChoraleError as error:
        return render_template('index.html', active_tab='encode', encode_error=str(error), message=text, max_bytes=maxBytes())
    return render_template(
        'index.html', active_tab='encode', message=text, chorale=chorale,
        png_data=pngToDataUri(png), max_bytes=maxBytes()
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

@app.route('/decode', methods=['GET', 'POST'])
def decode():
    if request.method == 'GET':
        return redirect('/')
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

#one scanner per browser session would be nicer; one shared scanner is enough for a demo
scanner = Scanner()

@app.route('/scan/frame', methods=['POST'])
def scanFrame():
    return jsonify(scanner.addFrame(request.get_data()))

@app.route('/scan/reset', methods=['POST'])
def scanReset():
    scanner.reset()
    return jsonify({"status": "reset"})

@app.route('/scan/midi')
def scanMidi():
    data = scanner.midi()
    if data is None:
        return "nothing scanned yet", 404
    return Response(data, mimetype='audio/midi', headers={'Content-Disposition': 'attachment; filename=chorale.mid'})

if __name__ == '__main__':
    import sys
    #by default only this computer can reach the server, with the debugger on. --lan (or --https) opens it to the
    #network for a phone and turns the debugger off, because the debugger lets anyone who reaches it run code.
    #phones only allow the camera on https (or localhost), so --https adds a self-signed certificate
    lan = '--lan' in sys.argv or '--https' in sys.argv
    ssl = 'adhoc' if '--https' in sys.argv else None
    #requests run in threads, because a browser keeps idle connections open and one at a time would hang on them;
    #drawing a chorale takes a lock instead (see chorales.py), as verovio is not thread safe
    #flask finishes the https handshake on its main thread, so one idle connection (browsers open several speculative
    #ones) would freeze the whole server: a short deadline on every connection drops them
    import socket
    socket.setdefaulttimeout(5)
    app.run(debug=not lan, host='0.0.0.0' if lan else '127.0.0.1', ssl_context=ssl, threaded=True)
