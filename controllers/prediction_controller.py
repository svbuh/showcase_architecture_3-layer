import os
from flask import Flask, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
from services.prediction_service import PredictionService

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__, template_folder='../templates')
prediction_service = PredictionService()


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        url = request.form.get('url')
        if url:
            try:
                predicted_class = prediction_service.predict_from_url(url)
                return render_template('index.html', url=url, prediction=predicted_class)
            except Exception as e:
                return f"An error occurred: {e}"

        if 'file' not in request.files:
            return redirect(request.url)
    return render_template('index.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


if __name__ == '__main__':
    app.run(debug=True)
