from flask import Flask, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# Load model (pastikan file model Anda berada pada path yang benar)
model = joblib.load('model_rekomendasi.pkl')  # Ganti dengan path model Anda

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Misalnya input berupa nilai pelajaran
    nilai_pp = data['nilai_pp']
    nilai_bi = data['nilai_bi']
    nilai_mtk = data['nilai_mtk']
    nilai_inggris = data['nilai_inggris']

    # Menggabungkan data input menjadi format yang sesuai dengan model
    input_data = np.array([nilai_pp, nilai_bi, nilai_mtk, nilai_inggris]).reshape(1, -1)

    # Melakukan prediksi dengan model ML
    prediction = model.predict(input_data)

    # Mengembalikan hasil prediksi sebagai JSON
    return jsonify({'prediksi': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
