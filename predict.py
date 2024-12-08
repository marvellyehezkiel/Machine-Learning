import sys
import joblib
import pandas as pd

# Ambil data input dari argumen
matematika = float(sys.argv[1])
bahasa_indonesia = float(sys.argv[2])
bahasa_inggris = float(sys.argv[3])
fisika = float(sys.argv[4])

# Load model
model = joblib.load('model_jurusan.pkl')

# Buat DataFrame untuk input
input_data = pd.DataFrame([[matematika, bahasa_indonesia, bahasa_inggris, fisika]], columns=['matematika', 'bahasa_indonesia', 'bahasa_inggris', 'fisika'])

# Prediksi
prediction = model.predict(input_data)

# Tampilkan hasil
print(prediction[0])
