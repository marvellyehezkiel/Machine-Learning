import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from collections import Counter
import joblib

# Load data
data = pd.read_csv('data_jurusan.csv')

# Pisahkan fitur dan target
X = data[['Bobot Bahasa Indonesia', 'Bobot Matematika', 'Bobot Bahasa Inggris', 
          'Bobot Pendidikan Jasmani Olahraga dan Kesehatan', 'Bobot Sejarah', 
          'Bobot Seni dan Budaya']]  # Kolom fitur yang sesuai dengan data Anda
y = data['Jurusan']  # Kolom target 'Jurusan'

# Lakukan standar skala pada data fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Memeriksa distribusi kelas
print("Distribusi Kelas:\n", Counter(y))

# Hanya gunakan data yang memiliki kelas lebih dari satu sampel
class_counts = y.value_counts()
valid_classes = class_counts[class_counts > 1].index

# Filter kelas yang memiliki lebih dari satu sampel
X_filtered = X_scaled[y.isin(valid_classes)]
y_filtered = y[y.isin(valid_classes)]

# Bagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered, test_size=0.3, random_state=42)

# Latih model dengan Random Forest, gunakan class_weight='balanced'
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Evaluasi model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi model: {accuracy * 100}%")

# Tampilkan laporan klasifikasi
print("Laporan Klasifikasi:\n", classification_report(y_test, y_pred, zero_division=1))

# Simpan model
joblib.dump(model, 'model_jurusan.pkl')
