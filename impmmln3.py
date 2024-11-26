import pickle
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Memuat model yang sudah disimpan
model = pickle.load(open('model_prediksi_harga_mobil.sav', 'rb'))

# Mengatur judul aplikasi
st.title('Prediksi Harga Mobil')

# Sidebar untuk navigasi
menu = st.sidebar.selectbox("Pilih Halaman", ["Home", "Dataset", "Visualisasi", "Evaluasi Model", "Prediksi"])

# Halaman Home
if menu == "Home":
    st.header("Selamat datang di Aplikasi Prediksi Harga Mobil!")
    st.write("""
    Aplikasi ini digunakan untuk memprediksi harga mobil berdasarkan fitur-fitur seperti:
    - *Highway MPG*
    - *Curbweight*
    - *Horsepower*

    Silakan jelajahi aplikasi ini untuk mendapatkan informasi lebih lanjut atau melakukan prediksi harga mobil.
    """)
    # Menampilkan gambar
    st.image("D:/hariini/Screenshot 2024-11-26 094810.png", caption="Aplikasi Prediksi Harga Mobil", use_column_width=True)

# Halaman Dataset
elif menu == "Dataset":
    st.header("Dataset Mobil")
    # Membaca dataset dan menampilkannya
    df1 = pd.read_csv('CarPrice.csv')
    st.dataframe(df1)

    # Menampilkan grafik distribusi fitur numerik
    st.write("Distribusi Fitur Mobil")
    numerical_features = df1.select_dtypes(include=[np.number]).columns

    for feature in numerical_features:
        st.write(f"Grafik {feature}")
        st.line_chart(df1[feature])

    # Menampilkan Heatmap untuk korelasi antar fitur (hanya untuk kolom numerik)
    st.write("Heatmap Korelasi Antar Fitur")
    df1_numerik = df1.select_dtypes(include=[np.number])  # Memilih hanya kolom numerik
    corr_matrix = df1_numerik.corr()  # Menghitung korelasi hanya untuk kolom numerik
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=.5)
    st.pyplot()

# Halaman Visualisasi
elif menu == "Visualisasi":
    st.header("Visualisasi Fitur")
    # Dataset untuk visualisasi
    df1 = pd.read_csv('CarPrice.csv')

    # Menampilkan Heatmap untuk korelasi antar fitur (hanya untuk kolom numerik)
    st.write("Heatmap Korelasi Antar Fitur")
    df1_numerik = df1.select_dtypes(include=[np.number])  # Memilih hanya kolom numerik
    corr_matrix = df1_numerik.corr()  # Menghitung korelasi hanya untuk kolom numerik
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=.5)
    st.pyplot()

# Halaman Evaluasi Model
elif menu == "Evaluasi Model":
    st.header("Evaluasi Model")
    # Membaca dataset
    df1 = pd.read_csv('CarPrice.csv')
    
    # Fitur dan target
    X = df1[['highwaympg', 'curbweight', 'horsepower']]
    y = df1['price']
    
    # Membagi data menjadi training dan testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Prediksi dengan model
    y_pred = model.predict(X_test)

    # Metrik Evaluasi
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    st.write(f"R² (Koefisien Determinasi): {r2:.2f}")
    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Halaman Prediksi
elif menu == "Prediksi":
    st.header("Masukkan Data Mobil untuk Prediksi")
    
    # Input nilai untuk fitur-fitur mobil
    highwaympg = st.number_input('Highway MPG', min_value=0, max_value=100, value=30)
    curbweight = st.number_input('Curbweight (kg)', min_value=0, max_value=5000, value=2000)
    horsepower = st.number_input('Horsepower', min_value=0, max_value=500, value=100)

    # Membuat DataFrame dari inputan pengguna
    input_data = pd.DataFrame([[highwaympg, curbweight, horsepower]], columns=['highwaympg', 'curbweight', 'horsepower'])

    # Standarisasi input data
    scaler = StandardScaler()
    input_data_scaled = scaler.fit_transform(input_data)

    # Menampilkan tombol untuk melakukan prediksi
    if st.button('Prediksi'):
        # Melakukan prediksi menggunakan model yang dimuat
        car_prediction = model.predict(input_data_scaled)

        # Mengubah hasil prediksi menjadi angka desimal
        harga_mobil_str = np.array(car_prediction)
        harga_mobil_float = float(harga_mobil_str[0])

        # Menampilkan hasil prediksi
        harga_mobil_formatted = "${:,.2f}".format(harga_mobil_float)
        st.write(f"Harga mobil yang diprediksi adalah: {harga_mobil_formatted}")

# Menambahkan sidebar informasi tentang aplikasi
st.sidebar.header("Tentang Aplikasi")
st.sidebar.write("""
Aplikasi ini digunakan untuk memprediksi harga mobil berdasarkan fitur-fitur tertentu seperti highway mpg, curbweight, dan horsepower.
Model ini telah dilatih menggunakan dataset mobil dan kemudian disimpan untuk digunakan dalam aplikasi web ini.
Selain itu, aplikasi ini juga menampilkan visualisasi untuk analisis data dan evaluasi model.
""")