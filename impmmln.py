import pandas as pd
import numpy as np
import seaborn as sns   
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
from wordcloud import WordCloud

# Membaca dataset
df_mobil = pd.read_csv('CarPrice.csv')

# Mengecek data kosong
missing_data = df_mobil.isnull().sum()
print("Jumlah data kosong pada setiap kolom:")
print(missing_data)

# Menampilkan statistik deskriptif
df_stats = df_mobil.describe().T
plt.figure(figsize=(12, 8))
sns.heatmap(df_stats, annot=True, fmt=".2f", cmap='coolwarm', linewidths=.5)
plt.title('Statistik Deskriptif Dataset Mobil')
plt.show()

# Filter hanya kolom numerik dan menambahkan kolom median
df_numerik = df_mobil.select_dtypes(include=np.number)
statistik = df_numerik.describe(percentiles=[0.25, 0.5, 0.75]).T
statistik['median'] = df_numerik.median()
statistik = statistik[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
print(statistik)

# Menampilkan distribusi harga mobil
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.title('Car Price Distribustion Plot')
sns.histplot(df_mobil.price)
plt.show()

# Visualisasi distribusi nama mobil
car_counts = df_mobil['CarName'].value_counts()
plt.figure(figsize=(10, 6))
car_counts.plot(kind="bar")
plt.title("CarName Distribution")
plt.xlabel("CarName")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

# Membuat word cloud untuk nama mobil
car_names = " ".join(df_mobil['CarName'].astype(str))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(car_names)
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Word Cloud Nama Mobil", fontsize=16)
plt.show()

# Scatter plot antara highwaympg dan price
plt.scatter(df_mobil['highwaympg'], df_mobil['price'])
plt.xlabel('highwaympg')
plt.ylabel('price')
plt.title('Scatter Plot highwaympg vs price')
plt.show()

# Memilih fitur dan target untuk model regresi
X = df_mobil[['highwaympg', 'curbweight', 'horsepower']]
y = df_mobil['price']

# Memisahkan dataset menjadi training dan testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat dan melatih model regresi linear
model_regresi = LinearRegression()
model_regresi.fit(X_train, y_train)

# Melakukan prediksi
model_regresi_pred = model_regresi.predict(X_test)

# Visualisasi hasil prediksi dan nilai aktual
plt.scatter(X_test.iloc[:, 0], y_test, label='Actual Prices', color='blue')
plt.scatter(X_test.iloc[:, 0], model_regresi_pred, label='Predicted Prices', color='red')
plt.xlabel('highwaympg')
plt.ylabel('Price')
plt.legend()
plt.title('Comparison of Actual and Predicted Prices')
plt.show()

# Menampilkan prediksi harga untuk data input tertentu
input_data = pd.DataFrame([[32, 2338, 75]], columns=['highwaympg', 'curbweight', 'horsepower'], dtype=int)
predicted_price = model_regresi.predict(input_data)
print(f"Prediksi harga mobil dengan highwaympg = 32, curbweight = 2338, dan horsepower = 75 adalah: ${predicted_price[0]:,.2f}")

# Evaluasi model menggunakan MAE, MSE, dan RMSE
mae = mean_absolute_error(y_test, model_regresi_pred)
print(f'Mean Absolute Error (MAE): {mae:.2f}')
mse = mean_squared_error(y_test, model_regresi_pred)
print(f'Mean Squared Error (MSE): {mse:.2f}')
rmse = np.sqrt(mse)
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')

# Menyimpan model ke dalam file .sav menggunakan pickle
filename = 'model_prediksi_harga_mobil.sav'
pickle.dump(model_regresi, open(filename, 'wb'))

print(f"Model telah disimpan dalam file {filename}")