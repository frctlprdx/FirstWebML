import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.write("# Predict Sales with Linear Regression")

st.write ("### Data Penjualan")

dataset = pd.read_csv("./data/DATA PENJUALAN.csv")
dataset = pd.DataFrame(dataset)
dataset


st.write ('### Hilangkan data yang tidak kita perlukan')
df = dataset.drop(['NO', 'NO. NOTA PENJUALAN' , 'NAMA PEMBELI', 'Ur Barang', 'Jumlah Satuan','Kd Satuan', 'Bruto','Fob Valuta', 'KODE VALUTA', 'NDPBM'],axis=1, inplace=True)
df = dataset.dropna()
df

df['Netto'] = df['Netto'].str.replace(',', '').str.replace('.', '').astype(int)
df.rename(columns={'FOB DALAM RUPIAH': 'Penghasilan (IDR)'}, inplace=True)
df['Penghasilan (IDR)'] = df['Penghasilan (IDR)'].str.replace(',', '').str.replace('.', '').astype(int) 
df['Penghasilan (IDR)'] = df['Penghasilan (IDR)'] // 100
df.rename(columns={'Ur Barang': 'Nama Barang'}, inplace=True)
df.rename(columns={'Netto': 'Berat Barang (KG)'}, inplace=True)

perusahaan = df.drop_duplicates(subset='NAMA PENJUAL')
option = st.selectbox("Pilih Perusahaan", (perusahaan['NAMA PENJUAL']))
df = df.loc[df['NAMA PENJUAL'] == option]

def kiloToTon(value):
    return value / 100000000
df['Berat Barang (KG)'] = df['Berat Barang (KG)'].apply(kiloToTon)

df['Penghasilan (IDR)'] = df['Penghasilan (IDR)'].astype(float)
def simplifyIDR(value):
    return value / 1000000000
df['Penghasilan (IDR)'] = df['Penghasilan (IDR)'].apply(simplifyIDR)

df.rename(columns={'Berat Barang (KG)': 'Berat Barang (Ribuan Ton)'}, inplace=True)
df.rename(columns={'Penghasilan (IDR)': 'Penghasilan (Miliar Rupiah)'}, inplace=True)

st.write("### Kita memakai data penjualan {}".format(option))
df
correlation = df['Berat Barang (Ribuan Ton)'].corr(df['Penghasilan (Miliar Rupiah)'])
st.write("Korelasi antar X dan Y adalah {}".format(correlation))
st.write("Hal ini membuktikan korelasi yang kuat antara X dan Y")

df['TANGGAL PENJUALAN'] = pd.to_datetime(df['TANGGAL PENJUALAN'], format='%m/%d/%Y')
df['TANGGAL PENJUALAN'] = df['TANGGAL PENJUALAN'].dt.strftime('%m/%Y')

st.write("### Kita kelompokkan data dalam 7 bulan kebelakang")
df = df.groupby(df['TANGGAL PENJUALAN']).agg({'Berat Barang (Ribuan Ton)': 'sum', 'Penghasilan (Miliar Rupiah)': 'sum'}).reset_index()
df

df['X^2'] = df['Berat Barang (Ribuan Ton)'] ** 2
df['Y^2'] = df['Penghasilan (Miliar Rupiah)'] ** 2
df['XY'] = df['Berat Barang (Ribuan Ton)']  * df['Penghasilan (Miliar Rupiah)']
sigmaX = df['Berat Barang (Ribuan Ton)'].sum()
sigmaY= df['Penghasilan (Miliar Rupiah)'].sum()
sigmaXSquare = df['X^2'].sum()
sigmaXY = df['XY'].sum()
n = len(df)
Xbar = df['Berat Barang (Ribuan Ton)'].mean()
Ybar = df['Penghasilan (Miliar Rupiah)'].mean()
b = ((n*sigmaXY) - (sigmaX * sigmaY))/((n*sigmaXSquare) - (sigmaX**2))
a = Ybar - (b*Xbar)

st.write("Model Linear Regresi yang kita dapatkan adalah :")
st.write("### Y = {} + {}*X".format(a,b))
st.write("dengan X adalah nilai yang akan kita inputkan")

nilai_array = []
bulan_prediksi =[]

jumlah_data = st.number_input("Input beberapa bulan kedepan untuk diprediksi:", min_value=1, step=1)

for i in range(jumlah_data):
    prediksi = i+1
    nilai = st.number_input(f"Masukkan penjualan bulan ke-{i + 1} (Satuan Berat dalam Ribuan Ton):")
    bulan_prediksi.append(prediksi)
    nilai_array.append(nilai)

hasil_array = np.array(nilai_array)*b + a

predict = pd.DataFrame({
    'Prediksi Bulan ke-' : bulan_prediksi,
    'Prediksi Penjualan (x1.000 Ton)' : nilai_array,
    'Prediksi Pendapatan (Dalam Miliar Rupiah)': hasil_array,
})

st.write(predict)

simplify1 = pd.DataFrame({'X1': predict['Prediksi Penjualan (x1.000 Ton)'],
                         'Y1': predict['Prediksi Pendapatan (Dalam Miliar Rupiah)']})

simplify2 = pd.DataFrame({'X2': df['Berat Barang (Ribuan Ton)'],
                         'Y2': df['Penghasilan (Miliar Rupiah)']})


sorteddf1 = simplify1.sort_values(by='X1', ascending=True)
sorteddf2 = simplify2.sort_values(by='X2', ascending=True)

combined_df = pd.concat([predict, df['Penghasilan (Miliar Rupiah)']], axis=1)

# Membuat scatter plot dengan matplotlib di Streamlit
fig, ax = plt.subplots()

# Scatter plot untuk data pertama
ax.scatter(sorteddf1['X1'], sorteddf1['Y1'], label='Prediksi Pendapatan')

# Scatter plot untuk data kedua
ax.scatter(sorteddf2['X2'], sorteddf2['Y2'], label='Penghasilan')

# Garis untuk data pertama
ax.plot(sorteddf1['X1'], sorteddf1['Y1'], linestyle='-', color='blue')

# Garis untuk data kedua
ax.plot(sorteddf2['X2'], sorteddf2['Y2'], linestyle='-', color='orange')

ax.set_title('Scatter Plot Prediksi dan Penghasilan per Bulan')
ax.set_xlabel('Berat Penjualan')
ax.set_ylabel('Pendapatan/Penghasilan (Miliar Rupiah)')
ax.legend()  # Menampilkan legenda

# Menampilkan plot di Streamlit
st.pyplot(fig)

X = sorteddf1[['X1']]
y = sorteddf1['Y1']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)


st.write(f"Mean Squared Error (MSE): {mse}")
st.write(f"Mean Absolute Error (MAE): {mae}")
st.write(f"R-squared (R²): {r_squared}")


