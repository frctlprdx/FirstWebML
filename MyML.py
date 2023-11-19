import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    return value / 100000000
df['Penghasilan (IDR)'] = df['Penghasilan (IDR)'].apply(simplifyIDR)

df.rename(columns={'Berat Barang (KG)': 'Berat Barang (Ribuan Ton)'}, inplace=True)
df.rename(columns={'Penghasilan (IDR)': 'Penghasilan (Ratusan Juta Rupiah)'}, inplace=True)

st.write("### Kita memakai data penjualan {}".format(option))
df
correlation = df['Berat Barang (Ribuan Ton)'].corr(df['Penghasilan (Ratusan Juta Rupiah)'])
st.write("Korelasi antar X dan Y adalah {}".format(correlation))
st.write("Hal ini membuktikan korelasi yang kuat antara X dan Y")

df['TANGGAL PENJUALAN'] = pd.to_datetime(df['TANGGAL PENJUALAN'], format='%m/%d/%Y')
df['TANGGAL PENJUALAN'] = df['TANGGAL PENJUALAN'].dt.strftime('%m/%Y')

st.write("### Kita kelompokkan data dalam 8 bulan kebelakang")
df = df.groupby(df['TANGGAL PENJUALAN']).agg({'Berat Barang (Ribuan Ton)': 'sum', 'Penghasilan (Ratusan Juta Rupiah)': 'sum'}).reset_index()
df

df['X^2'] = df['Berat Barang (Ribuan Ton)'] ** 2
df['Y^2'] = df['Penghasilan (Ratusan Juta Rupiah)'] ** 2
df['XY'] = df['Berat Barang (Ribuan Ton)']  * df['Penghasilan (Ratusan Juta Rupiah)']
sigmaX = df['Berat Barang (Ribuan Ton)'].sum()
sigmaY= df['Penghasilan (Ratusan Juta Rupiah)'].sum()
sigmaXSquare = df['X^2'].sum()
sigmaXY = df['XY'].sum()
n = len(df)
Xbar = df['Berat Barang (Ribuan Ton)'].mean()
Ybar = df['Penghasilan (Ratusan Juta Rupiah)'].mean()
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
    'Prediksi Pendapatan (Dalam Ratusan Juta Rupiah)': hasil_array,
})

st.write(predict)

# Menampilkan chart bar menggunakan Matplotlib
fig, ax = plt.subplots()
ax.bar(predict.index, predict['Prediksi Pendapatan (Dalam Ratusan Juta Rupiah)'], label='Prediksi Pendapatan (Dalam Ratusan Juta Rupiah)')
ax.set_ylabel('Prediksi Pendapatan')
ax.set_title('Prediksi Pendapatan Sesuai dengan Berat Penjualan')
ax.legend()

# Mengganti label sumbu X sesuai dengan input yang diberikan oleh pengguna
ax.set_xticks(predict.index)
ax.set_xticklabels(predict['Prediksi Bulan ke-'])
ax.set_xlabel("Bulan Ke-")

# Menampilkan plot di Streamlit
st.pyplot(fig)