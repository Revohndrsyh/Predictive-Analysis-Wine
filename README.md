# Laporan Proyek Machine Learning - Revo Hendriansyah

## Domain Proyek

Proyek ini berfokus pada bidang Pertanian, khususnya prediksi kualitas produk wine menggunakan teknik machine learning.
Penilaian kualitas wine secara tradisional masih mengandalkan panel ahli yang subjektif dan memakan waktu. Dengan memanfaatkan data parameter kimiawi wine, proyek ini bertujuan mengembangkan model prediksi kualitas wine yang cepat dan objektif, untuk membantu produsen meningkatkan mutu dan efisiensi proses kontrol kualitas.

## Business Understanding

### Problem Statements

- Bagaimana membangun model machine learning untuk memprediksi kualitas wine (baik vs buruk)?
- Algoritma mana yang paling efektif untuk prediksi kualitas?
- Bagaimana hasil prediksi dapat mendukung produsen dalam pengendalian kuaitas wine?
- Bahan apa yang berpengaruh besar pada kualitas wine?

### Goals

- Membuat model klasifikasi biner
- Membandingkan beberapa algoritma untuk mendapatkan model terbaik.
- Menentukan bahan terbaik untuk menghasilkan kualitas wine yang baik
- Menyusun dokumentasi dan skrip yang mudah digunakan.

### Solution Statements

- Pengujian Berbagai Algoritma Klasifikasi
  Menerapkan dan membandingkan beberapa model machine learning seperti K-Nearest Neighbors (KNN), Random Forest, Bernoulli Naive Bayes, Extra Trees Classifier, dan Support Vector Machine (SVM). Dengan membandingkan performa setiap algoritma menggunakan metrik akurasi, precision, recall, dan F1-score, dapat dipilih model terbaik yang sesuai dengan karakteristik dataset.
- Improvement Model dengan Hyperparameter Tuning
  Melakukan penyetelan parameter (hyperparameter tuning) pada model baseline yang sudah terbukti baik, seperti Random Forest dan SVM, untuk meningkatkan performa prediksi secara signifikan. Contohnya, mengatur jumlah estimator, kedalaman pohon, atau parameter kernel SVM menggunakan teknik Grid Search atau Random Search.
- Menerapkan Feature Importance untuk menentukan fitur atau kimiawi apa yang paling berpengaruh pada kualitas wine
- Validasi dan Evaluasi Model Secara Terukur
  Menggunakan data validasi dan metrik evaluasi yang relevan untuk mengukur performa model secara objektif, sehingga solusi yang diambil dapat dinilai keberhasilannya berdasarkan angka-angka evaluasi yang konkret dan dapat direproduksi.

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah Wine Quality Dataset yang bersumber dari platform publik Kaggle dan dapat diunduh di tautan berikut sumber datanya : [Wine Quality Dataset](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset)

## Informasi Dataset

### Jumlah Baris dan Kolom

Wine Quality Dataset memiliki jumlah baris dan kolom :

- Jumlah Baris : 1143
- Jumlah Kolom : 13

### Kondisi Data

Berdasarkan pemeriksaan awal terhadap kondisi data, beberapa yang perlu diperhatikan:

- Missing Value: Tidak terdapat missing value
- Duplikat Data: Tidak Terdapat data duplikat

### Fitur Pada Data

1.  fixed acidity: Tingkat keasaman tetap dalam wine yang berkontribusi pada rasa asam.
2.  volatile acidity: Keasaman yang mudah menguap, jika tinggi dapat menyebabkan rasa tidak sedap seperti cuka.
3.  citric acid: Kandungan asam sitrat yang memberikan rasa segar pada wine.
4.  residual sugar: Gula tersisa setelah proses fermentasi, memengaruhi rasa manis wine.
5.  chlorides: Kandungan garam klorida yang berpengaruh pada rasa asin.
6.  free sulfur dioxide: Konsentrasi SO₂ bebas yang mencegah oksidasi dan pertumbuhan mikroorganisme.
7.  otal sulfur dioxide: Jumlah total SO₂, termasuk yang bebas dan terikat, memengaruhi rasa dan daya simpan.
8.  density: Kepadatan wine, terkait dengan kadar alkohol dan gula.
9.  pH: Tingkat keasaman, yang memengaruhi rasa dan kestabilan wine.
10. sulphates: Kandungan kalium sulfat yang juga berperan sebagai pengawet.
11. alcohol: Persentase kadar alkohol dalam volume wine.
12. quality: Skor penilaian kualitas wine pada skala 0 sampai 10, target prediksi.
13. Id: Nomor identifikasi unik untuk setiap sampel data.

## Data Preparation

1. Penghapusan Kolom Tidak Relevan Kolom Id dihapus karena merupakan identifier unik dan tidak memberikan informasi prediktif terhadap kualitas wine.
2. Pengecekan data duplikat dan missing value.
3. Variabel target quality yang awalnya memiliki nilai kontinu pada skala 0–10 diubah menjadi kelas biner:
   - 1 untuk wine berkualitas baik dengan skor ≥ 7
   - 0 untuk wine dengan skor < 7 Hal ini mempermudah proses klasifikasi dan evaluasi model dalam konteks proyek.
4. Pembagian Data (Train-Test Split) Dataset dibagi menjadi data pelatihan dan pengujian dengan proporsi 80:20 menggunakan fungsi train_test_split dengan random_state=42 dan stratifikasi berdasarkan label agar distribusi kelas tetap seimbang.
5. dilakukan analisis feature importance pada model berbasis pohon seperti Random Forest dan Gradient Boosting untuk mengidentifikasi fitur-fitur yang paling berpengaruh dalam memprediksi kualitas wine. Analisis ini membantu memahami variabel mana yang memiliki kontribusi signifikan terhadap hasil prediksi, sehingga dapat memberikan insight penting bagi pengembangan produk dan proses kontrol kualitas. Misalnya, fitur seperti alcohol, volatile acidity, dan sulphates biasanya muncul sebagai variabel utama yang menentukan kualitas wine. Pemanfaatan feature importance tidak hanya memperkuat interpretabilitas model, tetapi juga membantu dalam proses feature selection untuk mengurangi kompleksitas model tanpa mengorbankan performa.

## Modeling

Pada tahap pemodelan, empat algoritma klasifikasi diuji untuk memprediksi kualitas wine berdasarkan fitur kimiawi. Berikut penjelasan singkat tiap algoritma dan parameter utamanya:

1. Logistic RegressionModel: klasifikasi linear yang mengestimasi probabilitas kelas target menggunakan fungsi logistik. Parameter max_iter=1000 memastikan iterasi optimasi cukup untuk konvergensi, dan random_state=42 untuk reproduksibilitas hasil.

2. Random Forest Classifier: Algoritma ensemble yang membangun 100 pohon keputusan secara acak (n_estimators=100) dan menggabungkan hasil voting mayoritas untuk prediksi akhir. Random state 42 dipakai untuk konsistensi hasil. Model ini efektif mengurangi overfitting dan mampu menangani data non-linear.

3. Support Vector Machine (SVM): Model yang mencari hyperplane optimal untuk memisahkan kelas dengan margin terbesar. Default kernel linear digunakan, dengan random_state=42 untuk konsistensi. SVM sangat baik untuk data berdimensi tinggi dan dapat bekerja dengan data tidak linear melalui kernel (jika diatur).

4. Gradient Boosting Classifier: Metode ensemble boosting yang menggabungkan banyak pohon keputusan lemah menjadi model kuat dengan 100 estimator (n_estimators=100). Random state 42 digunakan untuk reproduksibilitas. Algoritma ini dikenal mampu memberikan performa akurasi tinggi pada banyak kasus klasifikasi.

## Evaluation
1. Model Machine learning untuk Prediksi Kualitas Wine 
Berhasil dibangun model klasifikasi biner yang memprediksi wine berkualitas baik (label 1) dan buruk (label 0) menggunakan parameter kimiawi sebagai fitur input. Model ini mengubah data kualitas numerik menjadi kelas biner sesuai kebutuhan klasifikasi, sehingga menyederhanakan proses prediksi dan evaluasi

![alt text](https://github.com/Revohndrsyh/Predictive-Analysis-Wine/blob/main/Perbandingan%20Model.png?raw=true)

3. Perbandingan Algoritma
Empat algoritma klasifikasi diuji, yaitu Logistic Regression, Random Forest, Support Vector Machine (SVM), dan Gradient Boosting.
    - Random Forest tampil sebagai model terbaik dengan akurasi tertinggi (92.1%) serta metrik precision, recall, dan F1-score kelas positif yang paling seimbang. Hal ini menunjukkan kemampuannya mengenali wine berkualitas baik dengan cukup baik.
    - Gradient Boosting dan SVM juga menunjukkan performa yang kompetitif, walaupun recall kelas positif masih di bawah ideal.
    - Logistic Regression memiliki akurasi cukup tinggi, namun kurang efektif dalam mengenali wine baik (recall rendah), sehingga kurang optimal untuk tujuan utama.

4. Dukungan Prediksi untuk Pengendalian Kualitas
Model yang dikembangkan memberikan prediksi cepat dan objektif sehingga dapat membantu produsen dalam pengendalian mutu wine secara lebih efisien dan konsisten, menggantikan metode tradisional yang subjektif dan memakan waktu.

![alt text](https://github.com/Revohndrsyh/Predictive-Analysis-Wine/blob/main/Feature%20Importance.png?raw=true)

5. Feature Importance dan Interpretabilitas
Analisis feature importance pada model Random Forest dan Gradient Boosting berhasil mengidentifikasi fitur-fitur utama yang paling berpengaruh terhadap kualitas wine, yakni alcohol, volatile acidity, dan sulphates. Informasi ini sangat berharga bagi produsen untuk fokus mengontrol parameter kritis dalam proses produksi.

6. Pencapaian Goals
    - Model klasifikasi biner telah dibuat dan diuji menggunakan berbagai algoritma.
    - Perbandingan performa algoritma telah dilakukan secara menyeluruh dengan metrik evaluasi yang lengkap.
    - Feature importance diterapkan untuk mengidentifikasi fitur kunci yang mempengaruhi kualitas wine.
    - Dokumentasi dan skrip disusun secara terstruktur untuk memudahkan pemakaian dan pengembangan selanjutnya.


## Refrensi

1. [Wine Quality Dataset](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset)
2. [RandomForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
