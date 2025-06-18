# Laporan Proyek Machine Learning - Geralda Livia

---

## Project Overview

Industri skincare saat ini tengah mengalami lonjakan pertumbuhan yang signifikan. Perkembangan ini tercermin dari munculnya beragam produk dengan berbagai formulasi 
dan manfaat yang ditawarkan kepada konsumen. Dalam situasi seperti ini, sistem rekomendasi skincare menjadi sangat penting untuk membantu individu menemukan produk 
yang paling sesuai dengan kondisi kulit mereka. Sejumlah studi menunjukkan bahwa penggunaan produk yang tidak cocok dapat memicu berbagai permasalahan kulit, seperti 
iritasi, jerawat, hingga hasil perawatan yang tidak optimal (Draelos, Z. D. 2018).

Melimpahnya pilihan produk di pasaran sering kali membuat konsumen merasa bingung dan kewalahan saat mencoba menentukan produk yang benar-benar tepat bagi mereka. 
Tantangan ini semakin kompleks dengan kehadiran banyak reviewer palsu dan buzzer yang menyebarkan ulasan tidak jujur demi kepentingan promosi. Ulasan semacam ini 
tersebar luas di berbagai media, mulai dari platform e-commerce hingga media sosial, yang akhirnya menumbuhkan rasa tidak percaya di kalangan konsumen. Banyak orang 
menjadi ragu terhadap keaslian rekomendasi yang mereka temukan, dan kesulitan membedakan mana informasi yang obyektif dan mana yang merupakan bagian dari strategi 
pemasaran. Oleh karena itu, konsumen sangat membutuhkan panduan yang lebih kredibel dan terarah.

Menanggapi permasalahan tersebut, proyek ini menghadirkan sebuah sistem rekomendasi produk skincare yang berbasis pada komposisi bahan (ingredients). Sistem ini 
dirancang untuk membantu pengguna menemukan produk yang memiliki kemiripan dengan produk yang sudah mereka kenal atau sukai. Fokus utama sistem ini adalah pada 
kesamaan bahan aktif, karena kandungan inilah yang sangat menentukan efektivitas suatu produk. Penelitian oleh Draelos et al. (2021) menyebutkan bahwa pemahaman terhadap 
kandungan bahan aktif dapat meningkatkan kepuasan pengguna dan mengurangi kemungkinan munculnya efek samping. Dengan bantuan teknologi yang mampu menganalisis dan 
merekomendasikan produk berdasarkan komposisi bahannya, diharapkan pengguna dapat membuat pilihan yang lebih tepat dan berdasarkan informasi yang terpercaya saat memilih 
skincare yang sesuai dengan kebutuhan mereka.

**Referensi:**
1. Draelos, Z. D. (Ed.). (2021). Cosmetic dermatology: products and procedures. John Wiley & Sons.

---

## Business Understanding

### Problem Statements

1. Bagaimana cara membantu konsumen menemukan produk skincare yang sesuai dengan kebutuhan kulit mereka di tengah banyaknya pilihan yang tersedia di pasaran?
2. Bagaimana mengurangi ketergantungan konsumen terhadap ulasan online yang tidak kredibel akibat maraknya reviewer palsu dan buzzer?

### Goals

1. Mengembangkan sistem rekomendasi produk skincare berbasis ingredients untuk membantu pengguna menemukan produk yang mirip dengan yang telah mereka gunakan atau sukai.
2. Menyediakan rekomendasi (Top-N produk skincare) yang objektif dan terpercaya dengan menganalisis kemiripan komposisi bahan aktif, bukan berdasarkan ulasan atau popularitas.

### Solution Approach

Proyek ini mengimplementasikan dua pendekatan sistem rekomendasi:

1. **Content-Based Filtering (CBF)**  
   - Menggunakan TF-IDF dan cosine similarity berdasarkan kolom `ingredients`.
   - Membantu pengguna menemukan produk dengan komposisi bahan aktif yang serupa.
   - Ditujukan untuk rekomendasi yang tidak tergantung pada rating atau review.

2. **Collaborative Filtering (CF)**  
   - Menggunakan matrix factorization dari library `implicit`.
   - Berdasarkan interaksi historis pengguna dengan produk (user-product interaction matrix).
   - Ditujukan untuk rekomendasi yang bersifat personal dengan mempelajari pola preferensi pengguna.
     
---

## Data Understanding

Dataset yang digunakan adalah `skincare_products_clean.csv`. Kumpulan data tersebut terdiri dari nama produk, url produk, jenis produk (seperti pelembap, serum, dsb.), bahan-bahan (ingridients), dan harga. Dataset tersebut terdiri dari 1138 baris dan 5 kolom. Dataset ini sudah dibersihkan oleh pemilik dataset, dimana pada bagian bahan-bahan (ingridients) dibersihkan sehingga variasi nama bahan telah dihilangkan dan diberikan satu nama untuk setiap bahan. 

Fitur:
- `product_name` : menunjukkan nama produk
- `product_url` : menunjukkan tautan belanja produk secara online
- `product_type` : menunjukkan kategori produk (misal Moisturiser, Bath Oil, dll)
- `clean_ingreds` : menunjukan bahan-bahan yang terkandung dalam produk
- `price` : menunjukkan harga produk 

EDA (Exploratory Data Analysis) telah dilakukan untuk melihat beberapa informasi berikut:
- Pengecekan nilai yang hilang.
- Pengecekan nilai duplikat.
- Distribusi tipe produk
- Distribusi top 10 brand dengan produk terbanyak
- Menampilkan 14 produk termurah dan termahal
- Menampilkan distribusi proporsi tipe produk menggunakan pie chart

**Insight:**
- Dataset ini terdiri atas 1138 baris dan 5 kolom.
- Tidak adanya nilai yang hilang (no missing value) dan tidak ada data yang terduplikasi (no duplicated value).
- Berdasarkan distribusi tipe data, produk skincare paling banyak adalah Mask, Body Wash, dan Moisturiser. Sedangkan. produk paling jarang adalah Bath Oil, Bath Salts, dan Peel.
- Berdasarkan top10 brand dengan produk terbanyak diduduki oleh brand Clinique, disusul oleh La Roche-Posay, The Ordinary, dan Garnier.
- Pada dataset, produk termurah di setiap kategori didominasi oleh brand-brand dengan harga sangat terjangkau seperti Holika Holika (Mask: £1.95) dan Westlab (Body Wash: £1.99). Sedangkan, produk termahal berasal dari brand premium seperti Elemis, Chantecaille, dan Medik8, dengan harga hingga £99.00 (Moisturiser) dan £96.00 (Serum).
- Proporsi distribusi tipe produk berdasarkan kategori Mask, Body Wash, dan Cleanser mendominasi total produk dengan persentase terbesar (>10%). Sedangkan, kategori seperti Peel, Bath Oil, dan Bath Salts memiliki jumlah produk paling sedikit (<3.5%).

Sumber dataset : [Skincare Products Dataset](https://www.kaggle.com/datasets/eward96/skincare-products-clean-dataset)

---

## Data Preparation

Pada data preparation dilakukan 2 pendekatan yang digunakan. Pendekatan tersebut untuk persiapan data dalam membangun model content based dan collaborative filter.

**Langkah-langkah untuk Content Based (CB):**

1. Menyalin dataframe utama ke variabel `df_cb`.
2. Membersihkan teks pada kolom `clean_ingreds`. Dengan mengubah semua huruf menjadi huruf kecil (lowercase) dan menghapus karakter non-huruf (simbol, angka).
3. Mengonversi kolom `price` dari string ke numerik. Menghapus simbol £ dan mengubah ke floa.
4. Menghapus missing value di kolom `clean_ingreds`

📌 **Alasan langkah-langkah Content Based tersebut dilakukan:**

- Salinan  `df_cb` agar proses pembersihan data tidak memengaruhi dataset utama.
- Pembersihan pada kolom `clean_ingreds` agar konsisten saat vektorisasi dan menghindari noise saat perhitungan TF-IDF.
- Mengkonversi tipe data `price` dari string ke numerik agar harga bisa digunakan dalam analisis agar memastikan tipe data numerik yang valid.
- Hapus missing value pada kolom `clean_ingreds` untuk menghindari error saat membuat TF-IDF vectorizer menjaga kualitas data agar hanya produk dengan deskripsi bahan yang tersedia yang masuk ke model.

**Langkah-langkah untuk Collaborative Filter (CF):**
1. Membuat 200 pengguna dummy (user_1 hingga user_200).
2. Mengenerate interaksi acak (produk + rating) untuk tiap user.
3. Menyimpan data interaksi ke dalam DataFrame `df_interactions`.
4. Mengubah data menjadi `user-item matrix`.
5. Mengubah matrix menjadi sparse matrix (csr_matrix)

📌 **Alasan langkah-langkah Collaborative Filter tersebut dilakukan:**
- Karema dataset asli tidak mengandung data interaksi pengguna, maka dibuat data simulatif untuk memungkinkan training model CF.
- Membuat skenario realistis untuk collaborative filtering. Dimana setiap pengguna memberikan rating terhadap 5–15 produk acak. Rating dibuat antara 3 sampai 5 untuk mensimulasikan pengalaman positif (interaksi pengguna).
- Hasil data interaksi disimpan ke dataframe `df_interactions` dengan struktur data `user-item-rating` dibutuhkan sebagai input model CF nantinya.
- Pada `user-item matrix` berisikan 3 kolom yang terdiri dari pengguna, produk, dan rating. Kolom ini diperlukan untuk representasi numerik dalam training model.
- Mengubah matrik menjadi sparse matrix untuk menghemat memori dan mempercepat perhitungan saat training model CF dengan bantuan format dari library implicit.

---

## Modeling

### 🔍**MODELING : Content-Based Filtering (CB)**

Cara kerja model ini: <br>
1. Vektorisasi ingredients menggunakan TF-IDF
   ```
   vectorizer = TfidfVectorizer(stop_words='english')
   ingredient_matrix = vectorizer.fit_transform(df_cb['clean_ingreds'])
   ```
   Kalimat pada kolom `clean_ingreds` diubah menjadi vektor numerik menggunakan TF-IDF (Term Frequency - Inverse Document Frequency).

   TF-IDF memberi bobot lebih besar pada kata-kata yang unik dalam satu produk, tapi jarang muncul di produk lain, sehingga efektif membedakan kandungan tiap produk.

2. Hitung kemiripan antar produk
   ```
   cosine_sim = cosine_similarity(ingredient_matrix, ingredient_matrix)
   ```
   Kemiripan antar produk dihitung menggunakan cosine similarity, yaitu menghitung sudut antar vektor TF-IDF.

3. Fungsi rekomendasi
   ```
   def recommend_content(product_name, top_n=10):
   ...
   ```
   Fungsi akan mencari nama produk yang dimasukkan user.
   Lalu mengambil produk dengan kemiripan tertinggi berdasarkan nilai cosine similarity.
   Mengembalikan Top-N produk yang komposisi kandungannya paling mirip.

✅ Kelebihan Content-Based Filtering:
- Tidak membutuhkan data interaksi pengguna (berbasis deskripsi produk saja).
- Bisa merekomendasikan produk baru selama punya deskripsi.
- Objektif, karena berdasarkan konten nyata (ingredients), bukan ulasan subyektif.
  
❌ Kekurangan Content-Based Filtering:
- Terbatas hanya pada kemiripan atribut produk, tidak mempertimbangkan pengalaman pengguna nyata.
- Sulit untuk merekomendasikan produk yang berbeda namun relevan (tidak mirip secara konten tapi disukai oleh user serupa).
- Rentan terhadap cold start pada informasi produk (jika ingredients tidak lengkap).


### 🔍**MODELING : Collaborative Filtering (CF)**

Cara kerja model ini: <br>
1. Inisialisasi dan pelatihan model KNN
   ```
   model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
   model_knn.fit(sparse_matrix.T)
   ```
   Menggunakan K-Nearest Neighbors (KNN) berbasis cosine similarity.
   Model dilatih pada sparse matrix transpos (sparse_matrix.T) agar sistem menghitung kemiripan antar produk, bukan antar pengguna.

2. Fungsi rekomendasi berbasis interaksi
   ```
   def recommend_collab(product_name, top_n=10):
   ...
   ```
   Fungsi mencari produk yang memiliki pola interaksi pengguna serupa.
   Artinya, produk yang sering diberikan rating tinggi oleh pengguna yang sama dianggap "mirip".

✅ Kelebihan Collaborative Filtering:
- Mengandalkan pengalaman nyata pengguna, lebih sesuai dengan preferensi aktual.
- Bisa menangkap korelasi tersembunyi antar produk yang tidak bisa dilihat hanya dari konten (misalnya dua produk berbeda tapi sering dibeli bersama).
- Dinamis, menyesuaikan dengan perubahan pola perilaku pengguna.

❌ Kekurangan Collaborative Filtering:
- Rentan terhadap masalah cold start untuk produk baru (tidak ada interaksi → tidak bisa direkomendasikan).
- Butuh data interaksi yang cukup besar dan representatif.
- Jika data sparsity tinggi (banyak nilai kosong), performa bisa buruk.

---

## Evaluation

Pada tahap Evaluation dibedakan tahapannya berdasarkan kedual model yaitu evaluasi untuk CB dan evaluasi CF.

### 🏷️ Evaluasi Model Content-Based Filtering (CB)

1. Evaluasi cosine similarity
   `evaluate_cb()` <br>
   Hal yang diukur: Rata-rata cosine similarity antar produk.<br>
   Cara kerja: 
     - Ambil top-N produk yang paling mirip (berdasarkan TF-IDF ingredients).
     - Hitung rata-rata skor kemiripan.
   
   Kelebihan: Cepat dan langsung mengukur seberapa "mirip" produk secara konten. <br>
   Kekurangan: Tidak mengevaluasi apakah produk itu benar-benar relevan untuk pengguna. <br>

2. Ground Truth Evaluation Metrics (Berbasis product_type)
   `cb_precision_at_k_gt()` <br>
   Hal yang diukur: Berapa banyak dari top-K rekomendasi yang benar-benar relevan (produk dengan product_type yang sama). <br>
   Cara kerja:
     - Ambil daftar ground truth produk sejenis.
     - Bandingkan dengan hasil rekomendasi.
   
   Hitung: jumlah yang cocok / K. <br>

3. Metrik Tambahan Evaluasi
   - Precision@K <br>
     `cb_recall_at_k_gt()` <br>
     Hal yang diukur: Dari semua produk sejenis, berapa yang berhasil ditemukan oleh model. <br>
     Cara kerja: Sama seperti precision, tapi menghitung: jumlah cocok / jumlah ground truth. <br>
   - Recall@K <br>
     `cb_f1_at_k_gt()` <br>
     Hal yang diukur: Kombinasi seimbang antara Precision dan Recall. <br>
     Formula: <br>
     ```
            Precision+Recall     
     F1 = --------------------   
            2×Precision×Recall   
     ```
     Tujuannya: mengukur keseimbangan antara akurasi dan kelengkapan rekomendasi. <br>
     Kelebihan: Relevan jika ingin performa seimbang antara ketepatan & cakupan. <br>
   - MAP@K <br>
     `cb_map_at_k_gt()` dan `cb_average_precision_at_k_gt()` <br>
     Hal yang diukur: Posisi produk relevan di dalam list rekomendasi. <br>
     Cara kerja: <br>
        - Setiap kali produk yang benar ditemukan, precision dihitung & diakumulasi. <br>
        - Hasil akhirnya adalah Mean Average Precision dari banyak produk. <br>
     
     Kelebihan: Metrik yang sensitif terhadap urutan rekomendasi—jika produk relevan muncul di atas, skornya tinggi. <br>
     Cocok untuk: Menilai apakah model meletakkan produk penting di posisi atas (Top-N relevan di awal). <br>

#### Berikut merupakan ringkasan evaluasi model CB

**Rangkuman Cara Kerja Metrik:**

| Metrik                | Fokus Evaluasi                | Cocok Untuk                                 |
| --------------------- | ----------------------------- | ------------------------------------------- |
| Cosine Similarity Avg | Seberapa mirip dari konten    | Mengecek logika dasar kemiripan ingredients |
| Precision\@K          | Ketepatan rekomendasi         | Apakah yang direkomendasikan benar relevan  |
| Recall\@K             | Cakupan rekomendasi           | Apakah semua produk relevan ditemukan       |
| F1\@K                 | Keseimbangan presisi & recall | Evaluasi umum performa                      |
| MAP\@K                | Urutan produk relevan         | Evaluasi kualitas ranking/top list          |

**Hasil Evaluasi:**

| Metrik        | Nilai  | Interpretasi                                                             |
| ------------- | ------ | ------------------------------------------------------------------------ |
| Precision\@10 | 0.035  | Hanya 3.5% produk rekomendasi yang benar-benar sejenis                   |
| Recall\@10    | 0.035  | Hanya 3.5% produk sejenis yang berhasil direkomendasikan                 |
| F1\@10        | 0.035  | Model kurang baik dalam hal akurasi maupun cakupan                       |
| MAP\@10       | 0.0155 | Produk relevan cenderung muncul di posisi bawah dalam daftar rekomendasi |



### 🏷️ Evaluasi Model Content-Based Filtering (CB)



---

## Visualisasi



---

## Inferences



---

_Terima kasih telah membaca laporan proyek ini._
