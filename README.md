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
   - Cocok untuk rekomendasi yang objektif dan transparan, tidak tergantung pada rating atau review.

2. **Collaborative Filtering (CF)**  
   - Menggunakan matrix factorization dari library `implicit`.
   - Berdasarkan interaksi historis pengguna dengan produk (user-product interaction matrix).
   - Memungkinkan rekomendasi bersifat personal dengan mempelajari pola preferensi pengguna.
     
---

## Data Understanding
Paragraf awal bagian ini menjelaskan informasi mengenai jumlah data, kondisi data, dan informasi mengenai data yang digunakan. Sertakan juga sumber atau tautan untuk mengunduh dataset. Contoh: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Restaurant+%26+consumer+data).

Selanjutnya, uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:  

Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
- accepts : merupakan jenis pembayaran yang diterima pada restoran tertentu.
- cuisine : merupakan jenis masakan yang disajikan pada restoran.
- dst

**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data beserta insight atau exploratory data analysis.

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model sisten rekomendasi yang Anda buat untuk menyelesaikan permasalahan. Sajikan top-N recommendation sebagai output.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menyajikan dua solusi rekomendasi dengan algoritma yang berbeda.
- Menjelaskan kelebihan dan kekurangan dari solusi/pendekatan yang dipilih.

## Evaluation
Pada bagian ini Anda perlu menyebutkan metrik evaluasi yang digunakan. Kemudian, jelaskan hasil proyek berdasarkan metrik evaluasi tersebut.

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.
