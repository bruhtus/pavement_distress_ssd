## Dokumentasi
Berikut merupakan dokumentasi penggunaan aplikasi deteksi kerusakan jalan:

1. Masukkan video yang akan diproses difolder 'input'.
2. Jika user memiliki konfigurasi sendiri, konfigurasi user dapat dimasukkan difolder 'configs'.
3. Jika user menggunakan konfigurasi sendiri, maka masukkan folder model dari hasil training dengan konfigurasi tersebut ke folder 'outputs'.
4. Setelah semua file dan folder yang diperlukan telah dimasukkan ke folder yang telah ditentukan, lakukan refresh pada browser untuk melihat file dan folder yang baru ditambahkan.
5. Hasil dari aplikasi ini disimpan difolder 'results'. Untuk hasil video disimpan dalam format .mp4 dan hasil perhitungan kerusakan jalan disimpan dalam file .txt.
6. Format nama hasil dari aplikasi yaitu: 'Tahun-Bulan-Tanggal_Jam-Menit-Detik_Konfigurasi'.
7. Aplikasi ini hanya untuk proses testing, untuk proses training dapat menggunakan train.py dengan command *python train.py --config-file configs/konfigurasi.yaml*. Untuk proses training disarankan menggunakan cloud computing seperti google colab (jika data yang digunakan tidak confidential). Proses training tidak harus dilakukan di cloud computing platform dan dapat dilakukan di komputer user (local)
