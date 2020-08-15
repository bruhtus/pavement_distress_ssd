## Dokumentasi
### Bahasa
Berikut merupakan dokumentasi penggunaan aplikasi deteksi kerusakan jalan:

1. Masukkan video yang akan diproses difolder 'input'.
2. Jika user memiliki konfigurasi sendiri, konfigurasi user dapat dimasukkan difolder 'configs'.
3. Jika user menggunakan konfigurasi sendiri, maka masukkan folder model dari hasil training dengan konfigurasi tersebut ke folder 'outputs'.
4. Setelah semua file dan folder yang diperlukan telah dimasukkan ke folder yang telah ditentukan, lakukan refresh pada browser untuk melihat file dan folder yang baru ditambahkan.
5. Hasil dari aplikasi ini disimpan difolder 'results'. Untuk hasil video disimpan dalam format .mp4 dan hasil perhitungan kerusakan jalan disimpan dalam file .txt.
6. Format nama hasil dari aplikasi yaitu: 'Tahun-Bulan-Tanggal_Jam-Menit-Detik_Konfigurasi'.
7. Aplikasi ini hanya untuk proses testing, untuk proses training dapat menggunakan train.py dengan command *python train.py --config-file configs/konfigurasi.yaml*. Untuk proses training disarankan menggunakan cloud computing seperti google colab (jika data yang digunakan tidak confidential). Proses training tidak harus dilakukan di cloud computing platform dan dapat dilakukan di komputer user (local)

### English
Here is the documentation how to use pavement distress detector:

1. Copy or move video user want to process into folder 'input'.
2. If user have their own configuration, then user can copy or move the configuration file into folder 'configs'.
3. If user use their own configuration, then user should copy or move folder that has training results with user's configuration into folder 'outputs'.
4. After every file and folder in the right place, then hit refresh (or F5) to see the newly added file and folder.
5. The result of this application was saved in folder 'results'. For video result saved in format .mp4 and for counting result saved in format .txt.
6. The format name of the result is 'Year-Month-Date_Hours-Minutes-Seconds_configuration_name'.
7. This aplication is only for testing purposes.
