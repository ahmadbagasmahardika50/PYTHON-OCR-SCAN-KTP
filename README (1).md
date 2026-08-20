# Batch OCR KTP → Excel (Python, Advanced Pipeline)

Skrip Python modular untuk mengotomatisasi ekstraksi data KTP dari banyak
gambar sekaligus: **deteksi & pelurusan kartu (perspective transform) →
AI Super Resolution (Real-ESRGAN) → enhancement (CLAHE/denoise/sharpen) →
OCR (EasyOCR/PaddleOCR) → parsing & validasi NIK → ekspor Excel**, plus
gambar hasil pembersihan disimpan ke folder terpisah.

Didesain **resilient**: satu file yang rusak, super-resolution yang gagal
dimuat, atau kartu yang gagal terdeteksi TIDAK menghentikan proses batch —
semuanya dicatat di kolom "Status Scan" / "Catatan" dan lanjut ke file
berikutnya.

## Struktur Folder

```
ktp_ocr_advanced/
├── run.py                     <- jalankan ini
├── requirements.txt
├── input_ktp/                 <- taruh foto/scan KTP di sini
├── output_enhanced/            <- gambar hasil enhancement disimpan otomatis di sini
└── ktp_ocr/                   <- package modul
    ├── config.py               (semua parameter tuning terpusat di sini)
    ├── preprocessing.py        (deteksi kartu, perspective transform, CLAHE, denoise, unsharp mask, deskew)
    ├── super_resolution.py     (wrapper Real-ESRGAN + fallback bicubic)
    ├── ocr_engine.py           (wrapper EasyOCR / PaddleOCR)
    ├── ktp_parser.py           (parsing field + validasi NIK 16 digit)
    ├── excel_writer.py         (penulisan file .xlsx dengan styling)
    └── main.py                 (orkestrator batch + error handling)
```

## 1. Instalasi

```bash
pip install opencv-python numpy pandas openpyxl easyocr
```

Itu saja **sudah cukup untuk menjalankan pipeline lengkap** (perspective
transform, enhancement, OCR, Excel) — hanya belum termasuk AI Super
Resolution.

### (Opsional) Super Resolution dengan Real-ESRGAN
Super-resolution butuh `torch` yang cukup besar (~1-2GB). Kalau ingin
mengaktifkannya:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install basicsr realesrgan
```

Kalau library ini **tidak diinstall**, atau modelnya gagal di-download saat
pertama kali dipakai, skrip **tidak akan error** — otomatis fallback ke
upscaling bicubic biasa (lebih cepat, kualitas peningkatan lebih sederhana)
dan proses tetap lanjut. Kolom "Metode Super Resolution" di Excel akan
menunjukkan mana yang benar-benar pakai `realesrgan` vs `bicubic_fallback`.

### (Opsional) PaddleOCR sebagai alternatif EasyOCR
```bash
pip install paddleocr paddlepaddle
```
lalu jalankan dengan `--ocr paddleocr`.

## 2. Cara Pakai

```bash
# Taruh foto KTP (.jpg/.jpeg/.png/.bmp) di folder input_ktp/, lalu:
python run.py

# Custom folder input/output:
python run.py --input foto_ktp_saya --output hasil.xlsx

# Nonaktifkan super-resolution (lebih cepat, kalau torch belum diinstall):
python run.py --no-sr

# Pakai PaddleOCR alih-alih EasyOCR:
python run.py --ocr paddleocr

# Pakai GPU (kalau ada CUDA):
python run.py --gpu
```

Semua argumen opsional — cukup `python run.py` juga jalan dengan folder
`input_ktp/` dan output `hasil_ktp.xlsx` secara default.

## 3. Yang Dihasilkan

**File Excel** (`hasil_ktp.xlsx`) — satu baris per gambar, kolom:
`No, Nama File Original, Path Foto Enhanced, NIK, Nama, Tempat/Tgl Lahir,
Jenis Kelamin, Gol. Darah, Alamat, RT/RW, Kel/Desa, Kecamatan, Agama,
Status Perkawinan, Pekerjaan, Kewarganegaraan, Berlaku Hingga, Provinsi,
Kabupaten/Kota, Tipe Gambar, Kartu Terdeteksi (Perspective), Metode Super
Resolution, Sudut Koreksi (deskew), Skor Kelengkapan (%), Status Scan,
Catatan, Waktu Proses (detik)`.

Baris diberi warna otomatis: **hijau** = Valid, **kuning** = Perlu
Verifikasi, **merah** = Error — supaya cepat ditinjau.

**Folder `output_enhanced/`** — gambar KTP hasil perspective transform +
enhancement, disimpan sebagai `enhanced_<nama_file_asli>.jpg`.

**Status Scan** yang mungkin muncul:
- `Valid` — NIK persis 16 digit & data utama terbaca
- `Perlu Verifikasi` — NIK tidak 16 digit, nama tidak terbaca, atau banyak
  field kosong (perlu dicek manual)
- `Error` — file gagal dibaca sama sekali (rusak/format tidak didukung)

## 4. Tuning

Semua parameter (ambang deteksi kartu, kekuatan CLAHE/denoise/sharpening,
rentang deskew, ukuran output warp, dsb) ada di **`ktp_ocr/config.py`** —
ubah di situ tanpa perlu menyentuh logic di file lain.

## 5. Yang Sudah Diuji

Pipeline ini sudah diuji end-to-end (bukan cuma ditulis) dengan EasyOCR
sungguhan pada gambar KTP sintetis yang mencakup: normal berwarna, miring
berwarna, hitam-putih, miring+hitam-putih, dan file gambar korup — semua
field (termasuk NIK 16 digit) berhasil diekstrak dengan akurat, dan file
korup ditangani sebagai "Error" tanpa menghentikan batch.
Super-resolution (Real-ESRGAN) sendiri belum diuji end-to-end karena
keterbatasan waktu unduh model di lingkungan pengujian — tapi jalur
fallback-nya (bicubic upscaling saat library/model tidak tersedia) sudah
diuji dan bekerja sesuai desain resilient di atas.
