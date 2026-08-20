# -*- coding: utf-8 -*-
"""
config.py
Konfigurasi terpusat untuk pipeline OCR KTP.
Ubah nilai-nilai di sini untuk tuning tanpa menyentuh logic di modul lain.
"""

import os

# ---------------------------------------------------------------------------
# PATH
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # root proyek (satu level di atas folder ktp_ocr/)
DEFAULT_INPUT_DIR = os.path.join(BASE_DIR, "input_ktp")
DEFAULT_OUTPUT_ENHANCED_DIR = os.path.join(BASE_DIR, "output_enhanced")
DEFAULT_EXCEL_OUTPUT = os.path.join(BASE_DIR, "hasil_ktp.xlsx")

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

# ---------------------------------------------------------------------------
# PERSPECTIVE TRANSFORM / CARD DETECTION
# ---------------------------------------------------------------------------
# Rasio kartu KTP Indonesia (85.6mm x 54mm)
KTP_ASPECT_RATIO = 85.6 / 54.0  # ~1.585

# Ukuran output setelah warp (piksel) -- dikalikan faktor agar cukup detail utk OCR
WARP_OUTPUT_WIDTH = 1013
WARP_OUTPUT_HEIGHT = 638

# Kontur kandidat kartu harus menutupi minimal sekian % area gambar
MIN_CARD_AREA_RATIO = 0.08
MAX_CARD_AREA_RATIO = 0.95

# Toleransi rasio bentuk agar dianggap "masuk akal sebagai kartu ID"
# (mencegah salah crop ke tangan/background -- lebih baik gagal deteksi
# daripada salah potong)
MIN_PLAUSIBLE_ASPECT = 1.25
MAX_PLAUSIBLE_ASPECT = 2.4

# ---------------------------------------------------------------------------
# IMAGE ENHANCEMENT
# ---------------------------------------------------------------------------
CLAHE_CLIP_LIMIT = 3.0
CLAHE_TILE_GRID_SIZE = (8, 8)

DENOISE_STRENGTH = 8          # fastNlMeansDenoising 'h' parameter
UNSHARP_AMOUNT = 1.4          # kekuatan unsharp mask
UNSHARP_SIGMA = 1.0

# Deskew halus (rentang pencarian sudut, derajat)
DESKEW_ANGLE_RANGE = (-12, 12)
DESKEW_ANGLE_STEP = 1.0

# ---------------------------------------------------------------------------
# SUPER RESOLUTION (Real-ESRGAN)
# ---------------------------------------------------------------------------
SR_ENABLED_DEFAULT = True
SR_SCALE = 2                  # 2x atau 4x
SR_MODEL_NAME = "RealESRGAN_x4plus"
SR_MODEL_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/"
    "v0.1.0/RealESRGAN_x4plus.pth"
)
SR_MODEL_DIR = os.path.join(BASE_DIR, "sr_weights")
SR_MODEL_PATH = os.path.join(SR_MODEL_DIR, "RealESRGAN_x4plus.pth")

# Kalau gambar sudah cukup besar, super-resolution dilewati (tidak perlu,
# dan menghemat waktu proses) -- ambang dalam piksel (sisi terpanjang)
SR_SKIP_IF_LONGEST_SIDE_ABOVE = 1600

# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------
OCR_ENGINE = "easyocr"        # "easyocr" atau "paddleocr"
OCR_LANGUAGES_EASYOCR = ["id", "en"]
OCR_LANGUAGES_PADDLEOCR = "id"
OCR_GPU = False                # set True kalau CUDA tersedia

# Toleransi vertikal (piksel, pada gambar hasil warp WARP_OUTPUT_HEIGHT)
# untuk mengelompokkan hasil deteksi kata jadi satu "baris"
LINE_GROUPING_Y_TOLERANCE = 14

# ---------------------------------------------------------------------------
# VALIDASI NIK
# ---------------------------------------------------------------------------
NIK_LENGTH = 16
NIK_REGEX_LOOSE = r"[0-9OoIiLlBbSsZz]{14,18}"

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
LOG_LEVEL = "INFO"
LOG_FILE = os.path.join(BASE_DIR, "proses_ktp.log")
