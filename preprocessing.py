# -*- coding: utf-8 -*-
"""
preprocessing.py
Deteksi & pelurusan kartu KTP (perspective transform) + enhancement gambar
(CLAHE, denoise, unsharp masking) untuk mempersiapkan gambar sebelum OCR.

Didesain resilient: setiap tahap punya fallback -- kalau deteksi kontur kartu
gagal, pipeline tetap lanjut memakai gambar aslinya (hanya di-deskew ringan),
bukan berhenti/crash.
"""

import logging
import numpy as np
import cv2

from . import config

logger = logging.getLogger("ktp_ocr.preprocessing")


# ---------------------------------------------------------------------------
# Deteksi kartu & perspective transform (4-point transform)
# ---------------------------------------------------------------------------

def _order_points(pts):
    """Urutkan 4 titik jadi [top-left, top-right, bottom-right, bottom-left]."""
    pts = np.array(pts, dtype="float32")
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left: x+y terkecil
    rect[2] = pts[np.argmax(s)]   # bottom-right: x+y terbesar

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right: x-y terkecil
    rect[3] = pts[np.argmax(diff)]  # bottom-left: x-y terbesar
    return rect


def _quad_plausible_as_card(pts):
    """
    Cek apakah 4 titik ini masuk akal sebagai bentuk kartu ID (rasio ~1.586:1,
    dengan toleransi untuk distorsi perspektif). Mencegah salah crop ke objek
    lain (tangan, meja, dsb) -- lebih aman gagal deteksi daripada salah potong.
    """
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect

    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)

    w = (width_top + width_bottom) / 2.0
    h = (height_left + height_right) / 2.0

    if w < 20 or h < 20:
        return False

    ratio = max(w, h) / min(w, h)
    return config.MIN_PLAUSIBLE_ASPECT < ratio < config.MAX_PLAUSIBLE_ASPECT


def _find_card_quad(edges, img_area):
    """Cari kontur 4-titik (atau minAreaRect fallback) yang masuk akal sebagai kartu."""
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    candidates = [
        c for c in contours
        if config.MIN_CARD_AREA_RATIO * img_area < cv2.contourArea(c) < config.MAX_CARD_AREA_RATIO * img_area
    ]
    candidates.sort(key=cv2.contourArea, reverse=True)
    candidates = candidates[:6]

    # Pass 1: approxPolyDP -> harus persis 4 titik & bentuknya masuk akal
    for c in candidates:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            pts = approx.reshape(4, 2)
            if _quad_plausible_as_card(pts):
                return pts

    # Pass 2 (fallback): minAreaRect dari kontur besar, tetap divalidasi rasio
    for c in candidates:
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        if _quad_plausible_as_card(box):
            return box

    return None


def detect_and_warp_card(image_bgr):
    """
    Deteksi tepi kartu KTP di foto dan luruskan (perspective transform) jadi
    tampak difoto tegak lurus dari atas.

    Returns:
        (warped_image, card_found: bool)
        Kalau card_found False, warped_image = gambar original (tidak diubah).
    """
    try:
        h, w = image_bgr.shape[:2]
        max_dim = 1000
        scale = min(1.0, max_dim / max(h, w))
        resized = cv2.resize(image_bgr, (int(w * scale), int(h * scale))) if scale < 1.0 else image_bgr.copy()

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        median = float(np.median(blurred))
        low_t = int(max(0, 0.66 * median))
        high_t = int(min(255, 1.33 * median))

        img_area = resized.shape[0] * resized.shape[1]

        # Percobaan 1: Canny dengan ambang otomatis dari median kecerahan
        edged = cv2.Canny(blurred, low_t, high_t)
        edged = cv2.dilate(edged, np.ones((3, 3), np.uint8), iterations=2)
        quad = _find_card_quad(edged, img_area)

        # Percobaan 2 (fallback): threshold Otsu -> Canny, kalau percobaan 1 gagal
        if quad is None:
            _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            edged2 = cv2.Canny(otsu, 50, 150)
            edged2 = cv2.dilate(edged2, np.ones((3, 3), np.uint8), iterations=2)
            quad = _find_card_quad(edged2, img_area)

        if quad is None:
            logger.info("Kartu tidak terdeteksi -- lanjut pakai gambar asli (fallback).")
            return image_bgr, False

        # Skalakan titik kembali ke resolusi asli
        quad_full_res = quad.astype("float32") / scale
        rect = _order_points(quad_full_res)
        (tl, tr, br, bl) = rect

        out_w, out_h = config.WARP_OUTPUT_WIDTH, config.WARP_OUTPUT_HEIGHT
        dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image_bgr, M, (out_w, out_h))
        return warped, True

    except Exception as e:
        logger.warning("Deteksi/warp kartu gagal (%s) -- fallback ke gambar asli.", e)
        return image_bgr, False


# ---------------------------------------------------------------------------
# Deskew halus (untuk kemiringan kecil sisa setelah warp, atau kalau warp gagal)
# ---------------------------------------------------------------------------

def _projection_variance(binary_img):
    row_sums = np.sum(255 - binary_img, axis=1).astype(np.float64)
    return float(np.var(row_sums))


def fine_deskew(gray_img):
    """
    Cari & koreksi sudut kemiringan kecil lewat pencarian sudut yang
    memaksimalkan variansi proyeksi horizontal teks (metode umum utk deskew
    dokumen). Mengembalikan (gambar_terkoreksi, sudut_derajat).
    """
    small = cv2.resize(gray_img, (300, int(300 * gray_img.shape[0] / gray_img.shape[1])))
    _, binary_small = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    best_angle = 0.0
    best_score = -1.0
    lo, hi = config.DESKEW_ANGLE_RANGE
    for angle in np.arange(lo, hi + 0.01, config.DESKEW_ANGLE_STEP):
        h, w = binary_small.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rotated = cv2.warpAffine(binary_small, M, (w, h), borderValue=255)
        score = _projection_variance(rotated)
        if score > best_score:
            best_score = score
            best_angle = angle

    if abs(best_angle) < 0.4:
        return gray_img, 0.0

    h, w = gray_img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), best_angle, 1.0)
    rotated_full = cv2.warpAffine(
        gray_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated_full, float(best_angle)


# ---------------------------------------------------------------------------
# Deteksi warna vs hitam-putih
# ---------------------------------------------------------------------------

def is_color_image(image_bgr, sat_threshold=18, ratio_threshold=0.03):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    colored_ratio = float(np.mean(sat > sat_threshold))
    return colored_ratio > ratio_threshold


# ---------------------------------------------------------------------------
# Enhancement: CLAHE, denoise, unsharp masking
# ---------------------------------------------------------------------------

def apply_clahe(gray_img):
    clahe = cv2.createCLAHE(
        clipLimit=config.CLAHE_CLIP_LIMIT,
        tileGridSize=config.CLAHE_TILE_GRID_SIZE
    )
    return clahe.apply(gray_img)


def denoise(gray_img):
    return cv2.fastNlMeansDenoising(gray_img, h=config.DENOISE_STRENGTH)


def unsharp_mask(gray_img, amount=None, sigma=None):
    """Sharpening klasik: original + amount * (original - gaussian_blur)."""
    amount = config.UNSHARP_AMOUNT if amount is None else amount
    sigma = config.UNSHARP_SIGMA if sigma is None else sigma

    blurred = cv2.GaussianBlur(gray_img, (0, 0), sigma)
    sharpened = cv2.addWeighted(gray_img, 1 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def enhance_pipeline(image_bgr):
    """
    Pipeline enhancement lengkap untuk satu gambar KTP (BGR, sudah/atau belum
    di-warp). Mengembalikan dict berisi gambar hasil di beberapa tahap +
    metadata proses.

    Urutan: deteksi warna -> grayscale -> deskew halus -> CLAHE -> denoise ->
    unsharp mask -> hasil akhir (grayscale, siap OCR / siap disimpan).
    """
    is_color = is_color_image(image_bgr)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray, deskew_angle = fine_deskew(gray)

    enhanced = apply_clahe(gray)
    enhanced = denoise(enhanced)
    enhanced = unsharp_mask(enhanced)

    # Versi berwarna yang diselaraskan (untuk disimpan sebagai "foto bersih",
    # supaya foto wajah di KTP tidak ikut jadi hitam-putih di file output)
    enhanced_color = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    if is_color:
        # Terapkan CLAHE per-channel L (Lab color space) supaya warna asli
        # tetap terjaga tapi kontrasnya ikut membaik
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = apply_clahe(l)
        enhanced_color = cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)
        enhanced_color = cv2.fastNlMeansDenoisingColored(enhanced_color, h=config.DENOISE_STRENGTH)

    return {
        "gray_enhanced": enhanced,        # untuk OCR
        "color_enhanced": enhanced_color, # untuk disimpan sebagai file output
        "is_color": is_color,
        "deskew_angle": deskew_angle,
    }
