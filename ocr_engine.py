# -*- coding: utf-8 -*-
"""
ocr_engine.py
Wrapper OCR yang menyeragamkan output EasyOCR maupun PaddleOCR menjadi daftar
baris teks (list of str) yang siap diparsing oleh ktp_parser.py.

Model dimuat sekali (lazy singleton) -- penting untuk batch processing supaya
tidak reload model tiap gambar (bisa memakan waktu puluhan detik per load).
"""

import logging
import numpy as np

from . import config

logger = logging.getLogger("ktp_ocr.ocr_engine")

_reader = None


def _get_easyocr_reader():
    global _reader
    if _reader is None:
        import easyocr
        logger.info("Memuat model EasyOCR (bahasa: %s)...", config.OCR_LANGUAGES_EASYOCR)
        try:
            _reader = easyocr.Reader(config.OCR_LANGUAGES_EASYOCR, gpu=config.OCR_GPU)
        except Exception as e:
            # Beberapa kombinasi bahasa tidak kompatibel satu sama lain di EasyOCR;
            # fallback aman ke bahasa Inggris saja (karakter Latin KTP tetap terbaca).
            logger.warning(
                "Gagal memuat EasyOCR dgn bahasa %s (%s) -- fallback ke ['en'] saja.",
                config.OCR_LANGUAGES_EASYOCR, e
            )
            _reader = easyocr.Reader(["en"], gpu=config.OCR_GPU)
    return _reader


def _get_paddleocr_reader():
    global _reader
    if _reader is None:
        from paddleocr import PaddleOCR
        logger.info("Memuat model PaddleOCR (bahasa: %s)...", config.OCR_LANGUAGES_PADDLEOCR)
        _reader = PaddleOCR(use_angle_cls=True, lang=config.OCR_LANGUAGES_PADDLEOCR, show_log=False)
    return _reader


def _group_into_lines(detections, y_tolerance=None):
    """
    detections: list of (bbox, text, confidence) -- format umum EasyOCR/PaddleOCR
    setelah dinormalisasi. bbox = 4 titik [[x,y], [x,y], [x,y], [x,y]].

    Mengelompokkan kata-kata hasil deteksi jadi baris berdasarkan kedekatan
    posisi vertikal (y), lalu urutkan tiap baris berdasarkan posisi horizontal
    (x) -- merekonstruksi urutan baca alami dari kartu KTP.
    """
    y_tolerance = y_tolerance or config.LINE_GROUPING_Y_TOLERANCE

    items = []
    for bbox, text, conf in detections:
        ys = [p[1] for p in bbox]
        xs = [p[0] for p in bbox]
        items.append({
            "text": text,
            "y": sum(ys) / len(ys),
            "x": min(xs),
            "conf": conf,
        })

    items.sort(key=lambda i: i["y"])

    lines = []
    current_line = []
    current_y = None
    for item in items:
        if current_y is None or abs(item["y"] - current_y) <= y_tolerance:
            current_line.append(item)
            current_y = item["y"] if current_y is None else (current_y + item["y"]) / 2.0
        else:
            lines.append(current_line)
            current_line = [item]
            current_y = item["y"]
    if current_line:
        lines.append(current_line)

    line_texts = []
    for line in lines:
        line_sorted = sorted(line, key=lambda i: i["x"])
        line_texts.append(" ".join(i["text"] for i in line_sorted))

    return line_texts


def run_ocr(gray_or_bgr_image):
    """
    Jalankan OCR pada gambar (grayscale atau BGR, numpy array).
    Mengembalikan list baris teks (list[str]), terurut sesuai posisi baris
    asli di kartu.
    """
    engine = config.OCR_ENGINE.lower()

    if engine == "easyocr":
        reader = _get_easyocr_reader()
        raw = reader.readtext(gray_or_bgr_image)
        # EasyOCR readtext -> list of (bbox, text, confidence)
        detections = [(bbox, text, conf) for bbox, text, conf in raw]

    elif engine == "paddleocr":
        reader = _get_paddleocr_reader()
        raw = reader.ocr(gray_or_bgr_image, cls=True)
        detections = []
        # PaddleOCR: list per gambar -> list of [bbox, (text, confidence)]
        page = raw[0] if raw else []
        for line in page or []:
            bbox, (text, conf) = line
            detections.append((bbox, text, conf))

    else:
        raise ValueError(f"OCR_ENGINE tidak dikenal: {config.OCR_ENGINE}")

    if not detections:
        return []

    return _group_into_lines(detections)
