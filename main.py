# -*- coding: utf-8 -*-
"""
main.py
Orkestrator utama pipeline OCR KTP.

Alur per gambar:
  1. Baca gambar (skip + catat error kalau file rusak/tidak terbaca)
  2. Deteksi kartu & perspective transform (fallback: gambar asli)
  3. Super-resolution (Real-ESRGAN, fallback: bicubic upscaling)
  4. Enhancement (CLAHE, denoise, unsharp mask)
  5. Simpan gambar hasil enhancement ke output_enhanced/
  6. OCR + parsing field KTP
  7. Validasi NIK (harus 16 digit)
  8. Kumpulkan hasil -> tulis ke Excel di akhir

Didesain supaya SATU FILE BERMASALAH TIDAK MENGHENTIKAN SELURUH PROSES --
setiap tahap dibungkus try/except, error dicatat di kolom "Catatan" dan
"Status Scan" = "Error", lalu lanjut ke file berikutnya.

Jalankan:
    python main.py --input input_ktp --output hasil_ktp.xlsx
    python main.py --input input_ktp --output hasil_ktp.xlsx --no-sr
    python main.py --input input_ktp --output hasil_ktp.xlsx --ocr paddleocr
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime

import cv2

# Mendukung dijalankan langsung (python main.py) maupun sebagai modul paket
try:
    from . import config
    from . import preprocessing
    from . import super_resolution
    from . import ocr_engine
    from . import ktp_parser
    from . import excel_writer
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ktp_ocr import config, preprocessing, super_resolution, ocr_engine, ktp_parser, excel_writer


def setup_logging(log_file=None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )


logger = logging.getLogger("ktp_ocr.main")


def find_images(input_dir):
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Folder input tidak ditemukan: {input_dir}")
    files = sorted(
        f for f in os.listdir(input_dir)
        if f.lower().endswith(config.VALID_EXTENSIONS)
    )
    return files


def process_one_image(filepath, filename, output_enhanced_dir, use_sr=True):
    """
    Proses satu gambar KTP secara end-to-end. SELALU mengembalikan sebuah
    dict record (tidak pernah raise exception ke pemanggil) -- error apa pun
    ditangkap dan dicatat di dalam record itu sendiri.
    """
    t0 = time.time()
    record = {"original_filename": filename}

    try:
        image_bgr = cv2.imread(filepath)
        if image_bgr is None:
            record["_status"] = "Error"
            record["_catatan"] = "Gagal membaca file gambar (kemungkinan rusak/format tidak didukung)"
            record["elapsed_sec"] = round(time.time() - t0, 2)
            return record
    except Exception as e:
        record["_status"] = "Error"
        record["_catatan"] = f"Exception saat membaca file: {e}"
        record["elapsed_sec"] = round(time.time() - t0, 2)
        return record

    # 1. Deteksi kartu & perspective transform
    try:
        warped, card_found = preprocessing.detect_and_warp_card(image_bgr)
    except Exception as e:
        logger.warning("[%s] Perspective transform gagal (%s) -- pakai gambar asli.", filename, e)
        warped, card_found = image_bgr, False
    record["kartu_terdeteksi"] = "Ya" if card_found else "Tidak"

    # 2. Super-resolution (best-effort, tidak boleh menghentikan proses)
    sr_method = "skipped"
    if use_sr:
        try:
            warped, sr_method = super_resolution.super_resolve(warped)
        except Exception as e:
            logger.warning("[%s] Super-resolution gagal total (%s) -- lanjut tanpa SR.", filename, e)
            sr_method = "gagal_total_dilewati"
    record["sr_method"] = sr_method

    # 3. Enhancement (CLAHE, denoise, unsharp mask)
    try:
        enh = preprocessing.enhance_pipeline(warped)
    except Exception as e:
        logger.error("[%s] Enhancement gagal (%s) -- OCR tetap dicoba pada gambar warp mentah.", filename, e)
        gray_fallback = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        enh = {
            "gray_enhanced": gray_fallback,
            "color_enhanced": warped,
            "is_color": preprocessing.is_color_image(warped) if warped is not None else False,
            "deskew_angle": 0.0,
        }

    record["tipe_gambar"] = "Berwarna" if enh["is_color"] else "Hitam-Putih"
    record["deskew_angle"] = enh["deskew_angle"]

    # 4. Simpan gambar hasil enhancement ke folder output
    try:
        os.makedirs(output_enhanced_dir, exist_ok=True)
        base_name = os.path.splitext(filename)[0]
        enhanced_filename = f"enhanced_{base_name}.jpg"
        enhanced_path = os.path.join(output_enhanced_dir, enhanced_filename)
        cv2.imwrite(enhanced_path, enh["color_enhanced"])
        record["enhanced_path"] = enhanced_path
    except Exception as e:
        logger.warning("[%s] Gagal menyimpan gambar enhanced (%s).", filename, e)
        record["enhanced_path"] = ""

    # 5. OCR + parsing
    try:
        lines = ocr_engine.run_ocr(enh["gray_enhanced"])
        if not lines:
            # Fallback: coba OCR di gambar berwarna hasil enhancement
            # (kadang lebih baik daripada versi grayscale murni untuk kartu
            # dengan latar belakang bermotif/watermark)
            lines = ocr_engine.run_ocr(enh["color_enhanced"])

        parsed = ktp_parser.parse_ktp_lines(lines)
        record.update(parsed)

    except Exception as e:
        logger.error("[%s] OCR/parsing gagal (%s).", filename, e)
        record["_status"] = "Error"
        record["_catatan"] = f"OCR/parsing gagal: {e}"
        record["_fill_ratio_pct"] = 0

    record["elapsed_sec"] = round(time.time() - t0, 2)
    return record


def run_batch(input_dir, output_excel, output_enhanced_dir, use_sr=True):
    files = find_images(input_dir)
    if not files:
        logger.warning("Tidak ada gambar (jpg/jpeg/png/bmp) ditemukan di '%s'.", input_dir)
        return []

    logger.info("Ditemukan %d gambar. Mulai batch processing...", len(files))

    records = []
    for idx, filename in enumerate(files, start=1):
        filepath = os.path.join(input_dir, filename)
        logger.info("[%d/%d] Memproses: %s", idx, len(files), filename)

        try:
            record = process_one_image(filepath, filename, output_enhanced_dir, use_sr=use_sr)
        except Exception as e:
            # Pengaman terakhir -- seharusnya tidak pernah sampai sini karena
            # process_one_image sudah menangkap error internal, tapi tetap
            # dijaga supaya SATU FILE TIDAK PERNAH MENGHENTIKAN SELURUH BATCH.
            logger.error("[%s] Error tak terduga (%s) -- file dilewati.", filename, e)
            record = {
                "original_filename": filename,
                "_status": "Error",
                "_catatan": f"Error tak terduga: {e}",
            }

        records.append(record)
        status = record.get("_status", "Error")
        logger.info("  -> status: %s", status)

    return records


def main():
    parser = argparse.ArgumentParser(
        description="Batch OCR KTP (Perspective Transform + Super Resolution + OCR) -> Excel"
    )
    parser.add_argument("--input", "-i", default=config.DEFAULT_INPUT_DIR,
                         help="Folder berisi gambar KTP input")
    parser.add_argument("--output", "-o", default=config.DEFAULT_EXCEL_OUTPUT,
                         help="Path file Excel output")
    parser.add_argument("--output-enhanced-dir", default=config.DEFAULT_OUTPUT_ENHANCED_DIR,
                         help="Folder untuk menyimpan gambar hasil enhancement")
    parser.add_argument("--ocr", choices=["easyocr", "paddleocr"], default=config.OCR_ENGINE,
                         help="Engine OCR yang dipakai")
    parser.add_argument("--no-sr", action="store_true",
                         help="Nonaktifkan Real-ESRGAN super-resolution (lebih cepat)")
    parser.add_argument("--gpu", action="store_true",
                         help="Pakai GPU (CUDA) untuk OCR & super-resolution kalau tersedia")
    args = parser.parse_args()

    config.OCR_ENGINE = args.ocr
    config.OCR_GPU = args.gpu

    setup_logging(config.LOG_FILE)

    logger.info("=" * 70)
    logger.info("BATCH OCR KTP -- mulai")
    logger.info("Input       : %s", args.input)
    logger.info("Output xlsx : %s", args.output)
    logger.info("Output img  : %s", args.output_enhanced_dir)
    logger.info("OCR engine  : %s", config.OCR_ENGINE)
    logger.info("Super-res   : %s", "AKTIF" if not args.no_sr else "NONAKTIF")
    logger.info("=" * 70)

    t0 = time.time()
    records = run_batch(
        input_dir=args.input,
        output_excel=args.output,
        output_enhanced_dir=args.output_enhanced_dir,
        use_sr=not args.no_sr,
    )

    if not records:
        logger.warning("Tidak ada data untuk ditulis ke Excel. Selesai.")
        return

    excel_writer.write_excel(records, args.output)

    total = len(records)
    valid = sum(1 for r in records if r.get("_status") == "Valid")
    review = sum(1 for r in records if r.get("_status") == "Perlu Verifikasi")
    error = sum(1 for r in records if r.get("_status") == "Error")

    elapsed = round(time.time() - t0, 1)
    logger.info("=" * 70)
    logger.info("SELESAI dalam %.1f detik.", elapsed)
    logger.info("Total     : %d", total)
    logger.info("Valid     : %d", valid)
    logger.info("Perlu Cek : %d", review)
    logger.info("Error     : %d", error)
    logger.info("Hasil Excel: %s", args.output)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
