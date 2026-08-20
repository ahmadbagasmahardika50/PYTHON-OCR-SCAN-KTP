# -*- coding: utf-8 -*-
"""
super_resolution.py
Wrapper untuk AI Super Resolution (Real-ESRGAN) yang dipakai untuk
merekonstruksi detail gambar KTP beresolusi rendah/buram sebelum OCR.

Didesain resilient sesuai spesifikasi:
- Kalau library `realesrgan`/`torch` tidak terpasang, atau model gagal
  dimuat/download, pipeline TIDAK crash -- otomatis fallback ke upscaling
  bicubic biasa (OpenCV), dan proses tetap lanjut ke OCR.
- Model dimuat sekali (lazy singleton) supaya batch banyak gambar tidak
  reload model berulang-ulang.
"""

import os
import logging
import numpy as np
import cv2

from . import config

logger = logging.getLogger("ktp_ocr.super_resolution")

_upsampler = None          # singleton RealESRGANer, lazy-loaded
_sr_load_attempted = False
_sr_available = False


def _try_load_realesrgan():
    """
    Coba muat model Real-ESRGAN. Mengembalikan True kalau berhasil.
    Import di dalam fungsi (bukan di top-level module) supaya modul ini tetap
    bisa di-import walau `torch`/`realesrgan`/`basicsr` belum terpasang --
    baru gagal (dengan pesan jelas) saat benar-benar dipakai.
    """
    global _upsampler, _sr_available

    try:
        import torch
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
    except ImportError as e:
        logger.warning(
            "Library Real-ESRGAN belum terpasang (%s). "
            "Super-resolution akan pakai fallback bicubic upscaling biasa. "
            "Install dengan: pip install torch realesrgan basicsr", e
        )
        return False

    try:
        os.makedirs(config.SR_MODEL_DIR, exist_ok=True)

        if not os.path.exists(config.SR_MODEL_PATH):
            logger.info("Model Real-ESRGAN belum ada, mengunduh dari %s ...", config.SR_MODEL_URL)
            _download_model(config.SR_MODEL_URL, config.SR_MODEL_PATH)

        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=23, num_grow_ch=32, scale=4
        )
        _upsampler = RealESRGANer(
            scale=4,
            model_path=config.SR_MODEL_PATH,
            model=model,
            tile=200,          # proses per-tile supaya hemat memori untuk gambar besar
            tile_pad=10,
            pre_pad=0,
            half=False,        # False = lebih kompatibel di CPU-only
            gpu_id=0 if config.OCR_GPU else None,
        )
        _sr_available = True
        logger.info("Model Real-ESRGAN berhasil dimuat.")
        return True

    except Exception as e:
        logger.warning(
            "Gagal memuat/download model Real-ESRGAN (%s). "
            "Super-resolution akan pakai fallback bicubic upscaling biasa.", e
        )
        return False


def _download_model(url, dest_path):
    import urllib.request
    tmp_path = dest_path + ".part"
    urllib.request.urlretrieve(url, tmp_path)
    os.replace(tmp_path, dest_path)


def _ensure_loaded():
    global _sr_load_attempted
    if not _sr_load_attempted:
        _sr_load_attempted = True
        _try_load_realesrgan()
    return _sr_available


def _bicubic_fallback(image_bgr, scale):
    h, w = image_bgr.shape[:2]
    return cv2.resize(
        image_bgr, (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_CUBIC
    )


def super_resolve(image_bgr, scale=None, force=False):
    """
    Perbesar & perjelas gambar KTP.

    Args:
        image_bgr: gambar input (numpy array, BGR).
        scale: faktor pembesaran (default dari config, biasanya 2 atau 4).
        force: kalau True, tetap proses meski gambar sudah cukup besar.

    Returns:
        (image_hasil, method): method = "realesrgan" | "bicubic_fallback" | "skipped"
    """
    scale = scale or config.SR_SCALE
    h, w = image_bgr.shape[:2]

    if not force and max(h, w) >= config.SR_SKIP_IF_LONGEST_SIDE_ABOVE:
        logger.info("Gambar sudah cukup besar (%dx%d) -- super-resolution dilewati.", w, h)
        return image_bgr, "skipped"

    if not config.SR_ENABLED_DEFAULT:
        return _bicubic_fallback(image_bgr, scale), "bicubic_fallback"

    if _ensure_loaded():
        try:
            output, _ = _upsampler.enhance(image_bgr, outscale=scale)
            return output, "realesrgan"
        except Exception as e:
            logger.warning("Real-ESRGAN gagal saat memproses gambar (%s) -- fallback bicubic.", e)
            return _bicubic_fallback(image_bgr, scale), "bicubic_fallback"
    else:
        return _bicubic_fallback(image_bgr, scale), "bicubic_fallback"
