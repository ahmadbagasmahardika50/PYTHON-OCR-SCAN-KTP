# -*- coding: utf-8 -*-
"""
ktp_parser.py
Mengubah daftar baris teks hasil OCR menjadi field-field KTP terstruktur,
dengan validasi ketat untuk NIK (harus tepat 16 digit angka).
"""

import re

from . import config

_DIGIT_FIX_MAP = str.maketrans({
    "O": "0", "o": "0",
    "I": "1", "i": "1", "l": "1", "L": "1",
    "B": "8",
    "S": "5", "s": "5",
    "Z": "2", "z": "2",
})


def _clean(text):
    return re.sub(r"\s+", " ", text or "").strip(" :;.-")


def _fix_ocr_digit_confusions(s):
    return s.translate(_DIGIT_FIX_MAP)


def extract_nik(all_text):
    """
    Cari & validasi NIK. Mengembalikan (nik_or_none, is_valid_16_digit: bool).

    Toleran terhadap kesalahan baca umum OCR (O/0, I/1, B/8, S/5) sebelum
    divalidasi -- tapi validasi akhir tetap ketat: HARUS PERSIS 16 digit.
    """
    compact = all_text.replace(" ", "")
    candidates = re.findall(config.NIK_REGEX_LOOSE, compact)

    best_partial = None
    for cand in candidates:
        fixed = _fix_ocr_digit_confusions(cand)
        digits = re.sub(r"\D", "", fixed)
        if len(digits) == config.NIK_LENGTH:
            return digits, True
        if best_partial is None and len(digits) >= 14:
            best_partial = digits

    return best_partial, False


def _extract_field(lines, label_variants):
    pattern = re.compile(
        r"(" + "|".join(re.escape(l) for l in label_variants) + r")\s*[:\-]?\s*(.*)",
        re.IGNORECASE
    )
    for i, line in enumerate(lines):
        m = pattern.search(line)
        if m:
            value = _clean(m.group(2))
            if not value and i + 1 < len(lines):
                value = _clean(lines[i + 1])
            return value
    return ""


FIELD_LABEL_MAP = {
    "Nama": ["Nama"],
    "Tempat_Tgl_Lahir": ["Tempat/Tgl Lahir", "Tempat/Tgi Lahir", "Tempat Tgl Lahir", "TempatTgl Lahir"],
    "Jenis_Kelamin": ["Jenis Kelamin"],
    "Gol_Darah": ["Gol. Darah", "Gol Darah", "GolDarah"],
    "Alamat": ["Alamat"],
    "RT_RW": ["RT/RW", "RTRW", "RT / RW"],
    "Kel_Desa": ["Kel/Desa", "Kelurahan", "Desa"],
    "Kecamatan": ["Kecamatan"],
    "Agama": ["Agama"],
    "Status_Perkawinan": ["Status Perkawinan"],
    "Pekerjaan": ["Pekerjaan"],
    "Kewarganegaraan": ["Kewarganegaraan"],
    "Berlaku_Hingga": ["Berlaku Hingga"],
    "Provinsi": ["PROVINSI"],
    "Kabupaten_Kota": ["KABUPATEN", "KOTA"],
}


def parse_ktp_lines(lines):
    """
    Parsing utama. Menerima list baris teks hasil OCR, mengembalikan dict
    field-field KTP + status validasi.

    Status yang mungkin:
        - "Valid"             : NIK 16 digit valid & nama terbaca
        - "Perlu Verifikasi"  : ada yang meragukan (NIK tidak 16 digit,
                                 nama kosong, atau banyak field kosong)
    """
    lines = [l for l in lines if l and l.strip()]
    joined = "\n".join(lines)

    nik, nik_valid = extract_nik(joined)

    result = {"NIK": nik or ""}
    for field_key, label_variants in FIELD_LABEL_MAP.items():
        result[field_key] = _extract_field(lines, label_variants)

    filled = sum(1 for v in result.values() if v)
    fill_ratio = filled / len(result)

    reasons = []
    if not nik_valid:
        reasons.append("NIK bukan 16 digit / meragukan")
    if not result["Nama"]:
        reasons.append("Nama tidak terbaca")
    if fill_ratio < 0.5:
        reasons.append("Banyak field kosong (kualitas gambar rendah)")

    result["_nik_valid"] = nik_valid
    result["_fill_ratio_pct"] = round(fill_ratio * 100, 1)
    result["_status"] = "Valid" if not reasons else "Perlu Verifikasi"
    result["_catatan"] = "; ".join(reasons)

    return result
