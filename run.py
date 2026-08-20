# -*- coding: utf-8 -*-
"""
run.py
Entry point sederhana di root folder. Cukup jalankan:
    python run.py --input input_ktp --output hasil_ktp.xlsx

(Alternatifnya, dari root folder juga bisa: python -m ktp_ocr.main)
"""

from ktp_ocr.main import main

if __name__ == "__main__":
    main()
