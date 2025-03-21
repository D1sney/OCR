import os
import re
import io
import contextlib
import pandas as pd
import pdfplumber
from dateutil.relativedelta import relativedelta
import sqlite3
import PyPDF2
import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import pytesseract
import logging
import datetime
from calendar import monthrange

import telebot
from telebot import types

from systems_list import systems_list, systems_project_required  # Предполагается, что этот модуль у вас есть
from imutils.contours import sort_contours

# Создаем (или открываем) базу для логирования действий, если требуется
conn = sqlite3.connect("bot_using.db", check_same_thread=False, isolation_level=None)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_activity (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        username TEXT,
        action TEXT,
        content TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
''')
conn.commit()

# Настройка логгера
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Вывод в консоль
        logging.FileHandler("document_processing.log", encoding="utf-8")  # Запись в файл
    ]
)
logger = logging.getLogger(__name__)

# Глобальные переменные
smeta_stoimost = None
TEMP_DIR = "temp"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

# -------------------------- ФУНКЦИИ ПРЕДОБРАБОТКИ И РАСПОЗНАВАНИЯ --------------------------

def handwritten(image):
    logger.debug("Начало обработки изображения для рукописного текста")
    block_size = 9
    constant = 2
    blur = cv2.GaussianBlur(image, (7, 7), 0)
    fnoise = cv2.medianBlur(blur, 3)
    th1 = cv2.adaptiveThreshold(fnoise, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, constant)
    th2 = cv2.adaptiveThreshold(th1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, constant)
    blu = cv2.GaussianBlur(th2, (5, 5), 0)
    fnois = cv2.medianBlur(th2, 3)
    logger.debug("Обработка изображения завершена")
    return blu, fnois

def preprocess_image(image):
    logger.debug("Предобработка изображения")
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    logger.debug("Предобработка завершена")
    return Image.fromarray(thresh)

def correct_rotation(image):
    logger.debug("Коррекция поворота изображения")
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
    if lines is not None:
        angles = [np.degrees(np.arctan2(y2 - y1, x2 - x1)) for x1, y1, x2, y2 in [line[0] for line in lines]]
        median_angle = np.median(angles)
        if abs(median_angle) > 45:
            median_angle -= 90
        logger.debug(f"Поворот изображения на угол: {-median_angle}")
        return image.rotate(-median_angle, expand=True)
    logger.debug("Поворот не требуется")
    return image

def extract_text_from_pdf(pdf_path, skip_top_fraction=None):
    logger.info(f"Извлечение текста из PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    page = doc.load_page(0)
    rect = page.rect
    clip_rect = (fitz.Rect(rect.x0, rect.y0 + rect.height * skip_top_fraction, rect.x1, rect.y1)
                 if skip_top_fraction else rect)
    pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), clip=clip_rect)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    corrected = correct_rotation(img)
    preprocessed = preprocess_image(corrected)
    config_custom = r'--oem 3 -l rus --psm 3'
    text = pytesseract.image_to_string(preprocessed, lang="rus", config=config_custom)
    cleaned_text = re.sub(r'\s+', ' ', text)
    logger.debug(f"Извлеченный текст: {cleaned_text[:100]}...")
    return cleaned_text

def extract_text_from_pdf_all(pdf_path):
    logger.info(f"Извлечение полного текста из PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        corrected = correct_rotation(img)
        preprocessed = preprocess_image(corrected)
        config_custom = r'--oem 3 -l rus --psm 3'
        text = pytesseract.image_to_string(preprocessed, lang="rus", config=config_custom)
        cleaned_text = re.sub(r'\s+', ' ', text)
        full_text += " " + cleaned_text
        with open("output.txt", "a", encoding="utf-8") as f:
            f.write(text)
    logger.debug(f"Полный извлеченный текст: {full_text[:100]}...")
    return full_text.strip()

def extract_system(text, systems_list):
    logger.debug("Извлечение системы из текста")
    def remove_remont_endings(s):
        return re.sub(r'\bремонт\w*\b', 'ремонт', s, flags=re.IGNORECASE)

    processed_text = remove_remont_endings(text)
    for system in systems_list:
        processed_system = remove_remont_endings(system)
        pattern = r'\s*'.join(re.escape(word) for word in processed_system.split())
        if processed_system.lower() in processed_text.lower():
            logger.info(f"Найдена система: {system}")
            return system
    logger.warning("Система не найдена в тексте")
    return "Описание системы не найдено"

def clean_text(text):
    logger.debug("Очистка текста")
    text = re.sub(r'[^\w\s.,:-]', '', text)
    cleaned = re.sub(r'\s+', ' ', text).strip()
    logger.debug(f"Очищенный текст: {cleaned[:100]}...")
    return cleaned

def extract_address(text):
    logger.debug("Извлечение адреса из текста")
    text = clean_text(text)
    text = ' '.join(text.split())
    text = re.sub(r'(\S)\s*(по адресу:)', r'\1 \2', text, flags=re.IGNORECASE)
    address_marker = "по адресу:"
    address_pattern = (
        r'(?:(?:[А-ЯЁа-яё]+\s+область)[,]?\s*)?'
        r'(?:'
        r'(?:г|д|тер|ул|пл|мкр|с|п|пр-кт|проезд)[\.,]?\s*[А-ЯЁа-яё0-9\-\s]+'
        r'(?:[0-9]+[а-яА-Яa-zA-Z0-9\/\-]*)?'
        r'(?:\s*,\s*|\s+)'
        r')+'
        r'д[.,]?\s*\d+[а-яА-Яa-zA-Z0-9\/\-]*\b'
    )
    address = "Адрес не найден"
    marker_match = re.search(re.escape(address_marker), text, re.IGNORECASE)
    if marker_match:
        substring = text[marker_match.end():]
        addr_match = re.search(address_pattern, substring, re.IGNORECASE)
        if addr_match:
            address = addr_match.group().strip()
    else:
        addr_match = re.search(address_pattern, text, re.IGNORECASE)
        if addr_match:
            address = addr_match.group().strip()
    address = re.sub(
        r'^(?:ДОКУМЕНТ\s+ПОДПИСАН\s+ЭЛЕКТРОННОЙ\s+ПОДПИСЬЮ\s+)?(?:[А-ЯЁа-яё]+\s+область[,]?\s*)',
        '', address, flags=re.IGNORECASE
    )
    address = re.split(r'(?:Назначение|Месторасположение)', address, flags=re.IGNORECASE)[0].strip()
    m = re.search(r'(\bг\.?\s+\S.*)', address, re.IGNORECASE)
    if m:
        address = m.group(1).strip()
    logger.info(f"Извлеченный адрес: {address}")
    return address

def extract_doc_date_number(text):
    logger.debug("Извлечение номера и даты документа")
    pattern = r'№\s*[\d/-]+[\s_]*от\s*[«"]?\s*\d{1,2}[»"]?\s*(?:[а-яА-Я]+|\d{1,2})\s*\d{4}\s*г[.,]?'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        doc_date = match.group().strip().rstrip(' .,')
        logger.info(f"Найден номер и дата: {doc_date}")
        return doc_date
    logger.warning("Номер и дата не найдены")
    return "Номер и дата не найдены"

def clean_address(address):
    logger.debug(f"Очистка адреса: {address}")
    cleaned = re.sub(r'^(?:ДОКУМЕНТ PОДПИСАН ЭЛЕКТРОННОЙ PОДПИСЬЮ\s+)?(?:[А-ЯЁа-яё]+\s+область,\s*)', '', address,
                     flags=re.IGNORECASE).strip()
    logger.debug(f"Очищенный адрес: {cleaned}")
    return cleaned

def extract_table(pdf_path, page_num, table_num):
    logger.info(f"Извлечение таблицы из PDF: {pdf_path}, страница {page_num}, таблица {table_num}")
    pdf = pdfplumber.open(pdf_path)
    table_page = pdf.pages[page_num]
    table = table_page.extract_tables()[table_num]
    logger.debug(f"Извлечена таблица: {table[:2]}...")
    return table

def table_converter(table):
    logger.debug("Конвертация таблицы в строку")
    table_string = ''
    for row_num in range(len(table)):
        row = table[row_num]
        cleaned_row = [
            item.replace('\n', ' ') if item is not None and '\n' in item else 'None' if item is None else item for item
            in row]
        table_string += ('|' + '|'.join(cleaned_row) + '|' + '\n')
    table_string = table_string[:-1]
    logger.debug(f"Сконвертированная таблица: {table_string[:100]}...")
    return table_string

def extract_sro_date(text):
    logger.debug("Извлечение даты СРО")
    patterns = [
        r'(\d{1,2}[.]\d{1,2}[.]\d{4})',  # "dd.mm.yyyy"
        r'(\d{1,2}[-]\d{1,2}[-]\d{4})'   # "dd-mm-yyyy"
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            data_str = match.group(1)
            try:
                data_str = data_str.replace("-", ".")
                date_obj = datetime.datetime.strptime(data_str, "%d.%m.%Y").date()
                logger.info(f"Найдена дата СРО: {date_obj}")
                return date_obj
            except ValueError:
                logger.warning(f"Ошибка формата даты: {data_str}")
                return None
    logger.warning("Дата СРО не найдена")
    return None

def contract_preprocess_image(image):
    logger.debug("Предобработка изображения договора")
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    logger.debug("Предобработка завершена")
    return Image.fromarray(thresh)

def contract_correct_rotation(image):
    logger.debug("Коррекция поворота изображения договора")
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
    if lines is not None:
        angles = [np.degrees(np.arctan2(y2 - y1, x2 - x1))
                  for x1, y1, x2, y2 in [line[0] for line in lines]]
        median_angle = np.median(angles)
        if abs(median_angle) > 45:
            median_angle -= 90
        logger.debug(f"Поворот изображения на угол: {-median_angle}")
        return image.rotate(-median_angle, expand=True)
    logger.debug("Поворот не требуется")
    return image

def build_address_regex(input_address, max_gap=350):
    logger.debug(f"Построение регулярного выражения для адреса: {input_address}")
    parts = re.split(r'\s*,\s*', input_address.strip())
    token_list = []
    for part in parts:
        tokens = re.findall(r'[а-яА-Я0-9-]+', part)
        token_list.extend(tokens)
    regex_parts = [re.escape(token_list[0])]
    for token in token_list[1:]:
        separator = r'(?:[.,]?\s+)'
        gap = r'[\s\S]{0,' + str(max_gap) + r'}?'
        regex_parts.append(separator + gap + re.escape(token))
    pattern = ''.join(regex_parts)
    logger.debug(f"Сформированное выражение: {pattern[:100]}...")
    return pattern, token_list

def find_address_in_text(ocr_text, input_address, max_gap=350):
    logger.debug("Поиск адреса в тексте")
    pattern, tokens = build_address_regex(input_address, max_gap)
    normalized_text = re.sub(r'\s+', ' ', ocr_text.replace('\n', ' '))
    matches = list(re.finditer(pattern, normalized_text, flags=re.IGNORECASE))
    if matches:
        best_match = min(matches, key=lambda m: len(m.group(0)))
        logger.info(f"Найден адрес: {best_match.group(0)}")
        return best_match.group(0), tokens
    logger.warning("Адрес не найден в тексте")
    return None, tokens

def clean_contract_address(found_text, tokens, input_address):
    logger.debug("Очистка адреса из договора")
    cleaned = ""
    current_pos = 0
    prev_end = 0
    for token in tokens:
        match = re.search(re.escape(token), found_text[current_pos:], flags=re.IGNORECASE)
        if match:
            start = current_pos + match.start()
            end = current_pos + match.end()
            if end < len(found_text) and found_text[end] == ',':
                end += 1
            if not cleaned:
                cleaned += found_text[start:end]
            else:
                inter = found_text[prev_end:start]
                if re.fullmatch(r'[\s,.\-:;]*', inter):
                    cleaned += inter + found_text[start:end]
                else:
                    cleaned += " " + found_text[start:end]
            prev_end = end
            current_pos = end
        else:
            logger.debug(f"Использован исходный адрес: {input_address}")
            return input_address
    logger.debug(f"Очищенный адрес: {cleaned}")
    return cleaned.strip()

def add_dot_if_missing(address):
    logger.debug(f"Добавление точки в адрес: {address}")
    fixed = re.sub(r'\bд(?!\.)\s*(\d+)', r'д.\1', address, flags=re.IGNORECASE)
    logger.debug(f"Исправленный адрес: {fixed}")
    return fixed

def add_space_after_house(address):
    logger.debug(f"Добавление пробела после дома: {address}")
    fixed = re.sub(r'\b(д\.)\s*(\d)', r'\1 \2', address, flags=re.IGNORECASE)
    logger.debug(f"Исправленный адрес: {fixed}")
    return fixed

def check_contract_address(contract_file, input_address):
    logger.info(f"Проверка адреса в договоре: {contract_file}")
    try:
        pdf_document = fitz.open(contract_file)
        extracted_text = ""
        for page in pdf_document:
            scale = 4
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            resized_img = img.resize((int(img.width * 1.5), int(img.height * 1.5)), Image.LANCZOS)
            corrected_img = contract_correct_rotation(resized_img)
            enhancer = ImageEnhance.Contrast(corrected_img)
            contrast_img = enhancer.enhance(2.0)
            preprocessed_img = contract_preprocess_image(contrast_img)
            config_custom = r'--oem 3 -l rus --psm 3'
            page_text = pytesseract.image_to_string(preprocessed_img, lang="rus", config=config_custom)
            extracted_text += page_text + "\n"
        pattern, tokens = build_address_regex(input_address)
        found_address, tokens = find_address_in_text(extracted_text, input_address)
        if found_address:
            cleaned_address = clean_contract_address(found_address, tokens, input_address)
            cleaned_address = add_dot_if_missing(cleaned_address)
            cleaned_address = add_space_after_house(cleaned_address)
            logger.info(f"Извлеченный адрес из договора: {cleaned_address}")
            return cleaned_address
        else:
            logger.warning("Адрес в договоре не найден")
            return None
    except Exception as e:
        logger.error(f"Ошибка при обработке договора: {e}")
        return None

def parse_price_level(date_str):
    logger.debug(f"Парсинг уровня цен: {date_str}")
    quarter_pattern = r'([IVX]+)\s*(?:квартал|кв\.?)\s+(\d{4})'
    m = re.search(quarter_pattern, date_str, re.IGNORECASE)
    if m:
        roman = m.group(1).upper()
        year = int(m.group(2))
        roman_to_int = {"I": 1, "II": 2, "III": 3, "IV": 4}
        quarter = roman_to_int.get(roman, 4)
        date_obj = {1: datetime.date(year, 3, 31),
                    2: datetime.date(year, 6, 30),
                    3: datetime.date(year, 9, 30),
                    4: datetime.date(year, 12, 31)}.get(quarter)
        logger.info(f"Извлечен уровень цен: {quarter} квартал {year}")
        return date_obj

    numeric_quarter_pattern = r'(\d+)\s*(?:квартал|кв\.?)\s+(\d{4})'
    m_num = re.search(numeric_quarter_pattern, date_str, re.IGNORECASE)
    if m_num:
        quarter = int(m_num.group(1))
        if quarter > 4:
            quarter = 4
        year = int(m_num.group(2))
        date_obj = {1: datetime.date(year, 3, 31),
                    2: datetime.date(year, 6, 30),
                    3: datetime.date(year, 9, 30),
                    4: datetime.date(year, 12, 31)}.get(quarter)
        logger.info(f"Извлечен уровень цен: {quarter} квартал {year}")
        return date_obj

    month_year_pattern = r'([а-яА-Я]+)\s+(\d{4})\s*г'
    m3 = re.search(month_year_pattern, date_str, re.IGNORECASE)
    if m3:
        month_name = m3.group(1).lower()
        year = int(m3.group(2))
        months = {"январь": 1, "февраль": 2, "март": 3, "апрель": 4, "май": 5, "июнь": 6,
                  "июль": 7, "август": 8, "сентябрь": 9, "октябрь": 10, "ноябрь": 11, "декабрь": 12,
                  "января": 1, "февраля": 2, "марта": 3, "апреля": 4, "мая": 5, "июня": 6,
                  "июля": 7, "августа": 8, "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12}
        month = months.get(month_name)
        if month:
            day = monthrange(year, month)[1]
            date_obj = datetime.date(year, month, day)
            logger.info(f"Извлечена дата (месяц и год): {date_obj}")
            return date_obj

    date_pattern = r'(\d{1,2})\s+([а-яА-Я]+)\s+(\d{4})'
    m2 = re.search(date_pattern, date_str)
    if m2:
        day = int(m2.group(1))
        month_name = m2.group(2).lower()
        year = int(m2.group(3))
        months = {"января": 1, "февраля": 2, "марта": 3, "апреля": 4, "мая": 5, "июня": 6,
                  "июля": 7, "августа": 8, "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12}
        month = months.get(month_name)
        if month:
            try:
                date_obj = datetime.date(year, month, day)
                logger.info(f"Извлечена дата: {date_obj}")
                return date_obj
            except ValueError:
                logger.warning(f"Неверный формат даты: {date_str}")
                return None

    logger.warning("Уровень цен не распознан")
    return None

def check_smeta(smeta_file, reference_address, statement_system):
    logger.info(f"Проверка сметы: {smeta_file}")
    errors = []
    global smeta_stoimost
    result_lines = ["<h3><b>Смета</b></h3>"]

    df = pd.read_excel(smeta_file, header=None)
    df_address_system = df.iloc[10:50]
    all_text = ' '.join([
        str(x) for x in df_address_system.values.flatten()
        if pd.notna(x) and str(x).strip().lower() != "nan"
    ])
    address_pattern = (
        r'(?:(?:[А-ЯЁа-яё]+\s+область)[,]?\s*)?'
        r'(?:'
        r'(?:г|д|тер|ул|пл|мкр|с|п|пр-кт|проезд)[\.,]?\s*[А-ЯЁа-яё0-9\-\s]+'
        r'(?:[0-9]+[а-яА-Яa-zA-Z0-9\/\-]*)?'
        r'(?:\s*,\s*|\s+)'
        r')+'
        r'д[.,]?\s*\d+[а-яА-Яa-zA-Z0-9\/\-]*'
    )
    addresses = re.findall(address_pattern, all_text, re.IGNORECASE)
    if addresses:
        smeta_address = addresses[0].strip()
        result_lines.append(f"<u>Адрес:</u> {smeta_address}<br>")
        logger.info(f"Извлечен адрес сметы: {smeta_address}")
    else:
        smeta_address = "Адрес не найден"
        result_lines.append("<u>Адрес:</u> Адрес не найден<br>")
        logger.warning("Адрес сметы не найден")

    smeta_system = None
    for sys in systems_list:
        if sys.lower() in all_text.lower():
            smeta_system = sys
            break
    if smeta_system is None:
        smeta_system = "Описание системы не найдено"
        result_lines.append("<u>Система:</u> Описание системы не найдено<br>")
        logger.warning("Система сметы не найдена")
    else:
        result_lines.append(f"<u>Система:</u> {smeta_system}<br>")
        logger.info(f"Извлечена система сметы: {smeta_system}")

    smeta_address_clean = clean_address(smeta_address)
    reference_address_clean = clean_address(reference_address)
    if reference_address and smeta_address_clean.lower() != reference_address_clean.lower():
        result_lines.append("<u>Ошибка:</u> Адрес сметы не совпадает с эталонным адресом!<br>")
        result_lines.append(f"Смета: {smeta_address_clean}<br>")
        result_lines.append(f"Эталон: {reference_address_clean}<br>")
        errors.append("Адрес сметы не совпадает с эталонным адресом.")
        logger.warning(f"Адрес сметы не совпадает: {smeta_address_clean} != {reference_address_clean}")
    else:
        result_lines.append("<i>Адрес сметы совпадает с эталонным адресом.</i><br>")
        logger.info("Адрес сметы совпадает с эталонным")

    smetnaya_stoimost_match = re.search(r'Сметная стоимость\s+([\d.,]+)\s*\(?[\d.,]*\)?\s*тыс\.руб', all_text, re.IGNORECASE)
    if smetnaya_stoimost_match:
        smeta_stoimost = smetnaya_stoimost_match.group(1)
        result_lines.append(f"<u>Сметная стоимость:</u> {smeta_stoimost} тыс. руб.<br>")
        logger.info(f"Извлечена сметная стоимость: {smeta_stoimost}")
    else:
        smeta_stoimost = None
        result_lines.append("<u>Ошибка:</u> Сметная стоимость не найдена!<br>")
        errors.append("Сметная стоимость не найдена!")
        logger.warning("Сметная стоимость не найдена")

    osnovanie_match = re.search(r'\bОснование\b\s*[:\-]?\s*([\w\s]+)', all_text, re.IGNORECASE)
    if osnovanie_match:
        osnovanie_value = osnovanie_match.group(1).strip()
        if osnovanie_value:
            result_lines.append(f"<u>Основание:</u> {osnovanie_value}<br>")
            logger.info(f"Извлечено основание: {osnovanie_value}")
        else:
            result_lines.append("<u>Ошибка:</u> Основание не найдено!<br>")
            errors.append("Основание не найдено!")
            logger.warning("Основание пустое")
    else:
        result_lines.append("<u>Ошибка:</u> Основание не найдено!<br>")
        errors.append("Основание не найдено!")
        logger.warning("Основание не найдено")

    all_text_full = ' '.join([
        str(x) for x in df.values.flatten()
        if pd.notna(x) and str(x).strip().lower() != "nan"
    ])
    clean_text_for_nds = re.sub(r'\s+', ' ', all_text_full).strip()
    clean_text_for_nds = re.sub(r'[^\w\s,%.]', '', clean_text_for_nds)
    nds_found = None
    for match in re.finditer(r'НДС[^\d]+(\d{1,3}(?:[.,]\d{2})?)', clean_text_for_nds, re.IGNORECASE):
        value_str = match.group(1).strip()
        try:
            value = float(value_str.replace(',', '.'))
            if value == 20.0:
                nds_found = value
                break
        except ValueError:
            continue

    if nds_found is not None:
        result_lines.append(f"<u>НДС:</u> {nds_found} %<br>")
        logger.info(f"НДС найден: {nds_found} %")
    else:
        result_lines.append("<u>Ошибка:</u> НДС не найден!<br>")
        errors.append("НДС не найден!")
        logger.warning("НДС не найден")

    price_pattern = r'текущем(?:\s*\([^)]*\))?\s+уровне\s+цен\s*[:]?[\s]*([^\s].+?)(?=\s{2,}|$)'
    m_price = re.search(price_pattern, all_text, re.IGNORECASE)
    if m_price:
        price_level_str = m_price.group(1).strip()
        price_level_str = re.sub(r'\s*Сметная стоимость.*$', '', price_level_str, flags=re.IGNORECASE).strip()
        price_date = parse_price_level(price_level_str)
        if price_date:
            today = datetime.date.today()
            quarter = (price_date.month - 1) // 3 + 1
            price_level_display = f"{quarter} квартал {price_date.year} г."
            if price_date >= today - relativedelta(months=3):
                result_lines.append("<i>Уровень цен актуален (не старше 3 месяцев).</i><br>")
                logger.info("Уровень цен актуален")
            else:
                result_lines.append("<u>Ошибка:</u> Уровень цен устарел!<br>")
                errors.append("Уровень цен устарел!")
                logger.warning("Уровень цен устарел")
            result_lines.append(f"<u>Дата уровня цен:</u> {price_level_display}<br>")
            logger.info(f"Извлечена дата уровня цен: {price_level_display}")
        else:
            result_lines.append("<u>Ошибка:</u> Не удалось распознать дату уровня цен в смете.<br>")
            errors.append("Не удалось распознать уровень цен в смете.")
            logger.error("Не удалось распознать дату уровня цен")
    else:
        result_lines.append("<u>Ошибка:</u> Уровень цен не найден в смете.<br>")
        errors.append("Уровень цен не найден в смете.")
        logger.warning("Уровень цен не найден")

    html_output = "".join(result_lines)
    print(html_output)
    if errors:
        logger.warning(f"Обнаружены ошибки в смете: {errors}")
    else:
        logger.info("Смета проверена без ошибок")
    return errors

def extract_price_level_explanatory(text):
    logger.debug("Извлечение уровня цен из пояснительной записки")
    text = re.sub(r'(\S)\s*(квартал)', r'\1 \2', text, flags=re.IGNORECASE)
    pattern1 = r'текущих\s+ценах\s+на\s+(\d+|[IVXУШ]+)\s*(?:квартал|кв\.?)\s+(\d{4})(?:\s+года)?'
    m = re.search(pattern1, text, re.IGNORECASE)
    if not m:
        pattern2 = r'(\d+|[IVXУШ]+)\s*(?:квартал|кв\.?)\s+(\d{4})\s*(?:г|г\.)?'
        m = re.search(pattern2, text, re.IGNORECASE)
    if m:
        quarter_val = m.group(1).upper()
        year = int(m.group(2))
        try:
            quarter = int(quarter_val)
        except ValueError:
            if quarter_val == "Ш":
                quarter = 3
            else:
                roman_to_int = {"I": 1, "II": 2, "III": 3, "IV": 4}
                quarter = roman_to_int.get(quarter_val, None)
        if quarter is None or quarter > 4:
            quarter = 4
        quarter_date = {1: datetime.date(year, 3, 31),
                        2: datetime.date(year, 6, 30),
                        3: datetime.date(year, 9, 30),
                        4: datetime.date(year, 12, 31)}.get(quarter)
        quarter_str = f"{quarter} квартал {year} г."
        logger.info(f"Извлечен уровень цен: {quarter_str}")
        return (quarter_date, quarter_str)
    logger.warning("Уровень цен не распознан")
    return (None, None)

def normalize_system(s):
    logger.debug(f"Нормализация системы: {s}")
    normalized = re.sub(r'\s+', ' ', s.strip().lower())
    logger.debug(f"Нормализованная система: {normalized}")
    return normalized

def normalize_address_for_comparison(address):
    logger.debug(f"Нормализация адреса для сравнения: {address}")
    normalized = address.lower()
    normalized = re.sub(r'[,.]', '', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    logger.debug(f"Нормализованный адрес: {normalized}")
    return normalized

def classify_document(file_path):
    logger.info(f"Классификация документа: {file_path}")
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".xlsx":
        logger.info("Документ классифицирован как Смета (по расширению)")
        return "Смета"
    text = ""
    if ext == ".pdf":
        try:
            text = extract_text_from_pdf(file_path)
        except Exception as e:
            logger.error(f"Ошибка при извлечении текста для классификации: {e}")
            text = ""
    text_low = text.lower()

    if "пояснительная записка" in text_low:
        logger.info("Документ классифицирован как: Пояснительная записка (приоритетная проверка)")
        return "Пояснительная записка"

    norm_text = re.sub(r'\s+', '', text_low)
    keywords = {
        "Дефектная ведомость": ["дефектная ведомость", "лефектная ведомость", "ведомость"],
        "Акт обследования": ["акт обследования"],
        "Пояснительная записка": ["пояснительная записка"],
        "Смета": ["локальный сметный расчет", "сметный расчет"],
        "НОПРИЗ": ["уведомление"],
        "Проект": ["проектная документация"],
        "Заключение": ["Номер заключения"],
        "Заявление": ["экспертно-консультационные услуги", "заявление", "консультационные"],
        "Техническое задание": ["техническое задание", "адание на проектирование капитального ремонта"],
        "Информационно-удостоверяющий лист": ["уил", "информационно-удостоверяющий лист"],
        "Выписка СРО": ["выписка"],
        "Договор": ["договор о проведении"]
    }
    first_occurrence = None
    doc_type_found = None
    for doc_type, phrases in keywords.items():
        for phrase in phrases:
            index = text_low.find(phrase)
            if index == -1:
                norm_phrase = re.sub(r'\s+', '', phrase)
                index = norm_text.find(norm_phrase)
            if index != -1:
                if first_occurrence is None or index < first_occurrence:
                    first_occurrence = index
                    doc_type_found = doc_type
    if doc_type_found:
        logger.info(f"Документ классифицирован как: {doc_type_found}")
    else:
        logger.warning("Тип документа не определен")
    return doc_type_found if doc_type_found else "Неизвестный документ"

def check_approval_sheet_in_document(pdf_path):
    logger.debug(f"Проверка листа согласования в документе: {pdf_path}")
    full_text = extract_text_from_pdf_all(pdf_path)
    normalized_text = re.sub(r'\s+', ' ', full_text).lower()
    found = "лист согласования" in normalized_text
    logger.debug(f"Лист согласования {'найден' if found else 'не найден'}")
    return found, normalized_text[:500]

def extract_first_four_digits(cost_str):
    logger.debug(f"Извлечение первых четырех цифр из стоимости: {cost_str}")
    digits = ''.join(re.findall(r'\d', cost_str))
    result = digits[:4]
    logger.debug(f"Извлечены цифры: {result}")
    return result

# -------------------------- ОСНОВНАЯ ФУНКЦИЯ ОБРАБОТКИ ДОКУМЕНТОВ --------------------------

def process_documents(file_paths):
    """
    Функция принимает список путей к файлам, определяет тип каждого документа,
    выполняет все проверки и возвращает итоговый текстовый отчёт.
    """
    logger.info("Начало обработки документов")
    logger.debug(f"Переданные пути файлов: {file_paths}")

    docs = {}
    for file_path in file_paths:
        doc_type = classify_document(file_path)
        docs[doc_type] = file_path
        logger.info(f"Файл получен: {os.path.basename(file_path)}, тип: {doc_type}")

    result_lines = []
    final_errors = []

    # Заявление
    statement_address = None
    statement_system = None
    if "Заявление" in docs:
        result_lines.append("<h3><b>Заявление</b></h3>")
        statement_file = docs["Заявление"]
        st_text = extract_text_from_pdf(statement_file, skip_top_fraction=1/6)
        statement_system = extract_system(st_text, systems_list)
        statement_address = extract_address(st_text)
        result_lines.append(f"<u>Система:</u> {statement_system}<br>")
        result_lines.append(f"<u>Адрес:</u> {statement_address}<br>")
        logger.info(f"Извлечены данные из Заявления: система={statement_system}, адрес={statement_address}")
        statement_full_text = extract_text_from_pdf_all(statement_file)
        if "реестр смет" in statement_full_text.lower():
            result_lines.append("<i>В заявлении обнаружен 'Реестр смет': адрес и система совпадают с данными дефектной ведомости.</i><br>")
            logger.info("Обнаружен 'Реестр смет' в Заявлении")
    else:
        result_lines.append("<h3><b>Заявление</b></h3>")
        result_lines.append("<u>Ошибка:</u> Заявление не найдено.<br>")
        final_errors.append("Заявление отсутствует.")
        logger.warning("Заявление не найдено")

    # Договор
    contract_address = None
    if "Договор" in docs:
        result_lines.append("<h3><b>Договор</b></h3>")
        dog_file = docs["Договор"]
        contract_address = check_contract_address(dog_file, statement_address if statement_address else "")
        if contract_address:
            result_lines.append(f"<u>Извлечённый адрес:</u> {contract_address}<br>")
            logger.info(f"Извлечен адрес из договора: {contract_address}")
        else:
            result_lines.append("<i>Адрес в договоре не найден.</i><br>")
            logger.warning("Адрес в договоре не найден")

    # Дефектная ведомость
    defect_address = None
    defect_system = None
    if "Дефектная ведомость" in docs:
        result_lines.append("<h3><b>Дефектная ведомость</b></h3>")
        defect_file = docs["Дефектная ведомость"]
        df_text = extract_text_from_pdf(defect_file)
        defect_system = extract_system(df_text, systems_list)
        defect_address = extract_address(df_text)
        df_doc = extract_doc_date_number(df_text)
        result_lines.append(f"<u>Система:</u> {defect_system}<br>")
        result_lines.append(f"<u>Адрес:</u> {defect_address}<br>")
        logger.info(f"Извлечены данные из Дефектной ведомости: система={defect_system}, адрес={defect_address}")
        if df_doc == "Номер и дата не найдены":
            result_lines.append("<u>Ошибка:</u> Не удалось корректно выгрузить дату и номер. Будет направлена на ручную проверку.<br>")
            final_errors.append("Не удалось определить номер и дату дефектной ведомости.")
            logger.error("Не удалось извлечь номер и дату из Дефектной ведомости")
        else:
            result_lines.append(f"<u>Номер и дата:</u> {df_doc}<br>")
            logger.debug(f"Извлечены номер и дата: {df_doc}")
        found, preview = check_approval_sheet_in_document(defect_file)
        if found:
            result_lines.append("<i>Лист согласования присутствует.</i><br>")
            logger.info("Лист согласования найден")
        else:
            result_lines.append("<u>Ошибка:</u> Лист согласования не найден.<br>")
            final_errors.append("Лист согласования отсутствует в дефектной ведомости.")
            logger.warning("Лист согласования отсутствует")
    else:
        result_lines.append("<h3><b>Дефектная ведомость</b></h3>")
        result_lines.append("<u>Ошибка:</u> Дефектная ведомость не найдена.<br>")
        final_errors.append("Дефектная ведомость отсутствует.")
        logger.warning("Дефектная ведомость не найдена")

    if contract_address:
        reference_address = contract_address
        logger.info(f"Эталонный адрес взят из договора: {reference_address}")
    elif defect_address:
        reference_address = defect_address
        logger.info(f"Эталонный адрес взят из дефектной ведомости: {reference_address}")
    elif statement_address:
        reference_address = statement_address
        logger.info(f"Эталонный адрес взят из заявления: {reference_address}")
    else:
        reference_address = None
        final_errors.append("Не удалось определить эталонный адрес: отсутствуют договор, дефектная ведомость и заявление.")
        logger.error("Эталонный адрес не определен")

    reference_system = statement_system if statement_system else None
    logger.debug(f"Эталонная система: {reference_system}")

    if statement_address and reference_address and normalize_address_for_comparison(statement_address) != normalize_address_for_comparison(reference_address):
        result_lines.append("<u>Ошибка:</u> Адрес заявления не совпадает с эталонным адресом:<br>")
        result_lines.append(f"Заявление: {statement_address}<br>")
        result_lines.append(f"Эталон: {reference_address}<br>")
        final_errors.append("Адрес заявления не совпадает с эталонным адресом.")
        logger.warning(f"Адрес заявления не совпадает: {statement_address} != {reference_address}")

    if defect_address and reference_address and normalize_address_for_comparison(defect_address) != normalize_address_for_comparison(reference_address):
        result_lines.append("<u>Ошибка:</u> Адрес дефектной ведомости не совпадает с эталонным адресом:<br>")
        result_lines.append(f"Дефектная ведомость: {defect_address}<br>")
        result_lines.append(f"Эталон: {reference_address}<br>")
        final_errors.append("Адрес дефектной ведомости не совпадает с эталонным адресом.")
        logger.warning(f"Адрес дефектной ведомости не совпадает: {defect_address} != {reference_address}")

    if "Смета" in docs:
        result_lines.append("<h3><b>Смета</b></h3>")
        smeta_file = docs["Смета"]
        smeta_output = io.StringIO()
        with contextlib.redirect_stdout(smeta_output):
            smeta_errors = check_smeta(smeta_file, reference_address, reference_system)
        result_lines.append(smeta_output.getvalue().replace('\n', '<br>'))
        if smeta_errors:
            final_errors.extend(smeta_errors)
            logger.warning(f"Ошибки в смете: {smeta_errors}")
    else:
        result_lines.append("<h3><b>Смета</b></h3>")
        result_lines.append("<u>Ошибка:</u> Смета не найдена.<br>")
        final_errors.append("Смета отсутствует.")
        logger.warning("Смета не найдена")

    if "Акт обследования" in docs:
        result_lines.append("<h3><b>Акт обследования</b></h3>")
        act_file = docs["Акт обследования"]
        act_text = extract_text_from_pdf(act_file)
        act_system = extract_system(act_text, systems_list)
        act_address = extract_address(act_text)
        act_doc = extract_doc_date_number(act_text)
        if act_address == "Адрес не найден" and reference_address:
            act_address = reference_address
            logger.debug(f"Адрес акта обследования взят из эталонного: {act_address}")
        result_lines.append(f"<u>Система:</u> {act_system}<br>")
        result_lines.append(f"<u>Адрес:</u> {act_address}<br>")
        logger.info(f"Извлечены данные из Акта: система={act_system}, адрес={act_address}")
        if act_doc == "Номер и дата не найдены":
            result_lines.append("<u>Ошибка:</u> Не удалось корректно выгрузить дату и номер. Будет направлена на ручную проверку.<br>")
            final_errors.append("Не удалось определить номер и дату акта обследования.")
            logger.error("Не удалось извлечь номер и дату из Акта")
        else:
            result_lines.append(f"<u>Номер и дата:</u> {act_doc}<br>")
            logger.debug(f"Извлечены номер и дата: {act_doc}")
        found, preview = check_approval_sheet_in_document(act_file)
        if found:
            result_lines.append("<i>Лист согласования присутствует.</i><br>")
            logger.info("Лист согласования найден в Акте")
        else:
            result_lines.append("<u>Ошибка:</u> Лист согласования не найден.<br>")
            final_errors.append("Лист согласования отсутствует в акте обследования.")
            logger.warning("Лист согласования отсутствует в Акте")
        if reference_address and normalize_address_for_comparison(act_address) != normalize_address_for_comparison(reference_address):
            result_lines.append("<u>Ошибка:</u> Адрес акта обследования не совпадает с эталонным адресом:<br>")
            result_lines.append(f"Акт: {act_address}<br>")
            result_lines.append(f"Эталон: {reference_address}<br>")
            final_errors.append("Адрес акта обследования не совпадает с эталонным адресом.")
            logger.warning(f"Адрес акта не совпадает: {act_address} != {reference_address}")
        if reference_system and act_system.lower() != reference_system.lower():
            result_lines.append("<u>Ошибка:</u> Система акта обследования не совпадает с эталонной системой:<br>")
            result_lines.append(f"Акт: {act_system}<br>")
            result_lines.append(f"Эталон: {reference_system}<br>")
            final_errors.append("Система акта обследования не совпадает с эталонной системой.")
            logger.warning(f"Система акта не совпадает: {act_system} != {reference_system}")
    else:
        result_lines.append("<h3><b>Акт обследования</b></h3>")
        result_lines.append("<u>Ошибка:</u> Акт обследования не найден.<br>")
        final_errors.append("Акт обследования отсутствует.")
        logger.warning("Акт обследования не найден")

    if "Техническое задание" in docs:
        result_lines.append("<h3><b>Техническое задание</b></h3>")
        tech_file = docs["Техническое задание"]
        tech_text = extract_text_from_pdf(tech_file)
        tech_system = extract_system(tech_text, systems_list)
        tech_address = extract_address(tech_text)
        tech_doc = extract_doc_date_number(tech_text)
        if tech_address == "Адрес не найден" and reference_address:
            tech_address = reference_address
            logger.debug(f"Адрес ТЗ взят из эталонного: {tech_address}")
        result_lines.append(f"<u>Система:</u> {tech_system}<br>")
        result_lines.append(f"<u>Адрес:</u> {tech_address}<br>")
        logger.info(f"Извлечены данные из ТЗ: система={tech_system}, адрес={tech_address}")
        if tech_doc == "Номер и дата не найдены":
            result_lines.append("<u>Ошибка:</u> Не удалось корректно выгрузить дату. Будет направлена на ручную проверку.<br>")
            final_errors.append("Не удалось определить дату технического задания.")
            logger.error("Не удалось извлечь дату из ТЗ")
        else:
            result_lines.append(f"<u>Дата:</u> {tech_doc}<br>")
            logger.debug(f"Извлечена дата: {tech_doc}")
        if reference_address and normalize_address_for_comparison(tech_address) != normalize_address_for_comparison(reference_address):
            result_lines.append("<u>Ошибка:</u> Адрес технического задания не совпадает с эталонным адресом:<br>")
            result_lines.append(f"Техническое задание: {tech_address}<br>")
            result_lines.append(f"Эталон: {reference_address}<br>")
            final_errors.append("Адрес технического задания не совпадает с эталонным адресом.")
            logger.warning(f"Адрес ТЗ не совпадает: {tech_address} != {reference_address}")
        if reference_system and tech_system.lower() != reference_system.lower():
            result_lines.append("<u>Ошибка:</u> Система технического задания не совпадает с эталонной системой:<br>")
            result_lines.append(f"Техническое задание: {tech_system}<br>")
            result_lines.append(f"Эталон: {reference_system}<br>")
            final_errors.append("Система технического задания не совпадает с эталонной системой.")
            logger.warning(f"Система ТЗ не совпадает: {tech_system} != {reference_system}")
    else:
        result_lines.append("<h3><b>Техническое задание</b></h3>")
        result_lines.append("<u>Ошибка:</u> Техническое задание не найдено.<br>")
        final_errors.append("Техническое задание отсутствует.")
        logger.warning("Техническое задание не найдено")

    if "Выписка СРО" in docs:
        result_lines.append("<h3><b>Выписка СРО</b></h3>")
        sro_file = docs["Выписка СРО"]
        sro_text = extract_text_from_pdf(sro_file)
        sro_date = extract_sro_date(sro_text)
        logger.debug(f"Извлечённая дата СРО: {sro_date}, тип: {type(sro_date)}")
        if sro_date:
            deadline_date = sro_date + datetime.timedelta(days=30)
            today = datetime.date.today()
            if today <= deadline_date:
                result_lines.append(f"<i>Выписка СРО оформлена вовремя:</i> {sro_date} (срок до {deadline_date})<br>")
                logger.info(f"Выписка СРО оформлена вовремя: {sro_date}")
            else:
                result_lines.append(f"<u>Ошибка:</u> Выписка СРО оформлена с опозданием: {sro_date}. Допустимый срок - до {deadline_date}<br>")
                logger.warning(f"Выписка СРО просрочена: {sro_date}")
        else:
            result_lines.append("<u>Ошибка:</u> Дата выписки СРО не найдена в документе<br>")
            logger.warning("Дата СРО не найдена")
    else:
        result_lines.append("<h3><b>Выписка СРО</b></h3>")
        result_lines.append("<u>Ошибка:</u> Выписка СРО не найдена.<br>")
        final_errors.append("Выписка СРО отсутствует.")
        logger.warning("Выписка СРО не найдена")

    if "Пояснительная записка" in docs:
        result_lines.append("<h3><b>Пояснительная записка</b></h3>")
        explanatory_file = docs["Пояснительная записка"]
        expl_text = extract_text_from_pdf_all(explanatory_file)
        expl_system = extract_system(expl_text, systems_list)
        expl_address = extract_address(expl_text)
        expl_price_date, expl_price_display = extract_price_level_explanatory(expl_text)
        expl_smet_match = re.search(r'Сметная стоимость.*?([\d\s.,]+)\s*тыс\.?\s*руб', expl_text,
                                     re.IGNORECASE | re.DOTALL)
        result_lines.append(f"<u>Система:</u> {expl_system}<br>")
        result_lines.append(f"<u>Адрес:</u> {expl_address}<br>")
        logger.info(f"Извлечены данные из Пояснительной записки: система={expl_system}, адрес={expl_address}")
        if expl_smet_match:
            expl_stoimost = expl_smet_match.group(1).replace(" ", "")
            result_lines.append(f"<u>Сметная стоимость:</u> {expl_stoimost} тыс. руб.<br>")
            logger.debug(f"Извлечена стоимость: {expl_stoimost}")
        else:
            expl_stoimost = None
            result_lines.append("<u>Ошибка:</u> Сметная стоимость отсутствует, сравнение невозможно.<br>")
            final_errors.append("Сметная стоимость отсутствует в пояснительной записке.")
            logger.warning("Сметная стоимость отсутствует")
        if reference_address and normalize_address_for_comparison(expl_address) != normalize_address_for_comparison(reference_address):
            result_lines.append("<u>Ошибка:</u> Адрес пояснительной записки не совпадает с эталонным адресом:<br>")
            result_lines.append(f"Пояснительная: {expl_address}<br>")
            result_lines.append(f"Эталон: {reference_address}<br>")
            final_errors.append("Адрес пояснительной записки не совпадает с эталонным адресом.")
            logger.warning(f"Адрес пояснительной не совпадает: {expl_address} != {reference_address}")
        if reference_system and expl_system.lower() != reference_system.lower():
            result_lines.append("<u>Ошибка:</u> Система пояснительной записки не совпадает с эталонной системой:<br>")
            result_lines.append(f"Пояснительная: {expl_system}<br>")
            result_lines.append(f"Эталон: {reference_system}<br>")
            final_errors.append("Система пояснительной записки не совпадает с эталонной системой.")
            logger.warning(f"Система пояснительной не совпадает: {expl_system} != {reference_system}")
        if expl_price_date:
            today = datetime.date.today()
            if expl_price_date >= today - relativedelta(months=3):
                result_lines.append("<i>Уровень цен актуален (не старше 3 месяцев).</i><br>")
                logger.info("Уровень цен актуален")
            else:
                result_lines.append("<u>Ошибка:</u> Уровень цен устарел!<br>")
                final_errors.append("Уровень цен в пояснительной записке устарел.")
                logger.warning("Уровень цен устарел")
            result_lines.append(f"<u>Дата уровня цен:</u> {expl_price_display}<br>")
        else:
            result_lines.append("<u>Ошибка:</u> Не удалось распознать уровень цен.<br>")
            logger.warning("Уровень цен не распознан")
    else:
        result_lines.append("<h3><b>Пояснительная записка</b></h3>")
        result_lines.append("<u>Ошибка:</u> Пояснительная записка не найдена.<br>")
        final_errors.append("Пояснительная записка отсутствует.")
        logger.warning("Пояснительная записка не найдена")

    if "Заявление" in docs and normalize_system(statement_system) in [normalize_system(x) for x in systems_project_required]:
        result_lines.append("<h3><b>Проверка Проекта и Заключения</b></h3>")
        project_file = docs.get("Проект")
        conclusion_file = docs.get("Заключение")
        valid_project = False
        valid_conclusion = False
        if project_file and os.path.splitext(project_file)[1].lower() == ".pdf":
            proj_text = extract_text_from_pdf(project_file)
            if "проектная документация" in proj_text.lower():
                valid_project = True
                logger.info("Проектная документация найдена")
        if conclusion_file and os.path.splitext(conclusion_file)[1].lower() == ".pdf":
            concl_text = extract_text_from_pdf(conclusion_file)
            if "заключение" in concl_text.lower():
                valid_conclusion = True
                logger.info("Заключение найдено")
        if valid_project and valid_conclusion:
            result_lines.append("<i>Проект и заключение присутствуют и корректны.</i><br>")
            logger.info("Проект и Заключение корректны")
        else:
            missing = []
            if not valid_project:
                missing.append("проект")
            if not valid_conclusion:
                missing.append("заключение")
            result_lines.append(f"<u>Ошибка:</u> Для данной системы требуется проект и заключение, но следующие документы не найдены или некорректны: {', '.join(missing)}.<br>")
            final_errors.append(f"Отсутствуют или некорректны: {', '.join(missing)}")
            logger.warning(f"Отсутствуют или некорректны: {missing}")

    result_lines.append("<h3><b>Итог</b></h3>")
    if final_errors:
        result_lines.append("<u>Обнаружены следующие ошибки:</u><ul>")
        for err in final_errors:
            result_lines.append(f"<li>{err}</li>")
        result_lines.append("</ul>")
        logger.warning(f"Обнаружены ошибки: {final_errors}")
    else:
        result_lines.append("<i>Ошибок не обнаружено. Все документы соответствуют требованиям.</i>")
        logger.info("Ошибок не обнаружено")

    final_result = "".join(result_lines)
    logger.info("Обработка документов завершена")
    return final_result

# -------------------------- Телеграм-бот (telebot) --------------------------

# Замените 'YOUR_TELEGRAM_BOT_TOKEN' на токен вашего бота
BOT_TOKEN = '7617318744:AAG7ognxoGFLVbTubOwucXi8Jre9pO78WEY'
bot = telebot.TeleBot(BOT_TOKEN)

# Словарь для хранения файлов от каждого пользователя (по chat_id)
user_files = {}

@bot.message_handler(commands=['start'])
def start_handler(message):
    welcome_text = (
        "Добро пожаловать в бота обработки документов.\n"
        "Отправляйте документы (PDF, Excel и т.д.), а затем команду /process для запуска проверки."
    )
    bot.send_message(message.chat.id, welcome_text)

@bot.message_handler(content_types=['document'])
def document_handler(message):
    chat_id = message.chat.id
    file_info = bot.get_file(message.document.file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    # Формируем имя файла с учетом chat_id, чтобы избежать конфликтов
    file_path = os.path.join(TEMP_DIR, f"{chat_id}_{message.document.file_name}")
    with open(file_path, 'wb') as new_file:
        new_file.write(downloaded_file)
    # Сохраняем путь к файлу для данного пользователя
    if chat_id not in user_files:
        user_files[chat_id] = []
    user_files[chat_id].append(file_path)
    bot.reply_to(message, f"Документ '{message.document.file_name}' получен.")

@bot.message_handler(commands=['process'])
def process_handler(message):
    chat_id = message.chat.id
    if chat_id not in user_files or not user_files[chat_id]:
        bot.send_message(chat_id, "Нет документов для обработки. Пожалуйста, отправьте файлы.")
        return
    try:
        result = process_documents(user_files[chat_id])
        bot.send_message(chat_id, result, parse_mode="HTML")
    except Exception as e:
        bot.send_message(chat_id, f"Ошибка при обработке документов: {e}")
    finally:
        # Очистка списка файлов после обработки
        user_files[chat_id] = []

if __name__ == '__main__':
    logger.info("Бот запущен")
    bot.polling(none_stop=True)
