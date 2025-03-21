import os
import re
import io
import contextlib
import datetime
from PIL import ImageEnhance
import imutils
import pandas as pd
import numpy as np
import cv2
import fitz  # PyMuPDF
import pdfplumber
import pytesseract
from PIL import Image
from dateutil.relativedelta import relativedelta
import telebot
import time
import sqlite3
import PyPDF2

from systems_list_copy import systems_list, systems_project_required

from imutils.contours import sort_contours

conn = sqlite3.connect("tn_server/bot_using.db", check_same_thread=False, isolation_level=None)
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



def log_user_activity(message, action):
    user_id = message.from_user.id
    # Если username отсутствует, используем first_name в качестве альтернативы
    username = message.from_user.username if message.from_user.username else message.from_user.first_name
    text = message.text if message.text else ''
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO user_activity (user_id, username, action, content) VALUES (?, ?, ?, ?)",
        (user_id, username, action, text)
    )
    conn.commit()
    cur.close()



TOKEN = "7867163959:AAFgx289fM7Et42x6s5b6aXmRy99wxs-Sm0"
bot = telebot.TeleBot(TOKEN)


TEMP_DIR = "tn_server/temp"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)


user_files = {}

def handwritten(image):
    block_size = 9
    constant = 2
    blur = cv2.GaussianBlur(image, (7,7), 0)
    fnoise = cv2.medianBlur(blur, 3)
    th1 = cv2.adaptiveThreshold(fnoise, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, constant)
    th2 = cv2.adaptiveThreshold(th1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, constant)
    blu = cv2.GaussianBlur(th2, (5, 5), 0)
    fnois = cv2.medianBlur(th2, 3)
    return blu, fnois
def remove_noise(image):
    return cv2.bilateralFilter(image, 9, 25, 25)

def dilate(image):
    kernel = np.ones((3,3), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)

def erode(image):
    kernel = np.ones((1,1), np.uint8)
    return cv2.erode(image, kernel, iterations=1)


def preprocess_image(image):

    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    # extract_stamp
    img = erode(gray)
    img = remove_noise(img)
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpen = cv2.filter2D(img, -1, sharpen_kernel)
    thresh = cv2.threshold(sharpen, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # thresh = extract_stamp(thresh)
    # _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(thresh)


def correct_rotation(image: Image.Image,
                     canny_threshold1: int = 50,
                     canny_threshold2: int = 150,
                     hough_threshold: int = 100,
                     min_line_length: int = 100,
                     max_line_gap: int = 10,
                     angle_epsilon: float = 0.1) -> Image.Image:
    """
    Корректирует поворот изображения для повышения качества распознавания текста.

    Параметры:
        image (PIL.Image): исходное изображение.
        canny_threshold1 (int): нижний порог для детектора Canny.
        canny_threshold2 (int): верхний порог для детектора Canny.
        hough_threshold (int): порог для метода HoughLinesP.
        min_line_length (int): минимальная длина линии для HoughLinesP.
        max_line_gap (int): максимальное расстояние между линиями для их объединения.
        angle_epsilon (float): порог, ниже которого угол считается нулевым.

    Возвращает:
        PIL.Image: изображение, повернутое на вычисленный угол.
    """
    # Преобразуем изображение в формат OpenCV (RGB -> BGR)
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    # Преобразуем в оттенки серого
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    # Детектирование краев с помощью алгоритма Canny
    edges = cv2.Canny(gray, canny_threshold1, canny_threshold2, apertureSize=3)

    # Поиск линий методом HoughLinesP
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=hough_threshold,
                            minLineLength=min_line_length, maxLineGap=max_line_gap)

    if lines is not None and len(lines) > 0:
        # Извлекаем углы для каждой найденной линии
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            angles.append(angle)
        # Находим медианный угол (устойчив к выбросам)
        median_angle = np.median(angles)

        # Корректировка угла: если угол слишком велик, сдвигаем его на 90° для получения минимального поворота
        if abs(median_angle) > 45:
            median_angle = median_angle - 90 if median_angle > 0 else median_angle + 90

        # Если вычисленный угол слишком мал, поворот не требуется
        if abs(median_angle) < angle_epsilon:
            return image

        # Поворачиваем изображение с использованием высококачественной интерполяции
        rotated = image.rotate(-median_angle, expand=True, resample=Image.BICUBIC)
        return rotated
    else:
        # Если линии не обнаружены, возвращаем исходное изображение
        return image


def extract_text_from_pdf(pdf_path, skip_top_fraction=None):
    doc = fitz.open(pdf_path)
    page = doc.load_page(0)
    rect = page.rect
    clip_rect = (fitz.Rect(rect.x0, rect.y0 + rect.height * skip_top_fraction, rect.x1, rect.y1)
                 if skip_top_fraction else rect)
    pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), clip=clip_rect)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    corrected = correct_rotation(img)
    preprocessed = preprocess_image(corrected)
    text = pytesseract.image_to_string(preprocessed, lang="rus", config='--psm 3')

    return re.sub(r'\s+', ' ', text)


def extract_text_from_pdf_all(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        corrected = correct_rotation(img)
        preprocessed = preprocess_image(corrected)
        text = pytesseract.image_to_string(preprocessed, lang="rus", config='--psm 3')

        full_text += " " + re.sub(r'\s+', ' ', text)

        # Вывод промежуточного текста для отладки
        with open("output.txt", "a") as f:
            f.write(text)

    return full_text.strip()


def extract_system(text, systems_list):


    def remove_remont_endings(s):

        return re.sub(r'\bремонт\w*\b', 'ремонт', s, flags=re.IGNORECASE)


    processed_text = remove_remont_endings(text)

    for system in systems_list:
        processed_system = remove_remont_endings(system)
        if processed_system.lower() in processed_text.lower():
            return system
    return "Описание системы не найдено"

def clean_text(text):
    text = re.sub(r'[^\w\s.,:]', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def extract_address(text):
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
    m = re.search(r'(г\.\s*\S.*)', address, re.IGNORECASE)
    if m:
        address = m.group(1).strip()
    return address


def extract_doc_date_number(text):
    pattern = r'№\s*[\d/-]+[\s_]*от\s*[«"]?\s*\d{1,2}[»"]?\s*(?:[а-яА-Я]+|\d{1,2})\s*\d{4}\s*г[.,]?'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        doc_date = match.group().strip().rstrip(' .,')
        return doc_date
    return "Номер и дата не найдены"




def clean_address(address):

    return re.sub(r'^(?:ДОКУМЕНТ ПОДПИСАН ЭЛЕКТРОННОЙ ПОДПИСЬЮ\s+)?(?:[А-ЯЁа-яё]+\s+область,\s*)', '', address,
                  flags=re.IGNORECASE).strip()

def extract_table(pdf_path, page_num, table_num):
    # Открываем файл pdf
    pdf = pdfplumber.open(pdf_path)
    # Находим исследуемую страницу
    table_page = pdf.pages[page_num]
    # Извлекаем соответствующую таблицу
    table = table_page.extract_tables()[table_num]
    return table

# Преобразуем таблицу в соответствующий формат
def table_converter(table):
    table_string = ''
    # Итеративно обходим каждую строку в таблице
    for row_num in range(len(table)):
        row = table[row_num]
        # Удаляем разрыв строки из текста с переносом
        cleaned_row = [item.replace('\n', ' ') if item is not None and '\n' in item else 'None' if item is None else item for item in row]
        # Преобразуем таблицу в строку
        table_string+=('|'+'|'.join(cleaned_row)+'|'+'\n')
    # Удаляем последний разрыв строки
    table_string = table_string[:-1]
    return table_string

def extract_sro_date(text):
    """Извлечение даты выписки СРО"""
    # Ищем дату с разделителями . или -
    match = re.search(r'(\d{1,2}[.]\d{1,2}[.]\d{4})', text)
    if match:
        data_str = match.group(1)
        # Поиск этих форматов
        for fmt in ("%d.%m.%Y"):
            try:
                datetime.datetime.strptime(data_str, fmt).date()
            except ValueError:
                continue
    return None


# Функции предобработки и корректировки изображения для Договора (используются только для договора)
def contract_preprocess_image(image):
    # Применяем простой бинарный порог (как во втором коде)
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(thresh)

def contract_correct_rotation(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
    if lines is not None:
        angles = [np.degrees(np.arctan2(y2 - y1, x2 - x1))
                  for x1, y1, x2, y2 in [line[0] for line in lines]]
        median_angle = np.median(angles)
        if abs(median_angle) > 45:
            median_angle -= 90
        return image.rotate(-median_angle, expand=True)
    return image

# Функции для работы с адресом (из второго кода)
def build_address_regex(input_address, max_gap=350):
    parts = re.split(r'\s*,\s*', input_address.strip())
    token_list = []
    for part in parts:
        tokens = re.findall(r'[а-яА-Я0-9-]+', part)
        token_list.extend(tokens)
    regex_parts = [re.escape(token_list[0])]
    for token in token_list[1:]:
        separator = r'(?:[.,]?\s+)'  # допускаем разделитель с пробелом
        gap = r'[\s\S]{0,' + str(max_gap) + r'}?'
        regex_parts.append(separator + gap + re.escape(token))
    pattern = ''.join(regex_parts)
    return pattern, token_list

def find_address_in_text(ocr_text, input_address, max_gap=350):
    pattern, tokens = build_address_regex(input_address, max_gap)
    normalized_text = re.sub(r'\s+', ' ', ocr_text.replace('\n', ' '))
    matches = list(re.finditer(pattern, normalized_text, flags=re.IGNORECASE))
    if matches:
        best_match = min(matches, key=lambda m: len(m.group(0)))
        return best_match.group(0), tokens
    return None, tokens

def clean_contract_address(found_text, tokens, input_address):
    """
    Извлекает из найденного OCR адреса только фрагменты, соответствующие ключевым токенам.
    """
    cleaned = ""
    current_pos = 0
    prev_end = 0
    for token in tokens:
        match = re.search(re.escape(token), found_text[current_pos:], flags=re.IGNORECASE)
        if match:
            start = current_pos + match.start()
            end = current_pos + match.end()
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
            return input_address
    return cleaned.strip()

def add_dot_if_missing(address):
    fixed = re.sub(r'\bд(?!\.)\s*(\d+)', r'д.\1', address, flags=re.IGNORECASE)
    return fixed

def add_space_after_house(address):
    return re.sub(r'\b(д\.)\s*(\d)', r'\1 \2', address, flags=re.IGNORECASE)


def check_contract_address(contract_file, input_address):
    """
    Обрабатывает документ договора:
      - Выполняет OCR с использованием предобработки и корректировки поворота (только для договора)
      - Ищет в извлечённом тексте адрес, аналогичный input_address
      - Возвращает очищенный адрес, если найден, иначе None
    """
    try:
        pdf_document = fitz.open(contract_file)
        extracted_text = ""
        for page in pdf_document:
            scale = 4  # увеличение разрешения
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            # Увеличение изображения для повышения качества OCR
            resized_img = img.resize((int(img.width * 1.5), int(img.height * 1.5)), Image.LANCZOS)
            # Используем функции предобработки, специфичные для договора:
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
            return cleaned_address
        else:
            return None
    except Exception as e:
        print("Ошибка при обработке договора:", e)
        return None


def parse_price_level(date_str):

    quarter_pattern = r'([IVX]+)\s*(?:квартал|кв\.?)\s+(\d{4})'
    m = re.search(quarter_pattern, date_str, re.IGNORECASE)
    if m:
        roman = m.group(1).upper()
        year = int(m.group(2))
        roman_to_int = {"I": 1, "II": 2, "III": 3, "IV": 4}
        quarter = roman_to_int.get(roman, 4)
        if quarter > 4:
            quarter = 4
        return {1: datetime.date(year, 3, 31),
                2: datetime.date(year, 6, 30),
                3: datetime.date(year, 9, 30),
                4: datetime.date(year, 12, 31)}.get(quarter)
    numeric_quarter_pattern = r'(\d+)\s*(?:квартал|кв\.?)\s+(\d{4})'
    m_num = re.search(numeric_quarter_pattern, date_str, re.IGNORECASE)
    if m_num:
        quarter = int(m_num.group(1))
        if quarter > 4:
            quarter = 4
        year = int(m_num.group(2))
        return {1: datetime.date(year, 3, 31),
                2: datetime.date(year, 6, 30),
                3: datetime.date(year, 9, 30),
                4: datetime.date(year, 12, 31)}.get(quarter)
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
                return datetime.date(year, month, day)
            except ValueError:
                return None
    return None


smeta_stoimost = None


def check_smeta(smeta_file, reference_address, statement_system):
    errors = []
    global smeta_stoimost

    result_lines = ["<h3>Смета</h3>"]  # Начало секции с заголовком

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
        result_lines.append(f"<b>Адрес:</b> {smeta_address}<br>")
    else:
        smeta_address = "Адрес не найден"
        result_lines.append("<b>Адрес:</b> Адрес не найден<br>")

    smeta_system = None
    for sys in systems_list:
        if sys.lower() in all_text.lower():
            smeta_system = sys
            break
    if smeta_system is None:
        smeta_system = "Описание системы не найдено"
        result_lines.append("<b>Система:</b> Описание системы не найдено<br>")
    else:
        result_lines.append(f"<b>Система:</b> {smeta_system}<br>")

    smeta_address_clean = clean_address(smeta_address)
    reference_address_clean = clean_address(reference_address)
    if reference_address and smeta_address_clean.lower() != reference_address_clean.lower():
        result_lines.append("<b>Ошибка:</b> Адрес сметы не совпадает с эталонным адресом!<br>")
        result_lines.append(f"Смета: {smeta_address_clean}<br>")
        result_lines.append(f"Эталон: {reference_address_clean}<br>")
        errors.append("Адрес сметы не совпадает с эталонным адресом.")
    else:
        result_lines.append("<i>Адрес сметы совпадает с эталонным адресом.</i><br>")

    smetnaya_stoimost_match = re.search(r'Сметная стоимость\s+([\d.,]+)\s*тыс\.?руб', all_text, re.IGNORECASE)
    if smetnaya_stoimost_match:
        smeta_stoimost = smetnaya_stoimost_match.group(1)
        result_lines.append(f"<b>Сметная стоимость:</b> {smeta_stoimost} тыс. руб.<br>")
    else:
        smeta_stoimost = None
        result_lines.append("<b>Ошибка:</b> Сметная стоимость не найдена!<br>")
        errors.append("Сметная стоимость не найдена!")

    osnovanie_match = re.search(r'\bОснование\b\s*[:\-]?\s*([\w\s]+)', all_text, re.IGNORECASE)
    if osnovanie_match:
        osnovanie_value = osnovanie_match.group(1).strip()
        if osnovanie_value:
            result_lines.append(f"<b>Основание:</b> {osnovanie_value}<br>")
        else:
            result_lines.append("<b>Ошибка:</b> Основание не найдено!<br>")
            errors.append("Основание не найдено!")
    else:
        result_lines.append("<b>Ошибка:</b> Основание не найдено!<br>")
        errors.append("Основание не найдено!")

    price_pattern = r'текущем\s+уровне\s+цен\s*[:]?[\s]*([^\s].+?)(?=\s{2,}|$)'
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
            else:
                result_lines.append("<b>Ошибка:</b> Уровень цен устарел!<br>")
                errors.append("Уровень цен устарел!")
            result_lines.append(f"<b>Дата уровня цен:</b> {price_level_display}<br>")
        else:
            result_lines.append("<b>Ошибка:</b> Не удалось распознать дату уровня цен в смете.<br>")
            errors.append("Не удалось распознать уровень цен в смете.")
    else:
        result_lines.append("<b>Ошибка:</b> Уровень цен не найден в смете.<br>")
        errors.append("Уровень цен не найден в смете.")

    # Объединяем строки в HTML-вывод
    html_output = "".join(result_lines)
    # Перенаправляем вывод в stdout для совместимости с contextlib.redirect_stdout
    print(html_output)

    return errors


def extract_price_level_explanatory(text):
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
        return (quarter_date, quarter_str)

    return (None, None)

def normalize_system(s):
    return re.sub(r'\s+', ' ', s.strip().lower())

def normalize_address_for_comparison(address):

    normalized = address.lower()
    normalized = re.sub(r'[,.]', '', normalized)
    return re.sub(r'\s+', ' ', normalized).strip()

def extract_text_from_pdf_first_page(file_path):
    with open(file_path, 'rb') as f:
        reader = PyPDF2.PdfFileReader(f)
        first_page = reader.pages[0]
        return first_page.extractText()


def classify_document(file_path):

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".xlsx":
        return "Смета"
    text = ""
    if ext == ".pdf":
        try:
            text = extract_text_from_pdf(file_path)
        except Exception:
            text = ""
    text_low = text.lower()
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
        "Техническое задание": ["техническое задание"],
        "Информационно-удостоверяющий лист": ["уил", "информационно-удостоверяющий лист"],
        "Выписка СРО": ["выписка"],
        "Договор": ["договор о проведении"]
    }
    first_occurrence = None
    doc_type_found = None

    # Проходим по каждому типу документа и его ключевым словам
    for doc_type, phrases in keywords.items():
        for phrase in phrases:
            # Ищем фразу в исходном тексте
            index = text_low.find(phrase)
            if index == -1:
                # Если не найдено, ищем в нормализованном тексте (без пробелов)
                norm_phrase = re.sub(r'\s+', '', phrase)
                index = norm_text.find(norm_phrase)
            if index != -1:
                # Если найдено первое вхождение или оно раньше предыдущего, обновляем результат
                if first_occurrence is None or index < first_occurrence:
                    first_occurrence = index
                    doc_type_found = doc_type

    return doc_type_found if doc_type_found else "Неизвестный документ"


@bot.message_handler(commands=['start'])
def start_handler(message):
    log_user_activity(message, "/start")
    bot.send_message(message.chat.id,
                     "Здравствуйте! Отправьте мне пакет документов. Вы можете присылать файлы в любом порядке. "
                     "После загрузки всех файлов введите команду /process. Я определю тип каждого документа и подготовлю подробный отчёт. ")

@bot.message_handler(content_types=['document'])
def handle_document(message):
    log_user_activity(message, "document")
    chat_id = message.chat.id
    file_info = bot.get_file(message.document.file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    filename = message.document.file_name
    file_path = os.path.join(TEMP_DIR, filename)
    with open(file_path, "wb") as new_file:
        new_file.write(downloaded_file)
    if chat_id not in user_files:
        user_files[chat_id] = []
    user_files[chat_id].append(file_path)
    doc_type = classify_document(file_path)
    bot.reply_to(message,
                 f"Файл получен: {filename}\nОпределённый тип документа: {doc_type}\nЕсли все файлы загружены, введите команду /process.")
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO user_activity (user_id, username, action, content) VALUES (?, ?, ?, ?)",
        (message.from_user.id, message.from_user.username, "document_upload", f"{filename} | Тип: {doc_type}")
    )
    conn.commit()
    cur.close()



def check_approval_sheet_in_document(pdf_path):
    full_text = extract_text_from_pdf_all(pdf_path)
    normalized_text = re.sub(r'\s+', ' ', full_text).lower()
    found = "лист согласования" in normalized_text
    preview_text = normalized_text[:500]
    return found, preview_text


def extract_first_four_digits(cost_str):
    digits = ''.join(re.findall(r'\d', cost_str))
    return digits[:4]


@bot.message_handler(commands=['process'])
def process_handler(message):
    log_user_activity(message, "/process")
    chat_id = message.chat.id
    if chat_id not in user_files or not user_files[chat_id]:
        bot.send_message(chat_id, "Нет документов для обработки. Пожалуйста, отправьте файлы.")
        return
    bot.send_message(chat_id, "Идет анализ информации, пожалуйста, подождите...")


    docs = {}
    for file_path in user_files[chat_id]:
        doc_type = classify_document(file_path)
        if doc_type not in docs:
            docs[doc_type] = file_path

    result_lines = []

    if "Договор" not in docs or "НОПРИЗ" not in docs:
        docs["ЭКУ"] = "Документы по ЭКУ"

    # фУНКЦИЯ эку, Добавить определение документов для негос (6, 11, 5, 7, 12)
        expected_docs = ["Заявление", "Дефектная ведомость", "Акт обследования", "Пояснительная записка", "Смета", "Выписка СРО"]
        missing_docs = [doc for doc in expected_docs if doc not in docs]

        #result_lines = []
        result_lines.append("Получилось! Вот результаты проверки ваших документов ЭКУ:")

        if missing_docs:
            result_lines.append("Отсутствуют следующие обязательные документы: " + ", ".join(missing_docs))
        else:
            result_lines.append("Все обязательные документы присутствуют.\n")

    if "Договор" in docs or "НОПРИЗ" in docs:
        docs["негос"] = "Документы по негос"

        negos_docs = ["НОПРИЗ", "Техническое задание", "Выписка СРО", "Информационно-удостоверяющий лист", "Заявление", "Смета", "Акт обследования", "Дефектная ведомость", "Договор", ]
        missing_negos = [doc for doc in negos_docs if doc not in docs]
        #result_lines = []
        result_lines.append("Получилось! Вот результаты проверки ваших документов негос:")

        if missing_negos:
            result_lines.append("Отсутствуют следующие обязательные документы: " + ", ".join(missing_negos))
        else:
            result_lines.append("Все обязательные документы присутствуют.\n")

    final_errors = []

    if "Заявление" in docs:
        statement_file = docs.get("Заявление")
        st_text = extract_text_from_pdf(statement_file, skip_top_fraction=1/6)
        st_system = extract_system(st_text, systems_list)
        st_address = extract_address(st_text)
        result_lines.append("<b>Заявление:</b>")
        result_lines.append(f"Система: {st_system}")
        result_lines.append(f"Адрес: {st_address}\n")

        statement_full_text = extract_text_from_pdf_all(statement_file)
        if "реестр смет" in statement_full_text.lower():
            result_lines.append("В заявлении обнаружен 'Реестр смет': адрес и система заявления совпадают с данными дефектной ведомости.\n")
    else:
        result_lines.append("Заявление не найдено.\n")
        final_errors.append("Заявление отсутствует.")

    # ------------------- Обработка дефектной ведомости -------------------
    if "Дефектная ведомость" in docs:
        defect_file = docs.get("Дефектная ведомость")
        df_text = extract_text_from_pdf(defect_file)
        df_system = extract_system(df_text, systems_list)
        df_address = extract_address(df_text)
        df_doc = extract_doc_date_number(df_text)
        result_lines.append("<b>Дефектная ведомость:</b>")
        result_lines.append(f"Система: {df_system}")
        result_lines.append(f"Адрес: {df_address}")
        if df_doc == "Номер и дата не найдены":
            result_lines.append("Не удалось корректно выгрузить дату и номер у дефектной ведомости. Будет направлена на ручную проврку УТНКР\n")
            final_errors.append("Не удалось определить номер и дату дефектной ведомости. УТНКР вручную сравнит номер и дату у документов")
        else:
            result_lines.append(f"Номер и дата: {df_doc}\n")
        found, preview = check_approval_sheet_in_document(defect_file)
        if found:
            result_lines.append("Лист согласования присутствует в дефектной ведомости.\n")
        else:
            result_lines.append("Лист согласования не найден в дефектной ведомости.\n")
            final_errors.append("Лист согласования отсутствует в дефектной ведомости.")
    else:
        df_doc = "Номер и дата не найдены"
        df_address = "Адрес не найден"
        df_system = "Система не найден"
        result_lines.append("Дефектная ведомость не найдена.\n")
        final_errors.append("Дефектная ведомость отсутствует.")

    # Сравнение адресов и систем между заявлением и дефектной ведомостью
    if "Заявление" in docs and "Дефектная ведомость" in docs:
        if normalize_address_for_comparison(st_address) != normalize_address_for_comparison(df_address):
            result_lines.append("Обнаружено различие в адресах между Заявлением и Дефектной ведомостью:")
            result_lines.append(f"Заявление: {st_address}")
            result_lines.append(f"Дефектная ведомость: {df_address}\n")
            final_errors.append("Адреса заявления и дефектной ведомости не совпадают.")
        else:
            result_lines.append("Адреса заявления и дефектной ведомости совпадают.\n")
        if st_system.lower() != df_system.lower():
            result_lines.append("Обнаружено различие в системах между Заявлением и Дефектной ведомостью:")
            result_lines.append(f"Заявление: {st_system}")
            result_lines.append(f"Дефектная ведомость: {df_system}\n")
            final_errors.append("Системы заявления и дефектной ведомости не совпадают.")
        else:
            result_lines.append("Системы заявления и дефектной ведомости совпадают.\n")

    # -------------------------- Обработка сметы --------------------------
    if "Смета" in docs:
        smeta_file = docs.get("Смета")
        smeta_output = io.StringIO()
        with contextlib.redirect_stdout(smeta_output):
            smeta_errors = check_smeta(smeta_file, df_address if "Дефектная ведомость" in docs else "",
                                       st_system if "Заявление" in docs else "")
        result_lines.append(smeta_output.getvalue())
        if smeta_errors:
            final_errors.extend(smeta_errors)

    else:
        result_lines.append("Смета не найдена.\n")
        final_errors.append("Смета отсутствует.")

    # ------------------- Обработка акта обследования -------------------
    if "Акт обследования" in docs:
        act_file = docs.get("Акт обследования")
        act_text = extract_text_from_pdf(act_file)
        act_system = extract_system(act_text, systems_list)
        act_address = extract_address(act_text)
        act_doc = extract_doc_date_number(act_text)

        if act_address == "Адрес не найден" and "Дефектная ведомость" in docs:
            act_address = df_address

        result_lines.append("<b>Акт обследования:</b>")
        result_lines.append(f"Система: {act_system}")
        result_lines.append(f"Адрес: {act_address}\n")

        if act_doc == "Номер и дата не найдены":
            result_lines.append("Не удалось корректно выгрузить дату и номер у акта обследования. Будет направлена на ручную проверку УТНКР.\n")
            final_errors.append("Не удалось определить номер и дату акта обследования. УТНКР вручную сравнит номер и дату у документов")
        else:
            result_lines.append(f"Номер и дата: {act_doc}\n")

        found, preview = check_approval_sheet_in_document(act_file)
        if found:
            result_lines.append("Лист согласования присутствует в акте обследования.\n")
        else:
            result_lines.append("Лист согласования не найден в акте обследования.\n")
            final_errors.append("Лист согласования отсутствует в акте обследдения.")

        if act_doc != "Номер и дата не найдены" and df_doc != "Номер и дата не найдены":
            if act_doc.lower() == df_doc.lower():
                result_lines.append("Номер и дата акта обследования совпадают с данными дефектной ведомости.")
            else:
                result_lines.append("Номер и/или дата акта обследования НЕ совпадают с данными дефектной ведомости.")
                result_lines.append(f"Акт: {act_doc}")
                result_lines.append(f"Дефектная ведомость: {df_doc}\n")
                final_errors.append("Номер и/или дата акта обследования не совпадают с дефектной ведомостью.")

        if normalize_address_for_comparison(act_address) == normalize_address_for_comparison(df_address):
            result_lines.append("Адрес акта обследования совпадает с адресом дефектной ведомости.\n")
        else:
            result_lines.append("Адрес акта обследования НЕ совпадает с адресом дефектной ведомости.")
            result_lines.append(f"Акт: {act_address}")
            result_lines.append(f"Дефектная ведомость: {df_address}\n")
            final_errors.append("Адрес акта обследования не совпадает с адресом дефектной ведомости.")
        if act_system.lower() == df_system.lower():
            result_lines.append("Система акта обследования совпадает с системой дефектной ведомости.\n")
        else:
            result_lines.append("Система акта обследования НЕ совпадает с системой дефектной ведомости.")
            result_lines.append(f"Акт: {act_system}")
            result_lines.append(f"Дефектная ведомость: {df_system}\n")
            final_errors.append("Система акта обследования не совпадает с системой дефектной ведомости.")
    else:
        result_lines.append("Акт обследования не найден.\n")
        final_errors.append("Акт обследования отсутствует.")

    # --------------- Обработка технического задания ---------------

    if "Техническое задание" in docs:
        tech_file = docs.get("Техническое задание")
        tech_text = extract_text_from_pdf(tech_file)
        tech_system = extract_system(tech_text, systems_list)
        tech_address = extract_address(tech_text)
        tech_doc = extract_doc_date_number(tech_text)

        if tech_address == "Адрес не найден":
            tech_address = df_address

        result_lines.append("<b>Техническое задание:</b>")
        result_lines.append(f"Система: {tech_system}")
        result_lines.append(f"Адрес: {tech_address}")

        if tech_doc == "Дата не найдена":
            result_lines.append(
                "Не удалось корректно выгрузить дату у технического задания. Будет направлена на ручную проверку УТНКР.\n")
            final_errors.append(
                "Не удалось определить дату технического задания. УТНКР вручную сравнит номер и дату у документов")
        else:
            result_lines.append(f"Дата: {tech_doc}\n")

    if "Выписка СРО" in docs:
        sro_file = docs.get("Выписка СРО")
        sro_text = extract_text_from_pdf(sro_file)
        sro_date = extract_sro_date(sro_text)

        if sro_date:
            current_date = datetime.date.today()
            deadline_date = current_date + datetime.timedelta(days=30)
            if sro_date <= deadline_date:
                result_lines.append(f"Выписка СРО оформлена вовремя: {sro_date} (срок до {deadline_date}")
            else:
                result_lines.append(f"Выписка СРО оформлена с опозданием: {sro_date}. Допустимый срок - до {deadline_date}")
        else:
            result_lines.append("Дата выписки СРО не найдена в документе")


    # --------------- Обработка пояснительной записки ---------------
    if "Пояснительная записка" in docs:
        explanatory_file = docs.get("Пояснительная записка")
        expl_text = extract_text_from_pdf_all(explanatory_file)
        expl_system = extract_system(expl_text, systems_list)
        expl_address = extract_address(expl_text)
        expl_price_date, expl_price_display = extract_price_level_explanatory(expl_text)

        expl_smet_match = re.search(
            r'Сметная стоимость.*?([\d\s.,]+)\s*тыс\.?\s*руб',
            expl_text,
            re.IGNORECASE | re.DOTALL
        )
        result_lines.append("<b>Пояснительная записка:</b>")
        result_lines.append(f"Система: {expl_system}")
        result_lines.append(f"Адрес: {expl_address}")
        if expl_smet_match:
            expl_stoimost = expl_smet_match.group(1).replace(" ", "")
            result_lines.append(f"Сметная стоимость в пояснительной записке: {expl_stoimost} тыс. руб.")
        else:
            expl_stoimost = None

        if expl_stoimost is None:
            result_lines.append("Сметная стоимость в пояснительной записке отсутствует, сравнение невозможно.")
            final_errors.append("Сметная стоимость отсутствует в пояснительной записке.")
        elif smeta_stoimost is None:
            result_lines.append("Сметная стоимость в смете отсутствует, сравнение невозможно.")
            final_errors.append("Сметная стоимость отсутствует в смете.")
        else:
            expl_cost_4 = extract_first_four_digits(expl_stoimost)
            smeta_cost_4 = extract_first_four_digits(smeta_stoimost)
            if expl_cost_4 == smeta_cost_4:
                result_lines.append("Сметная стоимость в пояснительной записке соответствует смете.")
            else:
                result_lines.append("Сметная стоимость в пояснительной записке не соответствует смете.")
                final_errors.append("Сметная стоимость не соответствует смете.")

        if expl_price_display:
            result_lines.append(f"Дата уровня цен: {expl_price_display}")
        else:
            result_lines.append("Не удалось распознать уровень цен в пояснительной записке.")
            final_errors.append("Не удалось распознать уровень цен в пояснительной записке.")

        if normalize_address_for_comparison(expl_address) == normalize_address_for_comparison(df_address):
            result_lines.append("Адрес пояснительной записки совпадает с адресом дефектной ведомости.")
        else:
            result_lines.append("Адрес пояснительной записки НЕ совпадает с адресом дефектной ведомости.")
            result_lines.append(f"Пояснительная: {expl_address}")
            result_lines.append(f"Дефектная ведомость: {df_address}")
            final_errors.append("Адрес пояснительной записки не совпадает с адресом дефектной ведомости.")

        if expl_system.lower() == df_system.lower():
            result_lines.append("Система пояснительной записки совпадает с системой дефектной ведомости.")
        else:
            result_lines.append("Система пояснительной записки НЕ совпадает с системой дефектной ведомости.")
            result_lines.append(f"Пояснительная: {expl_system}")
            result_lines.append(f"Дефектная ведомость: {df_system}")
            final_errors.append("Система пояснительной записки не совпадает с системой дефектной ведомости.")

        if expl_price_date:
            today = datetime.date.today()
            if expl_price_date >= today - relativedelta(months=3):
                result_lines.append("Уровень цен в пояснительной записке актуален (не старше 3 месяцев).")
            else:
                result_lines.append("Уровень цен в пояснительной записке устарел!")
                final_errors.append("Уровень цен в пояснительной записке устарел.")
        else:
            result_lines.append("Не удалось распознать уровень цен в пояснительной записке.")
    else:
        result_lines.append("Пояснительная записка не найдена.\n")
        final_errors.append("Пояснительная записка отсутствует.")
    # Надо исправить !!!
    # Если найден документ "Договор", пытаемся извлечь из него адрес с помощью новой предобработки
    if "Договор" in docs:
        dog_file = docs.get("Договор")
        dog_address = check_contract_address(dog_file, st_address)
        result_lines.append("<b>Договор:</b>")
        if dog_address:
            result_lines.append(f"Извлечённый адрес из договора: {dog_address}")
            # Используем адрес из договора для дальнейших проверок (например, сравнения с адресами в других документах)
            st_address = dog_address
        else:
            result_lines.append("Адрес в договоре не найден, используется адрес из заявления.")

    # --------------- Проверка системы, требующей проект и заключение ---------------
    if ("Заявление" in docs) and (normalize_system(st_system) in [normalize_system(x) for x in systems_project_required]):
        project_file = docs.get("Проект")
        conclusion_file = docs.get("Заключение")
        valid_project = False
        valid_conclusion = False
        proj_text = ""
        concl_text = ""

        if project_file and os.path.splitext(project_file)[1].lower() == ".pdf":
            proj_text = extract_text_from_pdf(project_file)
            if "проектная документация" in proj_text.lower():
                valid_project = True
        if conclusion_file and os.path.splitext(conclusion_file)[1].lower() == ".pdf":
            concl_text = extract_text_from_pdf(conclusion_file)
            if "заключение" in concl_text.lower():
                valid_conclusion = True

        if valid_project and valid_conclusion:
            result_lines.append("Проект:")
            result_lines.append(proj_text + "\n")
            result_lines.append("Заключение:")
            result_lines.append(concl_text + "\n")
        else:
            missing = []
            if not valid_project:
                missing.append("проект")
            if not valid_conclusion:
                missing.append("заключение")
            result_lines.append("Для данной системы требуется проект и заключение, но следующие документы не найдены или некорректны: " + ", ".join(missing) + ". Убедитесь, пожалуйста, что они входят в комплект документов.\n")
            final_errors.append("Отсутствуют или некорректны: " + ", ".join(missing))


    user_files[chat_id] = []

    result_lines.append("\n<b>Итог:</b>")
    if final_errors:
        for err in final_errors:
            result_lines.append(f"- {err}")
    else:
        result_lines.append("Ошибок не обнаружено. Все документы соответствуют требованиям.")

    final_result = "\n".join(result_lines)
    for chunk in [final_result[i:i+4096] for i in range(0, len(final_result), 4096)]:
        bot.send_message(chat_id, chunk, parse_mode="HTML")

'''
while True:
    try:
        bot.polling(none_stop=True, interval=0)
    except Exception as e:
        print("Ошибка: ", e)
        time.sleep(15)
'''

'''    if "Договор" not in docs:
        docs["ЭКУ"] = "Документы по ЭКУ"

# фУНКЦИЯ эку, Добавить определение документов для негос (6, 11, 5, 7, 12)
    expected_docs = ["Заявление", "Дефектная ведомость", "Акт обследования", "Пояснительная записка", "Смета"]
    eku_docs_found = []
    for doc in expected_docs:
        if doc in docs:
            eku_docs_found.append((doc, docs[doc]))
    missing_docs = [doc for doc in expected_docs if doc not in docs]

    result_lines = []
    result_lines.append("Получилось! Вот результаты проверки ваших документов ЭКУ:")
    if eku_docs_found:
        for doc, doc_path in eku_docs_found:
            result_lines.append(f"{doc}: {doc_path}")

    if missing_docs:
        result_lines.append("Отсутствуют следующие обязательные документы: " + ", ".join(missing_docs))
    else:
        result_lines.append("Все обязательные документы присутствуют.\n")

    if "экспертно-консультационные услуги" in docs:
        docs["негос"] = "Документы по негос"

    negos_docs = ["НОПРИЗ", "Техническо задание", "Выписка СРО", "Информационно-удостоверяющий лист", "Заявление"]
    negos_docs_found = []
    for doc in negos_docs:
        if doc in docs:
            negos_docs_found.append((doc, docs[doc]))
    missing_negos = [doc for doc in negos_docs if doc not in docs]
    # result_negos = []
    if "негос" in docs or any(doc in docs for doc in negos_docs):
        result_lines.append("Получилось! Вот результаты проверки ваших документов негос:")
        if negos_docs_found:
            for doc, doc_path in negos_docs_found:
                result_lines.append(f"{doc}: {doc_path}")

    if missing_negos:
        result_lines.append("Отсутствуют следующие обязательные документы: " + ", ".join(missing_negos))
    else:
        result_lines.append("Все обязательные документы присутствуют.\n")

    final_message = "\n".join(result_lines)
    bot.send_message(chat_id, final_message)

    final_errors = []'''