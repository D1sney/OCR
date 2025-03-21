import os
import re
import io
import contextlib
import datetime
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

from systems_list import systems_list, systems_project_required  # Предполагается, что этот модуль у вас есть

from imutils.contours import sort_contours

# Создаём (или открываем) базу для логирования действий, если требуется
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


# Функция логирования (в данном примере просто вывод в консоль)
def log_user_activity(action, content):
    print(f"LOG: {action} -> {content}")
    # При необходимости можно также сохранить в БД, как было раньше


TEMP_DIR = "temp"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)


# -------------------------- ФУНКЦИИ ПРЕДОБРАБОТКИ И РАСПОЗНАВАНИЯ --------------------------

def handwritten(image):
    block_size = 9
    constant = 2
    blur = cv2.GaussianBlur(image, (7, 7), 0)
    fnoise = cv2.medianBlur(blur, 3)
    th1 = cv2.adaptiveThreshold(fnoise, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, constant)
    th2 = cv2.adaptiveThreshold(th1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, constant)
    blu = cv2.GaussianBlur(th2, (5, 5), 0)
    fnois = cv2.medianBlur(th2, 3)
    return blu, fnois


def preprocess_image(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(thresh)


def correct_rotation(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
    if lines is not None:
        angles = [np.degrees(np.arctan2(y2 - y1, x2 - x1)) for x1, y1, x2, y2 in [line[0] for line in lines]]
        median_angle = np.median(angles)
        if abs(median_angle) > 45:
            median_angle -= 90
        return image.rotate(-median_angle, expand=True)
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
    config_custom = r'--oem 3 -l rus --psm 3'
    text = pytesseract.image_to_string(preprocessed, lang="rus", config=config_custom)
    return re.sub(r'\s+', ' ', text)


def extract_text_from_pdf_all(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        corrected = correct_rotation(img)
        preprocessed = preprocess_image(corrected)
        config_custom = r'--oem 3 -l rus --psm 3'
        text = pytesseract.image_to_string(preprocessed, lang="rus", config=config_custom)
        full_text += " " + re.sub(r'\s+', ' ', text)
        # Для отладки можно сохранять промежуточный текст в файл:
        with open("output.txt", "a") as f:
            f.write(text)
    return full_text.strip()


def extract_system(text, systems_list):
    def remove_remont_endings(s):
        return re.sub(r'\bремонт\w*\b', 'ремонт', s, flags=re.IGNORECASE)

    processed_text = remove_remont_endings(text)
    for system in systems_list:
        processed_system = remove_remont_endings(system)
        pattern = r'\s*'.join(re.escape(word) for word in processed_system.split())
        if processed_system.lower() in processed_text.lower():
            return system
    return "Описание системы не найдено"


def clean_text(text):
    text = re.sub(r'[^\w\s.,:-]', '', text)
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
    return re.sub(r'^(?:ДОКУМЕНТ PОДПИСАН ЭЛЕКТРОННОЙ PОДПИСЬЮ\s+)?(?:[А-ЯЁа-яё]+\s+область,\s*)', '', address,
                  flags=re.IGNORECASE).strip()


def extract_table(pdf_path, page_num, table_num):
    pdf = pdfplumber.open(pdf_path)
    table_page = pdf.pages[page_num]
    table = table_page.extract_tables()[table_num]
    return table


def table_converter(table):
    table_string = ''
    for row_num in range(len(table)):
        row = table[row_num]
        cleaned_row = [
            item.replace('\n', ' ') if item is not None and '\n' in item else 'None' if item is None else item for item
            in row]
        table_string += ('|' + '|'.join(cleaned_row) + '|' + '\n')
    table_string = table_string[:-1]
    return table_string


def extract_sro_date(text):
    patterns = [
        r'(\d{1,2}[.]\d{1,2}[.]\d{4})',  # "dd.mm.yyyy"
        r'(\d{1,2}[-]\d{1,2}[-]\d{4})'   # "dd-mm-yyyy"
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            data_str = match.group(1)
            try:
                # Заменяем "-" на "." для единообразия, если нужно
                data_str = data_str.replace("-", ".")
                return datetime.datetime.strptime(data_str, "%d.%m.%Y").date()
            except ValueError:
                return None
    return None


# Функции для обработки договора (предобработка, корректировка поворота и адрес)
def contract_preprocess_image(image):
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


def build_address_regex(input_address, max_gap=350):
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
    else:
        smeta_address = "Адрес не найден"
    smeta_system = None
    for sys in systems_list:
        if sys.lower() in all_text.lower():
            smeta_system = sys
            break
    if smeta_system is None:
        smeta_system = "Описание системы не найдено"
    print("\nСмета:")
    print("Адрес:", smeta_address)
    print("Система:", smeta_system)
    smeta_address_clean = clean_address(smeta_address)
    reference_address_clean = clean_address(reference_address)
    if reference_address and smeta_address_clean.lower() != reference_address_clean.lower():
        print("Адрес сметы не совпадает с эталонным адресом!")
        print("Смета:", smeta_address_clean)
        print("Эталон:", reference_address_clean)
        errors.append("Адрес сметы не совпадает с эталонным адресом.")
    else:
        print("Адрес сметы совпадает с эталонным адресом.")
    smetnaya_stoimost_match = re.search(r'Сметная стоимость\s+([\d.,]+)\s*тыс\.?руб', all_text, re.IGNORECASE)
    if smetnaya_stoimost_match:
        smeta_stoimost = smetnaya_stoimost_match.group(1)
        print("Сметная стоимость:", smeta_stoimost, "тыс.руб.")
    else:
        smeta_stoimost = None
        print("Сметная стоимость не найдена!")
        errors.append("Сметная стоимость не найдена!")
    osnovanie_match = re.search(r'\bОснование\b\s*[:\-]?\s*([\w\s]+)', all_text, re.IGNORECASE)
    if osnovanie_match:
        osnovanie_value = osnovanie_match.group(1).strip()
        if osnovanie_value:
            print("Основание:", osnovanie_value)
        else:
            print("Основание не найдено!")
            errors.append("Основание не найдено!")
    else:
        print("Основание не найдено!")
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
                print("Уровень цен актуален (не старше 3 месяцев).")
            else:
                print("Уровень цен устарел!")
            print("Дата уровня цен:", price_level_display)
        else:
            print("Не удалось распознать дату уровня цен в смете.")
            errors.append("Не удалось распознать уровень цен в смете.")
    else:
        print("Уровень цен не найден в смете.")
        errors.append("Уровень цен не найден в смете.")
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
    return doc_type_found if doc_type_found else "Неизвестный документ"


# -------------------------- ФУНКЦИЯ ОБРАБОТКИ ДОКУМЕНТОВ --------------------------

def process_documents(file_paths):
    """
    Функция принимает список путей к файлам, определяет тип каждого документа,
    выполняет все проверки и возвращает итоговый текстовый отчёт.
    """
    # Словарь для хранения путей к документам по их типам
    docs = {}
    for file_path in file_paths:
        doc_type = classify_document(file_path)
        docs[doc_type] = file_path
        print(f"Файл получен: {os.path.basename(file_path)}")
        print(f"Определённый тип документа: {doc_type}\n")
        log_user_activity("document_upload", f"{os.path.basename(file_path)} | Тип: {doc_type}")

    # Списки для хранения результата и ошибок
    result_lines = []
    final_errors = []

    # Извлечение данных из ключевых документов
    statement_address = None
    statement_system = None
    if "Заявление" in docs:
        statement_file = docs["Заявление"]
        st_text = extract_text_from_pdf(statement_file, skip_top_fraction=1 / 6)
        statement_system = extract_system(st_text, systems_list)
        statement_address = extract_address(st_text)
        result_lines.append("Заявление:")
        result_lines.append(f"Система: {statement_system}")
        result_lines.append(f"Адрес: {statement_address}\n")
        statement_full_text = extract_text_from_pdf_all(statement_file)
        if "реестр смет" in statement_full_text.lower():
            result_lines.append(
                "В заявлении обнаружен 'Реестр смет': адрес и система заявления совпадают с данными дефектной ведомости.\n")
    else:
        result_lines.append("Заявление не найдено.\n")
        final_errors.append("Заявление отсутствует.")

    defect_address = None
    defect_system = None
    if "Дефектная ведомость" in docs:
        defect_file = docs["Дефектная ведомость"]
        df_text = extract_text_from_pdf(defect_file)
        defect_system = extract_system(df_text, systems_list)
        defect_address = extract_address(df_text)
        df_doc = extract_doc_date_number(df_text)
        result_lines.append("Дефектная ведомость:")
        result_lines.append(f"Система: {defect_system}")
        result_lines.append(f"Адрес: {defect_address}")
        if df_doc == "Номер и дата не найдены":
            result_lines.append(
                "Не удалось корректно выгрузить дату и номер у дефектной ведомости. Будет направлена на ручную проверку.\n")
            final_errors.append("Не удалось определить номер и дату дефектной ведомости.")
        else:
            result_lines.append(f"Номер и дата: {df_doc}\n")
        found, preview = check_approval_sheet_in_document(defect_file)
        if found:
            result_lines.append("Лист согласования присутствует в дефектной ведомости.\n")
        else:
            result_lines.append("Лист согласования не найден в дефектной ведомости.\n")
            final_errors.append("Лист согласования отсутствует в дефектной ведомости.")
    else:
        result_lines.append("Дефектная ведомость не найдена.\n")
        final_errors.append("Дефектная ведомость отсутствует.")

    contract_address = None
    if "Договор" in docs:
        dog_file = docs["Договор"]
        contract_address = check_contract_address(dog_file, statement_address if statement_address else "")
        result_lines.append("Договор:")
        if contract_address:
            result_lines.append(f"Извлечённый адрес из договора: {contract_address}\n")
        else:
            result_lines.append("Адрес в договоре не найден.\n")

    # Определение эталонного адреса (приоритет: договор → дефектная ведомость → заявление)
    if contract_address:
        reference_address = contract_address
    elif defect_address:
        reference_address = defect_address
    elif statement_address:
        reference_address = statement_address
    else:
        reference_address = None
        final_errors.append(
            "Не удалось определить эталонный адрес: отсутствуют договор, дефектная ведомость и заявление.")

    # Эталонная система — из заявления
    reference_system = statement_system if statement_system else None

    # Проверки адресов и систем для всех документов
    if statement_address and reference_address and reference_address != statement_address:
        if normalize_address_for_comparison(statement_address) != normalize_address_for_comparison(reference_address):
            result_lines.append("Адрес заявления не совпадает с эталонным адресом:")
            result_lines.append(f"Заявление: {statement_address}")
            result_lines.append(f"Эталон: {reference_address}\n")
            final_errors.append("Адрес заявления не совпадает с эталонным адресом.")

    if defect_address and reference_address and reference_address != defect_address:
        if normalize_address_for_comparison(defect_address) != normalize_address_for_comparison(reference_address):
            result_lines.append("Адрес дефектной ведомости не совпадает с эталонным адресом:")
            result_lines.append(f"Дефектная ведомость: {defect_address}")
            result_lines.append(f"Эталон: {reference_address}\n")
            final_errors.append("Адрес дефектной ведомости не совпадает с эталонным адресом.")

    if "Смета" in docs:
        smeta_file = docs["Смета"]
        smeta_output = io.StringIO()
        with contextlib.redirect_stdout(smeta_output):
            smeta_errors = check_smeta(smeta_file, reference_address, reference_system)
        result_lines.append(smeta_output.getvalue())
        if smeta_errors:
            final_errors.extend(smeta_errors)
    else:
        result_lines.append("Смета не найдена.\n")
        final_errors.append("Смета отсутствует.")

    if "Акт обследования" in docs:
        act_file = docs["Акт обследования"]
        act_text = extract_text_from_pdf(act_file)
        act_system = extract_system(act_text, systems_list)
        act_address = extract_address(act_text)
        act_doc = extract_doc_date_number(act_text)
        if act_address == "Адрес не найден" and reference_address:
            act_address = reference_address
        result_lines.append("Акт обследования:")
        result_lines.append(f"Система: {act_system}")
        result_lines.append(f"Адрес: {act_address}")
        if act_doc == "Номер и дата не найдены":
            result_lines.append(
                "Не удалось корректно выгрузить дату и номер у акта обследования. Будет направлена на ручную проверку.\n")
            final_errors.append("Не удалось определить номер и дату акта обследования.")
        else:
            result_lines.append(f"Номер и дата: {act_doc}\n")
        found, preview = check_approval_sheet_in_document(act_file)
        if found:
            result_lines.append("Лист согласования присутствует в акте обследования.\n")
        else:
            result_lines.append("Лист согласования не найден в акте обследования.\n")
            final_errors.append("Лист согласования отсутствует в акте обследования.")
        if reference_address and normalize_address_for_comparison(act_address) != normalize_address_for_comparison(
                reference_address):
            result_lines.append("Адрес акта обследования не совпадает с эталонным адресом:")
            result_lines.append(f"Акт: {act_address}")
            result_lines.append(f"Эталон: {reference_address}\n")
            final_errors.append("Адрес акта обследования не совпадает с эталонным адресом.")
        if reference_system and act_system.lower() != reference_system.lower():
            result_lines.append("Система акта обследования не совпадает с эталонной системой:")
            result_lines.append(f"Акт: {act_system}")
            result_lines.append(f"Эталон: {reference_system}\n")
            final_errors.append("Система акта обследования не совпадает с эталонной системой.")
    else:
        result_lines.append("Акт обследования не найден.\n")
        final_errors.append("Акт обследования отсутствует.")

    if "Техническое задание" in docs:
        tech_file = docs["Техническое задание"]
        tech_text = extract_text_from_pdf(tech_file)
        tech_system = extract_system(tech_text, systems_list)
        tech_address = extract_address(tech_text)
        tech_doc = extract_doc_date_number(tech_text)
        if tech_address == "Адрес не найден" and reference_address:
            tech_address = reference_address
        result_lines.append("Техническое задание:")
        result_lines.append(f"Система: {tech_system}")
        result_lines.append(f"Адрес: {tech_address}")
        if tech_doc == "Номер и дата не найдены":
            result_lines.append(
                "Не удалось корректно выгрузить дату у технического задания. Будет направлена на ручную проверку.\n")
            final_errors.append("Не удалось определить дату технического задания.")
        else:
            result_lines.append(f"Дата: {tech_doc}\n")
        if reference_address and normalize_address_for_comparison(tech_address) != normalize_address_for_comparison(
                reference_address):
            result_lines.append("Адрес технического задания не совпадает с эталонным адресом:")
            result_lines.append(f"Техническое задание: {tech_address}")
            result_lines.append(f"Эталон: {reference_address}\n")
            final_errors.append("Адрес технического задания не совпадает с эталонным адресом.")
        if reference_system and tech_system.lower() != reference_system.lower():
            result_lines.append("Система технического задания не совпадает с эталонной системой:")
            result_lines.append(f"Техническое задание: {tech_system}")
            result_lines.append(f"Эталон: {reference_system}\n")
            final_errors.append("Система технического задания не совпадает с эталонной системой.")

    if "Выписка СРО" in docs:
        sro_file = docs["Выписка СРО"]
        sro_text = extract_text_from_pdf(sro_file)
        sro_date = extract_sro_date(sro_text)
        print(f"Извлечённая дата СРО: {sro_date}, тип: {type(sro_date)}")
        if sro_date:
            current_date = datetime.date.today()
            deadline_date = current_date + datetime.timedelta(days=30)
            if sro_date <= deadline_date:
                result_lines.append(f"Выписка СРО оформлена вовремя: {sro_date} (срок до {deadline_date})\n")
            else:
                result_lines.append(
                    f"Выписка СРО оформлена с опозданием: {sro_date}. Допустимый срок - до {deadline_date}\n")
        else:
            result_lines.append("Дата выписки СРО не найдена в документе\n")

    if "Пояснительная записка" in docs:
        explanatory_file = docs["Пояснительная записка"]
        expl_text = extract_text_from_pdf_all(explanatory_file)
        expl_system = extract_system(expl_text, systems_list)
        expl_address = extract_address(expl_text)
        expl_price_date, expl_price_display = extract_price_level_explanatory(expl_text)
        expl_smet_match = re.search(r'Сметная стоимость.*?([\d\s.,]+)\s*тыс\.?\s*руб', expl_text,
                                    re.IGNORECASE | re.DOTALL)
        result_lines.append("Пояснительная записка:")
        result_lines.append(f"Система: {expl_system}")
        result_lines.append(f"Адрес: {expl_address}")
        if expl_smet_match:
            expl_stoimost = expl_smet_match.group(1).replace(" ", "")
            result_lines.append(f"Сметная стоимость в пояснительной записке: {expl_stoimost} тыс. руб.")
        else:
            expl_stoimost = None
            result_lines.append("Сметная стоимость в пояснительной записке отсутствует, сравнение невозможно.")
            final_errors.append("Сметная стоимость отсутствует в пояснительной записке.")
        if reference_address and normalize_address_for_comparison(expl_address) != normalize_address_for_comparison(
                reference_address):
            result_lines.append("Адрес пояснительной записки не совпадает с эталонным адресом:")
            result_lines.append(f"Пояснительная: {expl_address}")
            result_lines.append(f"Эталон: {reference_address}\n")
            final_errors.append("Адрес пояснительной записки не совпадает с эталонным адресом.")
        if reference_system and expl_system.lower() != reference_system.lower():
            result_lines.append("Система пояснительной записки не совпадает с эталонной системой:")
            result_lines.append(f"Пояснительная: {expl_system}")
            result_lines.append(f"Эталон: {reference_system}\n")
            final_errors.append("Система пояснительной записки не совпадает с эталонной системой.")
        if expl_price_date:
            today = datetime.date.today()
            if expl_price_date >= today - relativedelta(months=3):
                result_lines.append("Уровень цен в пояснительной записке актуален (не старше 3 месяцев).\n")
            else:
                result_lines.append("Уровень цен в пояснительной записке устарел!\n")
                final_errors.append("Уровень цен в пояснительной записке устарел.")
        else:
            result_lines.append("Не удалось распознать уровень цен в пояснительной записке.\n")
    else:
        result_lines.append("Пояснительная записка не найдена.\n")
        final_errors.append("Пояснительная записка отсутствует.")

    if "Заявление" in docs and normalize_system(statement_system) in [normalize_system(x) for x in
                                                                      systems_project_required]:
        project_file = docs.get("Проект")
        conclusion_file = docs.get("Заключение")
        valid_project = False
        valid_conclusion = False
        if project_file and os.path.splitext(project_file)[1].lower() == ".pdf":
            proj_text = extract_text_from_pdf(project_file)
            if "проектная документация" in proj_text.lower():
                valid_project = True
        if conclusion_file and os.path.splitext(conclusion_file)[1].lower() == ".pdf":
            concl_text = extract_text_from_pdf(conclusion_file)
            if "заключение" in concl_text.lower():
                valid_conclusion = True
        if valid_project and valid_conclusion:
            result_lines.append("Проект и заключение присутствуют и корректны.\n")
        else:
            missing = []
            if not valid_project:
                missing.append("проект")
            if not valid_conclusion:
                missing.append("заключение")
            result_lines.append(
                "Для данной системы требуется проект и заключение, но следующие документы не найдены или некорректны: " + ", ".join(
                    missing) + ".\n")
            final_errors.append("Отсутствуют или некорректны: " + ", ".join(missing))

    # Формирование итогового отчёта
    result_lines.append("\nИтог:")
    if final_errors:
        for err in final_errors:
            result_lines.append(f"- {err}")
    else:
        result_lines.append("Ошибок не обнаружено. Все документы соответствуют требованиям.")

    final_result = "\n".join(result_lines)
    return final_result


def check_approval_sheet_in_document(pdf_path):
    full_text = extract_text_from_pdf_all(pdf_path)
    normalized_text = re.sub(r'\s+', ' ', full_text).lower()
    found = "лист согласования" in normalized_text
    return found, normalized_text[:500]


def extract_first_four_digits(cost_str):
    digits = ''.join(re.findall(r'\d', cost_str))
    return digits[:4]


# -------------------------- ОСНОВНОЙ БЛОК ВЫПОЛНЕНИЯ --------------------------

if __name__ == "__main__":
    # Здесь можно указать список файлов для отладки. Например:
    file_paths = [
        "/home/userus/PycharmProjects/OCR/uploads/тест2/уил+Раздел++ПЗ+xml_28-01-2025_10-00_967354.pdf"
    ]

    if not file_paths:
        print("Не указаны файлы для обработки.")
    else:
        output = process_documents(file_paths)
        print("\n--- Результат обработки документов ---\n")
        print(output)
