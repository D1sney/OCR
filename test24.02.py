import fitz
from PIL import Image, ImageEnhance
import pytesseract
import cv2
import numpy as np
import re


# Функция предобработки изображения
def preprocess_image(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(thresh)


# Функция коррекции поворота изображения
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


# Функция для создания регулярного выражения из адреса
def build_address_regex(input_address, max_gap=350):
    # Разделяем адрес на части по запятым
    parts = re.split(r'\s*,\s*', input_address.strip())
    token_list = []
    for part in parts:
        # Извлекаем ключевые слова (токены): буквы и цифры
        tokens = re.findall(r'[а-яА-Я0-9-]+', part)
        token_list.extend(tokens)

    # Формируем гибкий паттерн
    regex_parts = [re.escape(token_list[0])]  # Первый токен без \b
    for i, token in enumerate(token_list[1:]):
        # Если токен — число, и перед ним "д" или "к", допускаем точку или запятую
        if token.isdigit() and token_list[i].lower() in ['д', 'к']:
            separator = r'(?:[.,]?\s*)'
        else:
            separator = r'(?:[.,]?\s+)'
        # Допускаем любые символы между токенами
        gap = r'[\s\S]{0,' + str(max_gap) + r'}?'
        regex_parts.append(separator + gap + re.escape(token))

    pattern = ''.join(regex_parts)
    return pattern, token_list


# Функция поиска адреса в тексте
def find_address_in_text(ocr_text, input_address, max_gap=350):
    pattern, tokens = build_address_regex(input_address, max_gap)
    # Нормализуем текст: заменяем переносы строк на пробелы, убираем лишние пробелы
    normalized_text = re.sub(r'\s+', ' ', ocr_text.replace('\n', ' '))
    matches = list(re.finditer(pattern, normalized_text, flags=re.IGNORECASE))
    if matches:
        # Выбираем наиболее короткое совпадение (меньше лишних символов)
        best_match = min(matches, key=lambda m: len(m.group(0)))
        return best_match.group(0), tokens
    return None, tokens




def clean_address(found_text, tokens, input_address):
    """
    Извлекает из найденного OCR адреса только те фрагменты, которые соответствуют каноническим токенам,
    оставляя между ними только пробелы и допустимые знаки препинания.
    Если между токенами находится «шум» (слова, содержащие буквы или цифры), такой фрагмент заменяется одним пробелом.
    Если не удалось найти хотя бы один токен, возвращаем input_address.
    """
    cleaned = ""
    current_pos = 0
    prev_end = 0

    # Для каждого токена ищем его появление в найденном тексте по порядку
    for token in tokens:
        # Ищем токен, начиная с текущей позиции (игнорируем регистр)
        match = re.search(re.escape(token), found_text[current_pos:], flags=re.IGNORECASE)
        if match:
            start = current_pos + match.start()
            end = current_pos + match.end()
            # Если это первый токен, просто добавляем его
            if not cleaned:
                cleaned += found_text[start:end]
            else:
                # Интервал между предыдущим токеном и текущим
                inter = found_text[prev_end:start]
                # Если между токенами содержатся только пробелы и знаки препинания, оставляем их,
                # иначе заменяем на один пробел.
                if re.fullmatch(r'[\s,.\-:;]*', inter):
                    cleaned += inter + found_text[start:end]
                else:
                    cleaned += " " + found_text[start:end]
            prev_end = end
            current_pos = end
        else:
            # Если хотя бы один токен не найден, можно вернуть исходный адрес (или продолжить, как вам удобнее)
            return input_address
    return cleaned.strip()

def add_dot_if_missing(address):
    """
    Если после слова "д" (дом) не стоит точка перед числом, добавляет её.
    Пример: "д 15" -> "д.15"
    """
    # Ищем слово 'д' (без точки) перед числом и добавляем точку, если её нет
    fixed = re.sub(r'\bд(?!\.)\s*(\d+)', r'д.\1', address, flags=re.IGNORECASE)
    return fixed

def add_space_after_house(address):
    """
    Если после 'д.' сразу следует цифра без пробела, добавляет пробел.
    Например, преобразует 'д.15' в 'д. 15'.
    """
    return re.sub(r'\b(д\.)\s*(\d)', r'\1 \2', address, flags=re.IGNORECASE)

# Основной код
if __name__ == '__main__':
    # Открываем PDF и извлекаем текст
    pdf_path = '/home/userus/PycharmProjects/OCR/uploads/тест2/Договор+2406-К.docx.pdf'
    pdf_document = fitz.open(pdf_path)
    extracted_text = ""
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        scale = 4  # Увеличиваем разрешение
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        resized_img = img.resize((int(img.width * 1.5), int(img.height * 1.5)), Image.LANCZOS)
        corrected_img = correct_rotation(resized_img)
        enhancer = ImageEnhance.Contrast(corrected_img)
        contrast_img = enhancer.enhance(2.0)
        preprocessed_img = preprocess_image(contrast_img)
        config_custom = r'--oem 3 -l rus --psm 3'  # Настройки OCR для русского языка
        text = pytesseract.image_to_string(preprocessed_img, lang="rus", config=config_custom)
        extracted_text += text + "\n"

    # Сохраняем распознанный текст в файл для проверки
    with open('extracted_text.txt', 'w', encoding='utf-8') as f:
        f.write(extracted_text)

    # Пример адреса для поиска
    input_address = "г. Серпухов ул Калинина д 28"
    pattern, tokens = build_address_regex(input_address)
    print("Сформированное регулярное выражение:", pattern)

    result, tokens = find_address_in_text(extracted_text, input_address)
    if result:
        cleaned_address = clean_address(result, tokens, input_address)
        cleaned_address = add_dot_if_missing(cleaned_address)
        cleaned_address = add_space_after_house(cleaned_address)
        print("\nНайден адрес:", cleaned_address)
    else:
        print("\nАдрес не найден.")
