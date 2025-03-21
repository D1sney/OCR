import fitz
from PIL import Image, ImageEnhance, ImageOps
import pytesseract
import cv2
import numpy as np


# Если Tesseract не в системном пути, раскомментируйте и укажите путь
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

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


# Путь к PDF файлу
pdf_path = '/home/userus/PycharmProjects/OCR/temp/7567660873_ДВ Октябрьская 4 цо.pdf'
pdf_document = fitz.open(pdf_path)

extracted_text = ""

for page_num in range(len(pdf_document)):
    page = pdf_document.load_page(page_num)  # Загружаем каждую страницу
    pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # Корректировка поворота
    corrected_img = correct_rotation(img)

    # Увеличиваем контраст
    enhancer = ImageEnhance.Contrast(corrected_img)
    contrast_img = enhancer.enhance(2.0)  # Коэффициент 1.5 можно менять по необходимости

    # Предобработка изображения для OCR
    preprocessed_img = preprocess_image(contrast_img)

    # Вывод изображения (открывается в стандартном просмотрщике изображений)
    #preprocessed_img.show()

    config_custom = r'--oem 3 -l rus --psm 6'
    text = pytesseract.image_to_string(preprocessed_img, lang="rus", config=config_custom)
    extracted_text += text + "\n"

print(extracted_text)

with open('extracted_text.txt', 'w', encoding='utf-8') as f:
    f.write(extracted_text)
