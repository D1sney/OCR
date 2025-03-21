import fitz
import pdfplumber
from PIL import Image, ImageEnhance, ImageOps
import pytesseract
import cv2
import numpy as np
import imutils
from imutils.contours import sort_contours


# Set the path to your Tesseract executable if not in default location
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# cv2.blur(image, (1,1))

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

def extract_stamp(image):
    (H, W) = image.shape
    # initialize a rectangular and circle structuring kernel
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    gray = cv2.GaussianBlur(image, (3, 3), 0)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)

    grad = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    grad = np.absolute(grad)
    (min_v, max_v) = (np.min(grad), np.max(grad))
    grad = (grad - min_v) / (max_v - min_v)
    grad = (grad * 255).astype('uint8')
    kernel = np.ones((3, 3), np.uint8)

    grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, rectKernel)
    thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
    thresh = cv2.erode(thresh, kernel, iterations=2)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sort_contours(cnts, method='bottom-to-top')[0]
    mrzBox = None

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        percentWidth = w / float(W)
        percentHeight = h / float(H)
        if percentWidth > 0.8 and percentHeight > 0.04:
            mrzBox = (x, y, w, h)
            break

    (x, y, w, h) = mrzBox
    pX = int((x + w) * 0.03)
    pY = int((y + h) * 0.03)
    (x, y) = (x - pX, y - pY)
    (w, h) = (w + (pX * 2), h + (pY * 2))

    mrz = image[y:y + h, x:x + w]
    return mrz
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

'''def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def remove_noise(image):
    return cv2.medianBlur(image, 5)

def thresholding(image):
    return cv2.threshold(sharpen(image), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

def erode(image):
    kernel = np.ones((5,5), np.uint8)
    return cv2.erode(image, kernel, iterations=1)

def opening(image):
    kernel = np.ones((5,5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

def canny(image):
    return cv2.Canny(image, 100, 200)

def sharpen(image):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(get_grayscale(image), -1, kernel)

def binarize(image):
    thresh = cv2.adaptiveThreshold(image,
                                   255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    return thresh

def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)'''

# Initialize PyMuPDF
pdf_path = '/home/userus/PycharmProjects/OCR/uploads/тест2/ТЗ+Калинина+28+кр_939266_967354_06-03-2025_17-22.pdf'

pdf_document = fitz.open(pdf_path)

# Extract text using OCR
extracted_text = ""

for page_num in range(len(pdf_document)):
    page = pdf_document.load_page(page_num)  # Load each page

    pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))

    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # Correct rotation if necessary
    corrected_img = correct_rotation(img)
    preprocessed_img = preprocess_image(corrected_img)
    config_custom = r'--oem 3 -l rus --psm 3'


    text = pytesseract.image_to_string(preprocessed_img, lang='rus', config=config_custom)

    extracted_text += text + "\n"  # Append the recognized text

print(extracted_text)

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

'''def image_smoothen(img):

    ret1, th1 = cv2.threshold(img, 88, 255, cv2.THRESH_BINARY)

    ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    blur = cv2.GaussianBlur(th2, (5, 5), 0)

    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return th3

def remove_noise_and_smooth(pdf_path):
    # Read the image from the specified file path in grayscale mode
    img = cv2.imread(pdf_path, 0)

    filtered = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 41)

    kernel = np.ones((1, 1), np.unit8)

    opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)

    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    img = image_smoothen(img)

    or_image = cv2.bitwise_or(img, closing)

    return or_image'''

with open('extracted_text.txt', 'w', encoding='utf-8') as f:
    f.write(extracted_text)