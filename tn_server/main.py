from flask import Flask, request, jsonify
from urllib.parse import urlparse, unquote
import requests
import os
# from bot_for_tn import document_analysis
import logging
from Bot123_copy import process_documents
import time

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Вывод в консоль
        logging.FileHandler("app.log")  # Запись в файл
    ]
)
logger = logging.getLogger(__name__)


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 30 * 1024 * 1024  # 30 МБ

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Создаем папку, если ее нет


# Главный эндпоинт для проверки
@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "server_work"}), 200


def download_and_save_files(form, upload_folder: str):
    file_paths = []
    # Проверяем наличие ключа "file" в данных формы
    if "file" not in form:
        logger.warning("Поле 'file' отсутствует в данных формы")
        return None, {"error": "Поле 'file' не найдено в форме"}
    # Извлекаем task_name и создаём подкаталог
    task_name = form.get('task_name')
    if not task_name:
        logger.warning("Поле 'task_name' отсутствует в данных формы")
        return None, {"error": "Поле 'task_name' не найдено в форме"}
    sub_dir = os.path.join(upload_folder, task_name)
    os.makedirs(sub_dir, exist_ok=True)
    logger.debug(f"Создан подкаталог: {sub_dir}")
    # Разделяем файлы, если их больше одного
    file_urls = form["file"].split(',')
    logger.debug(f"Получены URL файлов: {file_urls}")
    # Скачиваем файл по URL
    for file_url in file_urls:
        try:
            response = requests.get(file_url, stream=True)
            if response.status_code != 200:
                logger.error(f"Не удалось скачать файл по URL: {file_url}, статус: {response.status_code}")
                return None, {"error": f"Не удалось скачать файл по URL: {file_url}"}
            parsed_url = urlparse(file_url)
            file_name = os.path.basename(parsed_url.path)
            file_name = unquote(file_name)
            reconciling_path = os.path.join(sub_dir, file_name)
            file_paths.append(reconciling_path)
            logger.info(f"Скачивание файла: {file_name} по URL: {file_url}")
            with open(reconciling_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            logger.info(f"Файл успешно сохранен: {reconciling_path}")
        except requests.RequestException as e:
            logger.error(f"Ошибка при скачивании файла {file_url}: {str(e)}")
            return None, {"error": f"Ошибка при скачивании файла {file_url}"}
    return file_paths, None


def delete_incorrect_files(incorrect_files: list, form, upload_folder: str) -> None:
    task_name = form.get('task_name')
    if not task_name:
        logger.warning("Поле 'task_name' отсутствует в данных формы")
        return None, {"error": "Поле 'task_name' не найдено в форме"}
    sub_dir = os.path.join(upload_folder, task_name)
    for incorrect_file_name in incorrect_files:
        incorrect_file_path = os.path.join(sub_dir, incorrect_file_name)
        if os.path.exists(incorrect_file_path):
            try:
                os.remove(incorrect_file_path)
                logger.info(f"Удален некорректный файл: {incorrect_file_path}")
            except OSError as e:
                logger.error(f"Ошибка при удалении файла {incorrect_file_path}: {str(e)}")
        else:
            logger.debug(f"Файл {incorrect_file_path} не существует, пропускаем")


def send_comment(task_id, final_result):
    # url_n8n_webhook = "https://main.utnkr.space/webhook-test/dd69b43f-f1b2-4f93-abcd-ed2d1577f44c"
    url_tn_comment_webhook = "https://tehnadzor.utnkr.ru/webhook/post/9zcy-05r2-39z4-npcv"
    data = {
        "id": task_id,
        "comment": final_result
    }
    try:
        response = requests.post(url_tn_comment_webhook, data=data)
        logger.info(f"Запрос на отправку комментария отправлен на вебхук, статус: {response.status_code}")
        return response
    except requests.RequestException as e:
        logger.error(f"Ошибка при отправке комментария: {str(e)}")
        return None

def send_check_list(task_id, final_errors):
    # url_n8n_webhook = "https://main.utnkr.space/webhook-test/dd69b43f-f1b2-4f93-abcd-ed2d1577f44c"
    url_tn_check_list_webhook = "https://tehnadzor.utnkr.ru/webhook/post/hw0r-bqvd-i0er-aazd"
    data = {
        "id": task_id,
        "check_list": final_errors
    }
    try:
        response = requests.post(url_tn_check_list_webhook, data=data)
        logger.info(f"Запрос на отправку чек-листа отправлен на вебхук, статус: {response.status_code}")
        return response
    except requests.RequestException as e:
        logger.error(f"Ошибка при отправке чек-листа: {str(e)}")
        return None


@app.route("/upload", methods=["POST"])
def upload_file():
    file_paths = []
    if "file" not in request.form:
        logger.warning("Поле 'file' отсутствует в запросе")
    # Отладочная информация
    logger.debug(f"Headers: {request.headers}")
    logger.debug(f"Raw data: {request.get_data()}")
    logger.debug(f"Form: {request.form}") # файлы приходят именно в form, а не в files
    logger.debug(f"Files: {request.files}")
    # Вызываем функцию для загрузки файлов
    file_paths, error = download_and_save_files(request.form, UPLOAD_FOLDER)
    if error:
        logger.error(f"Ошибка при загрузке файлов: {error}")
        return jsonify(error), 400
    
    logger.info(f"Файлы успешно загружены: {file_paths}")
    final_result, final_errors = process_documents(file_paths)
    # incorrect_files = ['ТЗ+Западная+10_0001_963398.pdf', 'АО+Климовск+ул.+Западная+д.+10_0001.pdf']
    # delete_incorrect_files(incorrect_files, request.form, UPLOAD_FOLDER)
    task_id = request.form.get('id')
    # Отправка комментария
    send_comment(task_id, final_result)
    # Отправка чек-листов
    for final_err in final_errors:
        send_check_list(task_id, final_err)
        time.sleep(1)

    return jsonify({"message": "Файлы загружены", "file_paths": file_paths})


if __name__ == "__main__":
    # Убедитесь, что папка для загрузок существует
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(host="0.0.0.0", port=5000, debug=True)
