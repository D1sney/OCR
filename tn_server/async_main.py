from fastapi import FastAPI, Request, HTTPException, Form
from fastapi.responses import JSONResponse
from urllib.parse import urlparse, unquote
import requests
import os
import logging
import time
from typing import Dict, Annotated
from celery_tasks import process_documents_task

# Настройка логирования
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Вывод в консоль
        logging.FileHandler("app.log")  # Запись в файл
    ]
)
logger = logging.getLogger(__name__)

# Создание FastAPI приложения
app = FastAPI()

# Папка для загрузок
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)




# Функция для скачивания и сохранения файлов
def download_and_save_files(form: dict, upload_folder: str):
    file_paths = []
    if "file" not in form:
        logger.warning("Поле 'file' отсутствует в данных формы")
        return None, {"error": "Поле 'file' не найдено в форме"}
    task_name = form.get('task_name')
    if not task_name:
        logger.warning("Поле 'task_name' отсутствует в данных формы")
        return None, {"error": "Поле 'task_name' не найдено в форме"}
    sub_dir = os.path.join(upload_folder, task_name)
    os.makedirs(sub_dir, exist_ok=True)
    logger.debug(f"Создан подкаталог: {sub_dir}")
    file_urls = form["file"].split(',')
    logger.debug(f"Получены URL файлов: {file_urls}")
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
    # file_paths = [r'C:\Users\ivan3\Desktop\uploads\text.txt', r'C:\Users\ivan3\Desktop\uploads\text2.txt']
    return file_paths, None

# Функция для удаления некорректных файлов
def delete_incorrect_files(incorrect_files: list, form: dict, upload_folder: str) -> None:
    task_name = form.get('task_name')
    if not task_name:
        logger.warning("Поле 'task_name' отсутствует в данных формы")
        return
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


# Главный маршрут для проверки
@app.get("/")
def home():
    return {"message": "server_work"}

# Маршрут для загрузки файлов
@app.post("/upload",
    summary="Upload file links",
    description="Принимает список URL файлов (разделённых запятыми), название задачи и ID задачи для обработки в Celery.",
    response_description="Подтверждение запуска задачи с её ID."
)
async def upload_file(
    file: Annotated[str, Form(description="Список URL файлов, разделённых запятыми (например, http://example.com/file1.pdf,http://example.com/file2.pdf)")],
    task_name: Annotated[str, Form(description="Название задачи для создания подкаталога")],
    task_id: Annotated[str, Form(description="ID задачи")]
) -> Dict[str, str]:  # Указываем тип возвращаемого значения
    form = {"file": file, "task_name": task_name, "task_id": task_id}
    logger.debug(f"Form data: {form}")
    file_paths, error = download_and_save_files(form, UPLOAD_FOLDER)
    if error:
        logger.error(f"Ошибка при загрузке файлов: {error}")
        raise HTTPException(status_code=400, detail=error)
    logger.info(f"Файлы успешно загружены: {file_paths}")
    # Запускаем задачу Celery

    celery_task = process_documents_task.delay(file_paths, task_id)
    return JSONResponse({"message": "Задача запущена", "task_id": celery_task.id})

# Маршрут для получения результатов обработки
@app.get("/result/{task_id}")
def get_result(task_id: str):
    task = process_documents_task.AsyncResult(task_id)
    if task.state == 'PENDING':
        return {"status": "pending"}
    elif task.state == 'SUCCESS':
        final_result, final_errors = task.result
        return {"status": "success", "result": final_result, "errors": final_errors}
    else:
        return {"status": "failure"}
