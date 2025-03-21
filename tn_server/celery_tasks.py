import time
from celery import Celery
import logging
import requests
from Bot123_copy import process_documents

# Настройка логирования
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("app.log")]
)
logger = logging.getLogger(__name__)

# Настройка Celery: здесь мы подключаемся к Redis,
# но обратите внимание на адрес брокера — он будет доступен
# по имени сервиса в Docker-сети (в нашем случае "redis")
celery = Celery(
    "worker",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/1"
)

# Функция для отправки комментария
def send_comment(task_id, final_result):
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

# Функция для отправки чек-листа
def send_check_list(task_id, final_errors):
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

# Определение задачи Celery
@celery.task(bind=True)
def process_documents_task(self, file_paths, task_id):
    celery_task_id = self.request.id
    # Здесь должна быть ваша логика обработки документов из process_documents
    final_result, final_errors = process_documents(file_paths)
    logger.info(f"Обработка завершена для {celery_task_id}, результат: {final_result}")
    # Отправка комментария
    send_comment(task_id, final_result.append)
    logger.info(f"Комментарий отправлен для {celery_task_id}")
    # Отправка чек-листов
    for final_err in final_errors:
        send_check_list(task_id, final_err)
        time.sleep(1)
    logger.info(f"Чек-листы отправлены для {celery_task_id}, Чек-листы: {final_errors}")
    return final_result, final_errors