# Используем официальный образ Python
FROM python:3.9-slim

# Устанавливаем зависимости: celery и redis (не устанавливаем, они уже есть в requirements.txt, ниже)

# Указываем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем файл зависимостей в контейнер
COPY requirements.txt /app/requirements.txt

# скачиваем библиотеку для CV из-за ошибки ImportError: libGL.so.1: cannot open shared object file: No such file or directory
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Обновляем pip и устанавливаем зависимости из requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Здесь мы НЕ копируем файлы из хоста, т.к. будем монтировать их томом
# Если не планируете монтировать, можете раскомментировать:
# COPY . /app

# По умолчанию запускаем shell, чтобы можно было переопределить команду запуска.
CMD ["sh"]