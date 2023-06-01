FROM python:3.10-slim

WORKDIR /code

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    HNSWLIB_NO_NATIVE=1

RUN apt-get update && apt install python3-dev build-essential -y

COPY . .
RUN pip install --upgrade pip
RUN pip install duckdb
RUN pip install -r requirements.txt
EXPOSE 8071
CMD ["gradio", "app.py"]