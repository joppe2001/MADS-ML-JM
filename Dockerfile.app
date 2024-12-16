FROM python:3.11-slim-buster

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

ENV PYTHONPATH=/app

CMD ["python", "src/main.py"]
