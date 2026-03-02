FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
COPY evaluate.py .

RUN pip install -r requirements.txt

CMD ["python", "evaluate.py"]