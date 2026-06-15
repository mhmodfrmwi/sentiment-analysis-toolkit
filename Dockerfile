FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src:/app

COPY pyproject.toml requirements.txt README.md ./
COPY src ./src
COPY api ./api
COPY app ./app
COPY main.py ./

RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["python", "main.py", "api", "--host", "0.0.0.0", "--port", "8000"]
