FROM python:3.10-slim
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY services/frontend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -U pip && pip install --no-cache-dir -r /app/requirements.txt
COPY services/frontend/app.py /app/app.py

ENV API_URL="http://localhost:7860" \
    GRADIO_SERVER_NAME="0.0.0.0" \
    GRADIO_SERVER_PORT="7861"

EXPOSE 7861
CMD ["python", "/app/app.py"]
