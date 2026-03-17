FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY src/ src/

# Volume mount point for persistent state
VOLUME /data

ENV SQLITE_PATH=/data/state.db

ENTRYPOINT ["python", "main.py"]
