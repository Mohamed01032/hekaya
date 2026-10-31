FROM python:3.12-slim
WORKDIR /srv
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt
COPY data data
COPY frontend frontend
COPY backend/app backend/app
WORKDIR /srv/backend
ENV HEKAYA_DB=/tmp/hekaya.sqlite3
EXPOSE 8000
# Hosting platforms (Render, Railway, Fly) pass the port in $PORT; locally it defaults to 8000.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
