FROM python:3.11-slim

WORKDIR /app

COPY docker/chromadb/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY docker/chromadb/chroma_db/ /app/chroma_db/

EXPOSE 8000
CMD ["chroma", "run", "--path", "/app/chroma_db"]
