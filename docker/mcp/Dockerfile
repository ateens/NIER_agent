FROM nvidia/cuda:12.4.0-base-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*

COPY docker/mcp/requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

COPY docker/mcp/app/ /app

ENV PYTHONUNBUFFERED=1
EXPOSE 9999
CMD ["python3", "main.py"]
