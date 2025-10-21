FROM langflowai/langflow:1.5.1

WORKDIR /app

COPY docker/langflow/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY docker/langflow/flows/ /app/flows/
COPY docker/langflow/custom_nodes/ /app/custom_nodes/

ENV PYTHONPATH=/app
CMD ["langflow", "run", "--host", "0.0.0.0", "--port", "7860"]
