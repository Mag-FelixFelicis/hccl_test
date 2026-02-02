FROM python:3.11-slim
WORKDIR /app
COPY coordinator.py /app/coordinator.py
EXPOSE 8080
ENTRYPOINT ["python3", "/app/coordinator.py", "--host", "0.0.0.0", "--port", "8080"]
