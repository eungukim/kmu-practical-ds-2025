FROM python:3.13-slim
WORKDIR /app
COPY docker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ .
RUN apt-get update && apt-get install -y openjdk-17-jre
EXPOSE 8000 54321
CMD ["python", "server.py"]