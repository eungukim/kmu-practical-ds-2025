FROM python:3.12-slim

WORKDIR /app

# Install necessary build tools for C++ dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/results /app/reports

CMD ["python", "movie_review_categorizer.py"] 