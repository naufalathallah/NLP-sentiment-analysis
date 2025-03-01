# Gunakan base image Python yang lebih ringan
FROM python:3.9-slim

# Set lingkungan agar tidak ada buffering pada log
ENV PYTHONUNBUFFERED=1

# Install dependencies
RUN pip install --no-cache-dir requests pandas google-api-python-client

# Tentukan direktori kerja
WORKDIR /app

# Copy semua file ke dalam container
COPY scrape_youtube_api.py /app/scrape_youtube_api.py

# Jalankan skrip
CMD ["python", "/app/scrape_youtube_api.py"]