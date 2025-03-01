FROM python:3.9-slim

# Install dependencies
RUN pip install requests pandas google-api-python-client

# Copy file ke dalam container
WORKDIR /app
COPY scrape_youtube_api.py /app/scrape_youtube_api.py

# Jalankan skrip
CMD ["python", "/app/scrape_youtube_api.py"]