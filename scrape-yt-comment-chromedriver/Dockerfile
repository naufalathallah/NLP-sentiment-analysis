# Gunakan image resmi Selenium dengan Chrome untuk Mac M1 (ARM64)
FROM seleniarm/standalone-chromium:latest

# Install Python & dependensi
USER root
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip

# Install Selenium dan Pandas dengan opsi --break-system-packages
RUN pip3 install --break-system-packages selenium pandas

# Copy file Python ke dalam container
WORKDIR /app
COPY scrape_youtube.py /app/scrape_youtube.py

# Jalankan script saat container dimulai
CMD ["python3", "/app/scrape_youtube.py"]