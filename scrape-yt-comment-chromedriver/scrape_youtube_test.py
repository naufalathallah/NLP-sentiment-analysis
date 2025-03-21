import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By

# Setup ChromeOptions untuk Docker
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--headless")  # Wajib di Docker karena tanpa GUI
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

# Inisialisasi WebDriver untuk Docker (pakai ChromeDriver bawaan image)
service = Service("/usr/bin/chromedriver")  
driver = webdriver.Chrome(service=service, options=chrome_options)

# URL Video YouTube yang ingin di-scrape
video_url = "https://www.youtube.com/watch?v=jDKE5f6_T5w"
driver.get(video_url)
time.sleep(5)  # Tunggu halaman termuat

# Scroll ke bawah untuk memuat lebih banyak komentar
scroll_pause_time = 2
last_height = driver.execute_script("return document.documentElement.scrollHeight")

while True:
    driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
    time.sleep(scroll_pause_time)
    
    new_height = driver.execute_script("return document.documentElement.scrollHeight")
    if new_height == last_height:
        break
    last_height = new_height

# Ambil komentar dari elemen YouTube
comments = []
comment_elements = driver.find_elements(By.CSS_SELECTOR, "#content-text")

for comment in comment_elements:
    comments.append(comment.text)

# Simpan ke file CSV
df = pd.DataFrame(comments, columns=["Comment"])
df.to_csv("/app/youtube_comments.csv", index=False, encoding="utf-8")

print(f"Scraping selesai! Berhasil menyimpan {len(comments)} komentar di youtube_comments.csv")

# Tutup WebDriver
driver.quit()