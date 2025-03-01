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
video_url = "https://www.youtube.com/watch?v=6Dh-RL__uN4"  # Video dengan banyak komentar
driver.get(video_url)
time.sleep(5)  # Tunggu halaman termuat

# Scroll ke bawah untuk memuat lebih banyak komentar
scroll_pause_time = 4  # Ubah menjadi 4 detik agar komentar termuat lebih banyak
last_height = driver.execute_script("return document.documentElement.scrollHeight")

# Set batas maksimal komentar yang diambil
MAX_COMMENTS = 3000  
comment_list = []  # Pakai list agar lebih fleksibel dan bisa di-print

print("🚀 Memulai scraping komentar YouTube...")

while len(comment_list) < MAX_COMMENTS:
    # Scroll ke bawah
    driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
    time.sleep(scroll_pause_time)

    # Scroll ke atas sedikit untuk memicu loading baru
    driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight - 200);")
    time.sleep(1)

    # Cek apakah ada perubahan setelah scroll
    new_height = driver.execute_script("return document.documentElement.scrollHeight")
    if new_height == last_height:
        print("⚠️ Tidak ada komentar baru yang dimuat. Berhenti scrolling.")
        break  # Jika tidak ada perubahan, berarti tidak ada komentar baru yang dimuat
    last_height = new_height

    # Ambil komentar dari elemen YouTube
    comment_elements = driver.find_elements(By.CSS_SELECTOR, "#content-text")

    for comment in comment_elements:
        text = comment.text.strip()
        if text and text not in comment_list:  # Hindari komentar kosong & duplikat
            comment_list.append(text)
        if len(comment_list) >= MAX_COMMENTS:
            break

    print(f"📩 Komentar terkumpul: {len(comment_list)} / {MAX_COMMENTS}")  # Print progress

# Simpan ke file CSV
df = pd.DataFrame(comment_list[:MAX_COMMENTS], columns=["Comment"])

# Buat format timestamp
timestamp = time.strftime("%Y%m%d_%H%M%S")

# Simpan file dengan format youtube_comments-YYYYMMDD_HHMMSS.csv
filename = f"/app/youtube_comments-{timestamp}.csv"
df.to_csv(filename, index=False, encoding="utf-8")

print(f"✅ Scraping selesai! Berhasil menyimpan {len(comment_list)} komentar di {filename}")

# Tutup WebDriver
driver.quit()