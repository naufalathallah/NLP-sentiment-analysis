import requests
import pandas as pd
import time

# 🔑 API Key dari Google Cloud
API_KEY = "YOUR_YOUTUBE_API_KEY"  # Ganti dengan API Key Anda

# 🆔 ID Video YouTube yang ingin di-scrape
VIDEO_ID = "6Dh-RL__uN4"

# URL API untuk mendapatkan komentar
API_URL = f"https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId={VIDEO_ID}&maxResults=100&key={API_KEY}"

# Menyimpan komentar
comment_list = []

# Fungsi untuk mengambil komentar
def fetch_comments(api_url, max_comments=3000):
    next_page_token = None
    total_comments = 0

    while total_comments < max_comments:
        # Tambahkan token halaman berikutnya jika ada
        if next_page_token:
            url = f"{api_url}&pageToken={next_page_token}"
        else:
            url = api_url

        response = requests.get(url)
        data = response.json()

        # Jika API limit terlampaui, tunggu dan coba lagi
        if "error" in data:
            print(f"⚠️ Error: {data['error']['message']}")
            if "quotaExceeded" in data["error"]["message"]:
                print("⏳ API Quota habis. Coba lagi nanti.")
                break
            time.sleep(5)
            continue

        # Ambil komentar dari respons API
        for item in data.get("items", []):
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comment_list.append(comment)
            total_comments += 1

            if total_comments >= max_comments:
                break

        print(f"📩 Komentar terkumpul: {total_comments} / {max_comments}")

        # Periksa apakah masih ada halaman berikutnya
        next_page_token = data.get("nextPageToken")
        if not next_page_token:
            print("✅ Semua komentar telah diambil.")
            break

        # Hindari batasan API dengan menunggu sebentar sebelum request berikutnya
        time.sleep(1)

# Mulai scraping
print("🚀 Mengambil komentar dari YouTube API...")
fetch_comments(API_URL, max_comments=3000)

# Simpan ke file CSV
df = pd.DataFrame(comment_list, columns=["Comment"])

# Buat format timestamp
timestamp = time.strftime("%Y%m%d_%H%M%S")

# Simpan file dengan format youtube_comments-YYYYMMDD_HHMMSS.csv
filename = f"youtube_comments-{timestamp}.csv"
df.to_csv(filename, index=False, encoding="utf-8")

print(f"✅ Scraping selesai! Berhasil menyimpan {len(comment_list)} komentar di {filename}")