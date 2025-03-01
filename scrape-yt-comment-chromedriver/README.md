docker build -t youtube-scraper .

docker run --rm -v $(pwd):/app youtube-scraper
