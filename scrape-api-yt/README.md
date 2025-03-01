docker build -t youtube-scraper-api .

docker run --rm -v $(pwd):/app youtube-scraper-api
