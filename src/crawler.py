import scrapy
from scrapy.crawler import CrawlerProcess

class ContentSpider(scrapy.Spider):
    name = "content_spider"
    start_urls = [
        'https://example.com',  # Replace with target websites
    ]

    def parse(self, response):
        for article in response.css('article'):
            yield {
                'title': article.css('h2::text').get(),
                'content': article.css('p::text').getall(),
                'url': response.url,
            }

if __name__ == "__main__":
    process = CrawlerProcess(settings={
        'FEEDS': {
            'data/raw/crawled_data.json': {'format': 'json'},
        },
    })
    process.crawl(ContentSpider)
    process.start()
