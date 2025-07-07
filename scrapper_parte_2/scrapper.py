import requests
from bs4 import BeautifulSoup
from models import Product
from database import SessionLocal

BASE_URL = "http://books.toscrape.com/"

def mapRatingToInt(rating):
    return {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}.get(rating, 0)

def scrape_all_categories():
    db = SessionLocal()
    home = requests.get(BASE_URL)
    soup = BeautifulSoup(home.text, "html.parser")

    category_links = soup.select(".side_categories ul li ul li a")

    for link in category_links:
        category_name = link.text.strip()
        href = link["href"]
        category_url = BASE_URL + href

        scrape_category(category_url, category_name, db)

    db.commit()
    db.close()

def scrape_category(url, category, db):
    while url:
        res = requests.get(url)
        res.encoding = "UFT-8" 
        if res.status_code != 200:
            print(f"Error accediendo a {url}")
            break

        soup = BeautifulSoup(res.text, "html.parser")

        for article in soup.select(".product_pod"):
            title = article.h3.a["title"]
            price = float(article.select_one(".price_color").text.replace("£", ""))
            rating = mapRatingToInt(article.select_one(".star-rating")["class"][1])

            product = Product(title=title, price=price, category=category, rating=rating)
            db.add(product)

        next_link = soup.select_one(".next a")
        if next_link:
            parts = url.rsplit("/", 1)
            url = parts[0] + "/" + next_link["href"]
        else: break

def scrape():
    try:
        scrape_all_categories()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    scrape()
