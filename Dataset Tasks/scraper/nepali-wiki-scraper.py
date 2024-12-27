##Used to scrape the 10000 Nepali wikipedia articles. 

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
from bs4 import BeautifulSoup
import time
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from collections import deque

chrome_options = Options()
# chrome_options.add_argument("--headless")  # Commented out to display the browser
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
# Optional: Set window size for better visibility
chrome_options.add_argument("window-size=1200,600")

service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)

wiki_url = "https://ne.wikipedia.org/wiki/Main_Page"  

try:
    driver.get(wiki_url)

    # Wait until the page is fully loaded
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.TAG_NAME, "a"))
    )

    def is_nepali(text):
        for char in text:
            if not ('\u0900' <= char <= '\u097F' or char.isspace() or char in ['।', '॥', ',', '.', '!', '?', '॥']):
                return False
        return len(text) > 0 

    # Extract initial set of unique Nepali links from the main page
    a_tags = driver.find_elements(By.TAG_NAME, "a")
    links = []
    for a_tag in a_tags:
        link = a_tag.get_attribute('href')
        text = a_tag.text.strip()

        if link and 'ne.wikipedia.org/wiki/' in link and is_nepali(text):
            if link not in links:
                links.append(link)

    max_articles = 10000 
    data = []
    visited_links = set(links)  
    queue = deque(links)
    serial_no = 1

    while queue and len(data) < max_articles:
        current_link = queue.popleft() # this is the implementation of a bfs , webcrawler to be precise. 

        try:
            driver.get(current_link)

            # Wait until the content is loaded
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "p"))
            )

            soup = BeautifulSoup(driver.page_source, 'html.parser')
            paragraphs = soup.find_all('p')
            article_text = " ".join([p.get_text(strip=True) for p in paragraphs if is_nepali(p.get_text(strip=True))]) # this is to ensure that we are only getting nepali text.

            if article_text:  
                data.append({
                    "Id": serial_no,
                    "Article Link": current_link,
                    "Text": article_text
                })
                serial_no += 1

            new_a_tags = soup.find_all('a', href=True)
            for a in new_a_tags:
                href = a['href']
                if href.startswith("/wiki/") and not any(prefix in href for prefix in [":", "#"]):
                    full_link = "https://ne.wikipedia.org" + href
                    # Check if the link hasn't been visited and is not already in the queue
                    if full_link not in visited_links:
                        # Optionally, add a condition to filter links based on title or other criteria
                        visited_links.add(full_link)
                        queue.append(full_link)

        except Exception as e:
            print(f"Error processing {current_link}: {e}")
            continue

    df = pd.DataFrame(data)

    #need to update this path. this file is inside "data" folder rn.
    df.to_csv("nepali_wikipedia_articles.csv", index=False, encoding='utf-8')

    print(f"Scraping completed successfully. {len(data)} articles saved.")

except Exception as main_e:
    print(f"An error occurred: {main_e}")

finally:
    driver.quit()
