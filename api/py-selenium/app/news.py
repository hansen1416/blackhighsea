import csv
import logging
import random
import re
import sys

# from bs4 import BeautifulSoup
import psycopg2
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains

from useragent import UserAgents

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

def get_selenium_driver():

    user_agent = 'user-agent=' + UserAgents[random.randint(0, len(UserAgents)-1)]

    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')

    chrome_options.add_argument('--incognito')
    chrome_options.add_argument('--disable-extensions')
    chrome_options.add_argument('start-maximized')
    chrome_options.add_argument('disable-infobars')
    chrome_options.add_argument(user_agent)

    driver = webdriver.Chrome('chromedriver', options=chrome_options)

    driver.set_page_load_timeout(10)
    return driver

class at_least_all_present(object):
    def __init__(self, locator, n):
      self.locator = locator
      self.n = n

    def __call__(self, driver):
      elements = driver.find_elements(*self.locator)
      if len(elements) >= self.n:
        return elements
      else:
        return False

def get_news_east_money(ticker, page):

    url = "http://so.eastmoney.com/news/s?keyword={ticker}&pageindex={page}".format(ticker=ticker,page=page)

    driver = get_selenium_driver()

    driver.get(url)
    
    news_items = WebDriverWait(driver, 8).until(at_least_all_present((By.CSS_SELECTOR, '.news-item'), 3))

    i = 0
    values = []

    while i < len(news_items):

        try:
            title = news_items[i].find_element_by_css_selector('h3 a').text
            desc = news_items[i].find_element_by_css_selector('.news-desc').text

            searched_date = re.search(r'^(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})', desc)

            date_index = searched_date.group(1)

            desc = desc.replace(date_index + ' - ', '')

            link = news_items[i].find_element_by_css_selector('.link a').text

            values.append("('" + "', '".join([str(date_index), str(ticker), title, desc, link, 'east_money']) + "')")

        except:
            logging.error('error when read news from {0}'.format(url))
        finally:
            i += 1

    return values

def save_news_east_money(ticker):

    conn = psycopg2.connect(database="stock_info", user='postgres', password='12345678', host="postgres")

    cur = conn.cursor()

    page = 1

    while page <= 50:

        try:

            query = 'insert into stock_news ("date_time", "ticker", "title", "desc", "link", "source") values '

            values = get_news_east_money(ticker, page)

            query = query + ",".join(values)

            cur.execute(query)
            
            conn.commit()

            logging.info("{ticker} news of page {page} from east money saved".format(ticker=ticker, page=page))

        except psycopg2.DatabaseError as error:
            logging.error("sql error {0}".format(error))
            conn.rollback()
        except TimeoutException:
            # if timeout, we just retry
            logging.error("time out exception when {ticker} page {page}".format(ticker=ticker, page=page))
            # the finally will run regardless continue, so we minus the page
            # page -= 1
        finally:
            page += 1
                
    # close the database communication
    conn.close()


def save_news_one_page_east_money(ticker, page):

    conn = psycopg2.connect(database="stock_info", user='postgres', password='12345678', host="postgres")

    cur = conn.cursor()

    try:

        query = 'insert into stock_news ("date_time", "ticker", "title", "desc", "link", "source") values '

        values = get_news_east_money(ticker, page)

        query = query + ",".join(values)

        cur.execute(query)
        
        conn.commit()

        logging.info("{ticker} news of page {page} from east money saved".format(ticker=ticker, page=page))

    except psycopg2.DatabaseError as error:
        logging.error("sql error {0}".format(error))
        conn.rollback()
    except TimeoutException:
        # if timeout, we just retry
        logging.error("time out exception when {ticker} page {page}".format(ticker=ticker, page=page))
        # the finally will run regardless continue, so we minus the page
                
    # close the database communication
    conn.close()

sample_stock = []

for ticker in sample_stock:
    save_news_east_money(ticker)