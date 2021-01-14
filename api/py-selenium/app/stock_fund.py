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

from stock300 import stock300
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

def get_stock_fund_east_money(ticker):

    url = "http://data.eastmoney.com/zlsj/detail/{ticker}-1.html".format(ticker=ticker)

    driver = get_selenium_driver()

    driver.get(url)
    
    fund_items = WebDriverWait(driver, 8)\
      .until(at_least_all_present((By.CSS_SELECTOR, '#ccmx_table tbody tr'), 1))

    i = 0
    values = []

    while i < len(fund_items):

        try:
            
            fund_name = fund_items[i].find_element_by_css_selector('td:nth-child(2) a').text

            fund_link = fund_items[i].find_element_by_css_selector('td:nth-child(3) a:nth-child(2)')\
                                    .get_attribute('href')

            fund_company = fund_items[i].find_element_by_css_selector('td:nth-child(5)').text

            amount = fund_items[i].find_element_by_css_selector('td:nth-child(6)').text

            value = fund_items[i].find_element_by_css_selector('td:nth-child(7)').text

            total_per = fund_items[i].find_element_by_css_selector('td:nth-child(8)').text

            circulate_per = fund_items[i].find_element_by_css_selector('td:nth-child(9)').text

            values.append("('" + "', '".join([str(ticker), fund_name, fund_link, fund_company, amount,\
              value,total_per,circulate_per]) + "')")

        except:
            logging.error('error when read fund from {0}'.format(url))
        finally:
            i += 1

    return values

def save_news_east_money(ticker):

    conn = psycopg2.connect(database="stock_info", user='postgres', password='12345678', host="postgres")

    cur = conn.cursor()

    try:

        query = 'insert into stock_fund ("ticker", "fund_name", "fund_link",\
          "fund_company", "amount", "value", "total_per", "circulate_per") values '

        values = get_stock_fund_east_money(ticker)

        query = query + ",".join(values)

        cur.execute(query)
        
        conn.commit()

        logging.info("{ticker} fund info from east money saved".format(ticker=ticker))

    except psycopg2.DatabaseError as error:
        logging.error("sql error {0}".format(error))
        conn.rollback()
    except TimeoutException:
        # if timeout, we just retry
        logging.error("time out exception when read {ticker} fund info".format(ticker=ticker))
        # the finally will run regardless continue, so we minus the page
        # page -= 1
    finally:
        if conn is not None:
            conn.close()

for ticker in stock300:
    save_news_east_money(ticker)
