import csv
import random
import re

# from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains

from useragent import UserAgents

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

def get_news_east_money(code):
    page = 50

    while page > 0:

        url = "http://so.eastmoney.com/news/s?keyword={stock_code}&pageindex={page}".format(stock_code=code,page=page)

        driver = get_selenium_driver()

        driver.get(url)

        news_item = WebDriverWait(driver, 3).until(at_least_all_present((By.CSS_SELECTOR, '.news-item'), 10))

        i = 0
        titles = []
        desc = []
        dates = []
        links = []


        while i < len(news_item):

            titles.append(news_item[i].find_element_by_css_selector('h3 a').text)
            desc.append(news_list[i].find_element_by_css_selector('.news-desc').text)
            searched = re.search(r'^(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})', desc[-1])
            dates.append(searched.group(1))
            links.append(news_list[i].find_element_by_css_selector('.link a').text)

            i += 1



        break

        page -= 1

get_news_east_money('600104')