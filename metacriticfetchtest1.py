from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import csv
import time

list_temp = []

# 设置Chrome浏览器的路径
chrome_path = "C:\chromedriver_win32\chromedriver.exe"
# 创建一个Chrome浏览器对象
driver = webdriver.Chrome(chrome_path)

# 定义一个函数，用于爬取游戏信息
def scrape_game_info(game_url):
    # 打开游戏页面
    driver.get(game_url)

    # 等待页面元素加载完成
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "product_title")))

    # 获取游戏的标题和类型
    game_title = driver.find_element(By.CLASS_NAME, "product_title").text
    game_genre = driver.find_element(By.CLASS_NAME, "summary_detail.product_genre").text

    # 获取Metascore和Critic Reviews的数量
    metascore = driver.find_element(By.CLASS_NAME, "metascore_w.xlarge.game").text
    critic_reviews = driver.find_element(By.PARTIAL_LINK_TEXT, " Critic Reviews").text[:-15]
    
    #
    # 获取User Score和Ratings的数量
    user_score = driver.find_element(By.CLASS_NAME, "metascore_w.user.large.game").text
    ratings = driver.find_element(By.PARTIAL_LINK_TEXT, " Ratings").text[:-8]

    # 输出游戏信息
    print("Title:", game_title)
    print("Genre:", game_genre)
    dict_temp = {"Title": game_title, "Genre": game_genre}
    list_temp.append(dict_temp)
    with open('GamesSheet_MC.csv', 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list_temp[0].keys())
        writer.writeheader()
        writer.writerows(list_temp)
    # print("Metascore:", metascore)
    # print("Critic Reviews:", critic_reviews)
    # print("User Score:", user_score)
    # print("Ratings:", ratings)

# 定义一个列表，包含需要爬取的游戏页面的URL
df = pd.read_csv('GamesSheet.csv')
game_urls = df['Link']

# 遍历游戏页面的URL列表，依次爬取游戏信息
for url in game_urls:
    scrape_game_info(url)
    time.sleep(1)


# 关闭浏览器
quit()