from selenium import webdriver
import time
driver = webdriver.Firefox(executable_path='C:/Python27/geckodriver.exe')
driver.get("https://search.bilibili.com/video?keyword=%E9%92%A2%E4%B9%8B%E7%82%BC%E9%87%91%E6%9C%AF%E5%B8%88&page=2")
A ="//*[@class='video-list clearfix']/li["
B="]/a"
C=A +"*"+B

list_links = driver.find_elements_by_xpath(C)

for i in list_links:
        print(i.get_attribute('href'))
time.sleep(5)
driver.quit()