import re
import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from bs4 import BeautifulSoup

string = "pewdiepie";

url = "https://www.youtube.com/results?search_query=" + string
res = requests.get(url, verify=False)
soup = BeautifulSoup(res.text,'html.parser')
last = None

for entry in soup.select('a'):
    m = re.search("v=(.*)",entry['href'])
    if m:
        target = m.group(1)
        if target == last:
            continue
        if re.search("list",target):
            continue
        last = target
        print target


print m