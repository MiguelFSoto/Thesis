from bs4 import BeautifulSoup
import requests

link = "https://www.sciencedirect.com/science/article/pii/S2090123221001491"
resHtml = requests.get(link).text
soup = BeautifulSoup(resHtml, "html.parser")

print(soup)

h3 = soup.find("h3")
#print(soup.h3)
