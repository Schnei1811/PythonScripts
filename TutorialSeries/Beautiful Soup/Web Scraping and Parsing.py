import bs4 as bs
import urllib.request
import pandas as pd

sauce = urllib.request.urlopen('https://pythonprogramming.net/parsememcparseface/').read()      #in word for source
soup = bs.BeautifulSoup(sauce, 'lxml')

print(soup)
print(soup.title)
print(soup.title.name)
print(soup.title.string)            #or .txt

print(soup.p)                       #first paragraph class element
print(soup.find_all('p'))           #find all paragraph tags

for paragraph in soup.find_all('p'):
   print('\n',paragraph.text)           #string will work when no child tags included. (bold, italics, etc)

print(soup.get_text())

for url in soup.find_all('a'):
    print(url.get('href'))

nav = soup.nav                          #navigation tab
body = soup.body

for paragraph in body.find_all('p'):
   print(paragraph.text)
for div in soup.find_all('div', class_='body'):        # multiple body tags. might be a div section
   print(div.text)

table = soup.table
table = soup.find('table')              #same method

table_rows = table.find_all('tr')

for tr in table_rows:
    td = tr.find_all('td')
    row = [i.text for i in td]
    print(row)


dfs = pd.read_html('https://pythonprogramming.net/parsememcparseface/', header=0)
for df in dfs:
    print(df)


sauce = urllib.request.urlopen('https://pythonprogramming.net/sitemap.xml').read()      #in word for source
soup = bs.BeautifulSoup(sauce, 'xml')                   #sitemap: links to all the links on a website
print(soup)
for url in soup.find_all('loc'):
    print(url.text)