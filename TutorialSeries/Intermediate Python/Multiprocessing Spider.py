from multiprocessing import Pool
import bs4 as bs
import random
import requests
import string

def random_starting_url():
    starting = ''.join(random.SystemRandom().choice(string.ascii_lowercase) for _ in range(3))      #3 random lowercase characters
    url = ''.join(['http://' , starting , '.com'])
    return url

url = random_starting_url()

# create a spider: go to a website, find all the links on that website, and go to all those links

def handle_local_links(url,link):
    if link.startswith('/'):
        return ''.join([url,link])
    else:
        return link

def get_links(url):
    try:
        resp = requests.get(url)
        soup = bs.BeautifulSoup(resp.text, 'lxml')
        body = soup.body
        links = [link.get('href') for link in body.find_all('a')]
        links = [handle_local_links(url, link) for link in links]       #list of links that we find on any given url. Handle from local link by appending to original link
        links = [str(link.encode('ascii')) for link in links]
        return links

    except TypeError as e:
        print(e)
        print('TypeError. Probaby got a None that we tried to iterate over.')
        return []
    except IndexError as e:
        print(e)
        print('IndexError. Probably did not find any useful links. No list')
        return []
    except AttributeError as e:
        print(e)
        print('AttrtibuteError. Tried to assign attribute when there are none')
        return []
    except Exception as e:
        print(str(e))
        print('Unpredictable event') #log this error. Do not leave an open exception statement
        return []

def main():
    how_many = 250
    p = Pool(processes=how_many)
    parse_us = [random_starting_url() for _ in range(how_many)]         # list of 3 character urls
    data = p.map(get_links, [link for link in parse_us])
    data = [url for url_list in data for url in url_list]
    p.close()

    with open('urls.txt','w') as f:
        f.write(str(data))

if __name__ == '__main__':
    main()






# process forever
    # while True:
    #     Data = p.map(get_links, [link for link in parse_us])                # map of lists of url for all of the links in parse_us
    #     Data = [url for url_list in Data for url in url_list]
    #     parse_us = Data
    #     p.close()




# for every url in each of the mini url lists, in each of the mini url lists in all of the urls that we have (Data) we're saying we want to have a new list
    #    with all of their content. List of lists. Taking the Data and putting it into a single list.
