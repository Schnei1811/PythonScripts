from multiprocessing import Pool
import bs4 as bs
import random
import requests
import string
import webbrowser
import tqdm
from pytube import YouTube


# with open('urls.txt') as f:
#     urls = f.read().splitlines()
#
# for url in tqdm.tqdm(urls):
#     YouTube('http://youtube.com/{}'.format(url)).streams.first().download()


from pytube import Playlist

pl = Playlist("https://www.youtube.com/playlist?list=UU1WpPRwJ0zag_KNYNlDF-xg")
pl.populate_video_urls()
print("List size is {}s:".format(len(pl.video_urls)))
pl.download_all()




