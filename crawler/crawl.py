#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
抓取豆瓣电影和imdb数据
'''

import requests
from bs4 import BeautifulSoup
import re

class MovieRecord(object):
    def __init__(self):
        self.movie_name = ""
        self.actor = ""
        self.genre = ""
        self.rating = ""
        self.mpaa_rating = ""
        self.director = ""
        self.release_year = ""

def crawl():
    movie_records = []
    original_link = "https://movie.douban.com/top250"
    r = requests.get(original_link)
    if r.status_code == 200:
        s = BeautifulSoup(r.text, "lxml")
        pages_left = s.find_all("div", "paginator")[0].find_all("a")
        page_addresses = [original_link + link["href"] for link in pages_left]
        page_addresses.append(original_link)
        print page_addresses
        for page_address in page_addresses:
            movie_records += parse_page(page_address)
    save_data(movie_records, "data/raw_db.txt")

def parse_page(address):
    print address
    page_movies = []
    r = requests.get(address)
    soup = BeautifulSoup(r.text, "lxml")
    items = soup.find_all("ol", "grid_view")[0].find_all("div", "item")
    for item in items:
        movie_record = MovieRecord()
        movie_record.movie_name = item.find_all("span", "title")[0].text
        detail_req = requests.get(item.find_all("a")[0]["href"])
        detail_soup = BeautifulSoup(detail_req.text, "lxml")
        for address in detail_soup.find_all("div", "subject clearfix")[0].find_all("a"):
            if "www.imdb.com" in address["href"]:
                imdb_req = requests.get(address["href"])
                imdb_soup = BeautifulSoup(imdb_req.text, "lxml")
                movie_record.rating = imdb_soup.find_all("div", "ratingValue")[0].text.strip().split("/")[0].strip()
                mpaa_rating = imdb_soup.find_all("div", "subtext")[0].text.strip().split("|")[0].strip()
                if u"h" in mpaa_rating or u'min' in mpaa_rating or u'Ban' in mpaa_rating:
                    movie_record.mpaa_rating = u'UNK'
                else:
                    movie_record.mpaa_rating = mpaa_rating
        info_list = detail_soup.find_all("div", "info")
        for info in info_list:
            roles =[x.text for x in info.find_all("span", "role")]
            names =[x.text for x in info.find_all("span", "name")]
            for role, name in zip(roles, names):
                print role, name
            if len(roles) > 0 and u"导演" in roles[0] and movie_record.director == "":
                movie_record.director = names[0].strip()
            if len(roles) > 0 and (u"饰" in roles[0] or u"配" in roles[0] or u'配音' in roles[0] or u'演员' in roles[0] or u'自己' in roles[0]) and movie_record.actor == "":
                movie_record.actor = names[0].strip()
        for intro in item.find_all("p")[0].text.split("\n"):
            intro = intro.strip()
            release_year = re.findall(ur'^\d+', intro)
            if len(release_year) > 0:
                movie_record.release_year = release_year[0]
                if len(movie_record.release_year) > 4:
                    movie_record.release_year = movie_record.release_year[:4]
            year_locality_genre = intro.split("/")
            if len(year_locality_genre) > 2:
                genre = year_locality_genre[2].strip().split(" ")[0]
                movie_record.genre = genre
        page_movies.append(movie_record)
    return page_movies

def save_data(data, path):
    with open(path, "wb") as f:
        f.write(u'片名	主演	评分	类别	评级	发行年份	导演\n')
        for movie in data:
            str = u"\t".join([movie.movie_name, movie.actor, movie.rating, movie.genre,
                             movie.mpaa_rating, movie.release_year, movie.director])
            str += u'\n'
            f.write(str.encode("utf-8"))

def load_data():
    pass

if __name__ == '__main__':
    crawl()