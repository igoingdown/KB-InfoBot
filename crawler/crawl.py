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
    for movie in movie_records:
        print "\t".join([movie.movie_name, movie.actor, movie.rating, movie.genre,
                         movie.mpaa_rating, movie.release_year, movie.director])
    # with open("tmp.txt", "wb") as f:
    #     for movie in movie_records:
    #         f.write("\t".join([movie.movie_name, movie.actor, movie.rating, movie.genre,
    #                          movie.mpaa_rating, movie.release_year, movie.director]))

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
                movie_record.mpaa_rating = imdb_soup.find_all("div", "subtext")[0].text.strip().split("|")[0].strip()
        info_list = detail_soup.find_all("div", "info")
        for info in info_list:
            print info.text
        for intro in item.find_all("p")[0].text.split("\n"):
            intro = intro.strip()
            directors = re.findall(ur'导演: .*? ', intro)
            actors = re.findall(ur"主演: .*? ", intro)
            release_year = re.findall(ur'^\d+', intro)
            if len(directors) > 0:
                movie_record.director = directors[0].split(" ")[1]
            if len(actors) > 0:
                movie_record.actor = actors[0].split(" ")[1]
            if len(release_year) > 0:
                movie_record.release_year = release_year[0]
            year_locality_genre = intro.split("/")
            if len(year_locality_genre) > 2:
                genre = year_locality_genre[2].strip().split(" ")[0]
                movie_record.genre = genre
        page_movies.append(movie_record)
    return page_movies

def save_data():
    pass


def load_data():
    pass

if __name__ == '__main__':
    crawl()