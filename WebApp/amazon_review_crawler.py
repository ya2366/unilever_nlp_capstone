import requests
import os
from bs4 import BeautifulSoup
import pprint 
import json
import re
import pandas as pd
import time

reviews = []

def getReviews(page, product_name):
	global reviews

	content = requests.get(page)
	html = content.text
	soup = BeautifulSoup(html, "lxml")
	total = 0

	try:
		for level1 in soup.find_all("div", {'class': 'a-section a-spacing-none reviews-content a-size-base'}):
			for level2 in level1("div", {'id': 'cm_cr-review_list'}):
				for level3 in level2("div", {'class': 'a-section review'}):
					review = {}
					review['product'] = product_name

					# Avoid overwriting of stars
					stars_text = ""

					# Get review ratings and title
					for level4 in level3("div", {'class': 'a-row'}):
						for level5 in level4("span", {'class': 'a-icon-alt'}):
							if stars_text == "":
								stars_text = level5.text.split(" ")[0]
								review['stars'] = stars_text
					
					for level4 in level3("a", {'class': 'a-size-base a-link-normal review-title a-color-base a-text-bold'}):
						review['title'] = level4.text

					# Get review text
					for level4 in level3("div", {'class': 'a-row review-data'}):
						for level5 in level4("span", {'class': 'a-size-base review-text'}):
							text = re.sub(r"<.*>", " ", level5.text)
							review['text'] = text.strip()

					reviews.append(review)
					total += 1

		return total

	except:
		return 0


def extractReviews(product_url, num_reviews):
	global reviews

	p = product_url.split("/")
	product_name = p[3]
	product_id = p[5]

	product_page = "/".join(p[:3]) + '/product-reviews/' + p[5]

	fl = 0
	if num_reviews % 10 > 0:
		fl = 1
	total_pages = num_reviews // 10 + fl
	total_reviews = 0

	reviews =  []

	for pageno in range(total_pages):
		page = product_page + "/ref=cm_cr_getr_d_paging_btm_" + str(pageno) + "?ie=UTF8&reviewerType=all_reviews&pageNumber=" + str(pageno)
		total = getReviews(page, product_name)

		if total == 0 and total_reviews == 0:
			return reviews, product_id, product_name, "Unsuccessful!! Retrieved only  " + str(total_reviews) + "  reviews"

		total_reviews += total

	return reviews, product_id, product_name, "Successful!! Retrieved total of  " + str(num_reviews) + "  reviews"
