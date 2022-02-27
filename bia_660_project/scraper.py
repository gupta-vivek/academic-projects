"""
Created by Vivek Gupta

Gets the job ads from indeed.com for given job types and locations
"""
import csv
import os
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

INDEED_WORLDWIDE = "https://www.indeed.com/worldwide"
STATE_ABB_PATH = 'data/state_abb.csv'
FILE_NAME = "data/jobs.csv"

CITIES = {
    'United States': {
        'Arizona': ['Phoenix', 'Scottsdale', 'Tempe'],
        'California': ['Mountain View', 'San Francisco'],
        'Connecticut': ['Stamford', 'Shelton'], 'Florida': ['Tampa', 'Miami', 'Orlando'],
        'Maryland': ['Bethesda', 'Fort Meade'], 'Michigan': ['Detroit'],
        'New Jersey': ['Jersey City'], 'North Carolina': ['Wilmington'],
        'Oregon': ['Portland'], 'Pennsylvania': ['Pittsburgh'],
        'Texas': ['Austin'],
        'Utah': ['Lehi', 'Salt Lake City'],
        'Washington': ['Seattle']},

    'Singapore': {
        ' ': ['Singapore', 'Outram', 'Jurong Island', 'Clementi']
    },

    'Ireland': {
        ' ': ['Limerick']
    },

    'Switzerland': {
        'Basel-Stadt': ['Basel'],
        'Geneva': ['Geneva'],
        'Zurich': ['Zürich']
    },

    'Australia': {
        ' ': ['Sydney NSW', 'Melbourne VIC', 'Canberra ACT']
    },

    'South Africa': {
        'Gauteng': ['Johannesburg'],
        'Western Cape': ['Cape Town']
    },

    'United Kingdom': {
        ' ': ['London']
    }
}

JOBS = ['Data Engineer', 'Software Engineer', 'Data Scientist']
COUNT = 25000
CWD = os.getcwd()

jobs_dict = {'Job': [], 'Title': [], 'Company': [], 'Country': [], 'State': [], 'City': [],
             'Link': []}  # Dictionary to maintain job details


def get_state_abb(state_abb_path):
    """
    Gets state abbreviation

    :param state_abb_path: state abbreviation file path
    :return:
    """
    df = pd.read_csv(state_abb_path)
    state_dict = {k: {} for k in df['Country'].unique()}
    for i in df.values:
        state_dict[i[0]][i[1]] = i[2]
    return state_dict


def get_indeed_url(worldwide_url, countries):
    """
    Gets indeed url of given country

    :param worldwide_url: Worldwide URL Link
    :param countries:
    :return: a dictionary with countries as key and corresponding indeed link of the country
    """
    country_dict = {}
    response = None
    for i in range(5):  # try 5 times
        # send a request to access the url
        response = requests.get(worldwide_url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36', })
        if response:
            break
        else:
            time.sleep(2)

    if not response:
        return None

    html = response.text
    soup = BeautifulSoup(html, features="lxml")
    countries_links = soup.find('tr', {'class': 'countries'}).find_all('a')

    for country in countries:
        if country == 'United States':
            country_dict[
                country] = "https://www.indeed.com/stc?_ga=2.156731067.1828171807.1603800916-701039332.1603800916"
            continue
        for link in countries_links:
            if country == link.text.strip():
                country_url = link.get('href')
                country_dict[country] = country_url
                break

    return country_dict


def close_popup(driver):
    """
    Closes the annoying popup window
    :return:
    """
    try:
        driver.find_element_by_css_selector('button[aria-label="Close"]').click()
        time.sleep(2)
    except:
        pass


def click_what(driver, job):
    """
    Enters job on what field
    :param driver: Chrome driver
    :param job: job
    :return:
    """
    try:
        what_field = driver.find_element_by_css_selector('input[id="text-input-what"]')
    except:
        what_field = driver.find_element_by_css_selector('input[id="what"]')
    value = what_field.get_attribute("value")
    for i in range(len(value)):
        what_field.send_keys(Keys.BACK_SPACE)
    what_field.send_keys(job)


def click_where(driver, state):
    """
    Enter state on where field
    :param driver: Chrome driver
    :param state: state
    :return:
    """
    try:
        where_field = driver.find_element_by_css_selector('input[id="text-input-where"]')
    except:
        where_field = driver.find_element_by_css_selector('input[id="where"]')
    value = where_field.get_attribute("value")
    for i in range(len(value)):
        where_field.send_keys(Keys.BACK_SPACE)
    where_field.send_keys(state)


def find_job(driver):
    """
    Clicks on find job button
    :param driver: Chrome driver
    :return:
    """
    try:
        driver.find_element_by_css_selector(
            'button.icl-Button.icl-Button--primary.icl-Button--md.icl-WhatWhere-button').click()
    except:
        driver.find_element_by_css_selector('input.input_submit').click()


def get_links(input_cities, jobs, count):
    """
    Gets job links for given jobs and cities

    :param input_cities: Location of the job
    :param jobs: Job Designation
    :param count: Number of jobs to be scraped
    :return:
    """
    driver = webdriver.Chrome('./chromedriver.exe')
    # driver.maximize_window()
    state_dict = get_state_abb(STATE_ABB_PATH)
    INDEED_URLS = get_indeed_url(INDEED_WORLDWIDE, input_cities.keys())

    fp = open(FILE_NAME, "a", newline='', encoding="utf-8")
    csv_writer = csv.writer(fp)
    csv_writer.writerow(['Job', 'Title', 'Company', 'Country', 'State', 'City', 'Link'])

    # Go through each jobs
    for job in jobs:
        print("\n----- {} -----".format(job))
        job_counter = 0
        close_popup(driver)
        CountFlag = False

        countries = input_cities.keys()
        # Go through each country
        for country in countries:
            print("\nCOUNTRY: ", country)
            country_counter = 0
            url = INDEED_URLS[country]
            driver.get(url)
            time.sleep(2)
            click_what(driver, job)

            states = input_cities[country].keys()
            # Go through each states
            for state in states:
                close_popup(driver)
                state_counter = 0
                print("\nSTATE: ", state)
                if CountFlag:
                    break

                click_where(driver, state)
                find_job(driver)
                time.sleep(2)

                # Click on advanced search
                adv_search = driver.find_element_by_css_selector('td.npl.advanced-search').find_element_by_css_selector(
                    'a')
                adv_search.click()
                time.sleep(2)

                # Select max radius
                select = driver.find_element_by_css_selector('select[id="radius"]')
                options = select.find_elements_by_css_selector('option')
                options[-1].click()

                # Select max age
                select = driver.find_element_by_css_selector('select[id="fromage"]')
                options = select.find_elements_by_css_selector('option')
                options[0].click()

                # Select max display
                select = driver.find_element_by_css_selector('select[id="limit"]')
                options = select.find_elements_by_css_selector('option')
                options[1].click()

                driver.find_element_by_css_selector('button[value="Find Jobs"]').click()
                time.sleep(2)

                cities = input_cities[country][state]

                # Go through each cities
                for city in cities:
                    city_counter = 0
                    close_popup(driver)

                    if CountFlag:
                        break

                    try:
                        state_abb = ', ' + state_dict[country][state]
                    except:
                        print("No state abbreviation found for {}: {}".format(country, state))
                        state_abb = ''

                    # Click location button
                    location_button = driver.find_element_by_id('filter-location')
                    location_button.click()

                    # Click city from location drop down
                    city_list = location_button.find_elements_by_css_selector('span.rbLabel')
                    city_found_flag = False
                    for c in city_list:
                        city_ = city + state_abb
                        if c.text == city_:
                            city_found_flag = True
                            c.click()
                            time.sleep(2)
                            break

                    if not city_found_flag:
                        print("{}: {}".format(city, city_counter))
                        location_button.click()
                        continue

                    CityFlag = False
                    # Get all job cards in the city
                    while not CityFlag:
                        close_popup(driver)
                        # Get all job cards in the page
                        job_cards = driver.find_elements_by_css_selector(
                            'div.jobsearch-SerpJobCard.unifiedRow.row.result.clickcard')

                        for job_card in job_cards:
                            city_counter += 1
                            job_counter += 1
                            job_title = job_card.find_element_by_css_selector(
                                'a[data-tn-element="jobTitle"]').get_attribute(name="title")
                            try:
                                job_company = job_card.find_element_by_css_selector('span.company').text
                            except:
                                job_company = "no_name"
                            job_location = job_card.find_element_by_css_selector('div.recJobLoc').get_attribute(
                                name='data-rc-loc')
                            job_link = job_card.find_element_by_css_selector(
                                'a[data-tn-element="jobTitle"]').get_attribute(
                                name="href")

                            jobs_dict['Job'].append(job)
                            jobs_dict['Title'].append(job_title)
                            jobs_dict['Company'].append(job_company)
                            jobs_dict['Country'].append(country)
                            jobs_dict['State'].append(state)
                            jobs_dict['City'].append(job_location)
                            jobs_dict['Link'].append(job_link)

                            csv_writer.writerow([job, job_title, job_company, country, state, job_location, job_link])

                        if job_counter <= count:
                            pass
                        else:
                            fp.flush()
                            CountFlag = True
                            break

                        # Click on next button
                        try:
                            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                            time.sleep(2)
                            driver.find_element_by_css_selector('a[aria-label="Next"]').click()
                            time.sleep(3)
                        except:
                            fp.flush()
                            clear_location = driver.find_element_by_css_selector(
                                'a[aria-label="Clear Location filter"]')
                            clear_location.click()
                            CityFlag = True

                    print("{}: {}".format(city, city_counter))
                    state_counter += city_counter

                country_counter += state_counter
                print("Total jobs from {}: {}".format(state, state_counter))

            print("\nTotal jobs from {}: {}".format(country, country_counter))

        print("\nTotal jobs found for {}: {}\n".format(job, job_counter))
    fp.close()


def get_html(df):
    """
    Gets the link from the dataframe and saves the html file
    :param df: dataframe

    """
    CWD = "..\.."
    if not os.path.exists(CWD + "\Jobs"):
        os.mkdir(CWD + "\Jobs")
    start = 0
    for job in JOBS:
        print(job)

        job_path = CWD + "\Jobs" + '\\' + job
        if not os.path.exists(job_path):
            os.mkdir(job_path)
        temp_df = df[df["Job"] == job]
        failed = []

        for ind, row in enumerate(temp_df.values[start:]):
            index = start + ind
            file_name = job_path + "\\" + row[1].replace('\\', ' ').replace('/', ' ') + "_" + row[2].split(' ')[
                0] + '_' + row[-2].split(',')[-1] + '.html'

            if os.path.exists(file_name):
                continue
            if index % 100 == 0:
                print(index)
            response = None
            for i in range(2):  # try 2 times
                # send a request to access the url
                response = requests.get(row[-1], headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36', })
                if response:
                    break
                else:
                    time.sleep(3)

            if not response:
                print(index, row[-1])
                failed.append(row[-1])
                continue

            html = response.content
            if len(html) == 0:
                time.sleep(3)
                continue

            soup = BeautifulSoup(html, "html5lib")
            try:
                with open(file_name, "w", encoding="utf-8") as fp:
                    fp.write(str(soup))
            except:
                print("File: ", index, row[-1])
            time.sleep(3)
        print(failed)


get_links(CITIES, JOBS, COUNT)
new_df = pd.read_csv(FILE_NAME, engine="python")
get_html(new_df)
