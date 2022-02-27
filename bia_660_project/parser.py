"""
Created by Vivek Gupta

Gets the job description from the html file and writes it to a csv file
"""

import csv
import os

from bs4 import BeautifulSoup

jobs_dir = "..\\..\\Jobs\\"
fp = open("data/data.csv", "w", newline='', encoding="utf-8")
csv_writer = csv.writer(fp)
csv_writer.writerow(["Description", "Label"])

for root, subdirs, files in os.walk(jobs_dir):
    for file in files:
        file_path = os.path.join(root, file)

        try:
            with open(file_path, "r", encoding="utf8") as fp:
                soup = BeautifulSoup(fp, "html.parser")

            desc = soup.find('div', {'class': 'jobsearch-jobDescriptionText'})
            text = desc.text
            text = " ".join(line.strip() for line in text.splitlines())
            job_title = file_path.split('\\')[-2]
            csv_writer.writerow([text, job_title])
        except:
            print(file_path)

