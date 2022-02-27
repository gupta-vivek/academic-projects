# PROJECT INDEED

**For predictions**

Run _predictor.py_. Pass the input filename as system argument.

`python3 predictor.py filename.csv`

A new file _predictions.csv_ will be created and predictions will be saved there.

_scraper.py_ - scrapes job ads from indeed for the given jobs and cities

_parser.py_ - gets the job descriptions from the html file and save it to a csv file

_xgb_model.py_ - trains and saves the model


**Notes**

Job ads csv file is present under data folder. The name of the file is _data.csv_.

Raw _HTML_ files are present under the _Jobs_ folder.

_cities.txt_ contains the cities from where the job ads were scraped
