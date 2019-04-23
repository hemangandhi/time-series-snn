import csv
from brian2 import *
import dateutil.parser
import numpy as np
def parseCsvAdj(file):
    out = []
    with open(file) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            out.append(row['Adj Close'])
    return out

def returnNon2019Data(file):
    out = []
    with open(file) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            date = dateutil.parser.parse(row['Date'],ignoretz=True).isoformat()
            date_split = date.split("-")
            if(date_split[0] != '2019'):
                out.append(float(row['Adj Close']))
    return np.asarray(out)

def return2019Data(file):
    out = []
    with open(file) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            date = dateutil.parser.parse(row['Date'],ignoretz=True).isoformat()
            date_split = date.split("-")
            if(date_split[0] == '2019'):
                out.append(float(row['Adj Close']))
    return np.asarray(out)

