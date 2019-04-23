import csv

def parseCsvAdj(file):
    out = []
    with open(file) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            out.append(row['Adj Close'])
    return out
       