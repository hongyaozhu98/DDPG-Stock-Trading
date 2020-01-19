# Import packages

from urllib import request
from bs4 import BeautifulSoup
import pandas as pd
import pandas_datareader as pdr

# Get the url
base_url = 'https://en.wikipedia.org/wiki/List_of_S&P_500_companies'
page = request.urlopen(base_url)
soup = BeautifulSoup(page, 'html.parser')

# Extract the table for SP 500
SP_500 = soup.find('table', {"class" : 'wikitable sortable'})

# Create lists for columns
symbol = list()
security = list()
SEC = list()
GICS_Sector = list()
GICS_Sub = list()
location = list()
date = list()
CIK = list()
founded = list()

# Subtract cells
for row in SP_500.findAll("tr"):
    cells = row.findAll('td')
    if len(cells) == 9:
        symbol.append(cells[0].find(text=True))
        security.append(cells[1].find(text = True))
        SEC.append(cells[2].find(text = True))
        GICS_Sector.append(cells[3].find(text = True))
        GICS_Sub.append(cells[4].find(text = True))
        location.append(cells[5].find(text = True))
        date.append(cells[6].find(text = True))
        CIK.append(cells[7].find(text = True))
        founded.append(cells[8].find(text = True))

# Manually adjust some symbols for yahoo search
for i in range(len(symbol)):
    if symbol[i]=='BRK.B':
        symbol[i] = 'BRK-B'
    if symbol[i]=='BF.B':
        symbol[i] = 'BF-B'

df = pd.DataFrame(symbol, columns = ['Symbol'])
df['Security'] = security
df['SEC fillings'] = SEC
df['GICS Sector'] = GICS_Sector
df['GICS Sub Industry'] = GICS_Sub
df['Headerquaters Location'] = location
df['Date first added'] = date
df['CIK'] = CIK
df['Founded'] = founded

# Add the baseline SP 500 index
symbol.append("^GSPC")

# Get and print all historic data
start_date = "2010-1-1"
end_date = "2020-1-1"
Adj_close = pdr.get_data_yahoo(symbol, start_date, end_date)['Adj Close']
'''
Open = pdr.get_data_yahoo(symbol, start_date, end_date)['Open']
High = pdr.get_data_yahoo(symbol, start_date, end_date)['High']
Low = pdr.get_data_yahoo(symbol, start_date, end_date)['Low']
Volume = pdr.get_data_yahoo(symbol, start_date, end_date)['Volume']
'''

Adj_close.to_csv("D:/jiashi/adj_close.csv", index = True)
'''
Open.to_csv("./open.csv", index = True)
High.to_csv("./high.csv", index = True)
Low.to_csv("./low.csv", index = True)
Volume.to_csv("./volume.csv", index = True)
'''

