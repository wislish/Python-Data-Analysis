import pandas as pd
import time
from datetime import datetime
from datetime import timedelta

##国家
country = pd.read_csv("/home/maoan/Downloads/ad_country.csv")

##　找到为空的index
pad_index = country.loc[country['left(IMSI,3)'].isnull(),'Date'].index

##找到第一天，存储
first_date = datetime.strptime("1/1/2015", "%d/%m/%Y")

##循环，每一个为空的index
for i in pad_index:
    if i !=0:
        ##除了第一天，每天加１,
        first_date = first_date+timedelta(1)
    country.loc[i,'Date'] = first_date.strftime("%d/%m/%Y")

##小时
hourl = pd.read_csv("/home/maoan/Downloads/ad_hour.csv")

pad_index = hourl.loc[hourl['HOUR(CREATETIME)']==0,].index

first_date = datetime.strptime("1/1/2015", "%d/%m/%Y")

for i in pad_index:
    if i !=0:
        first_date = first_date+timedelta(1)
    hourl.loc[i,'date'] = first_date.strftime("%d/%m/%Y")