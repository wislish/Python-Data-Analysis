import os
import configparser
import pandas as pd
import time
import collections
from datetime import datetime
from datetime import timedelta
import sqlite3


cf = configparser.ConfigParser()
# cf.read('test.conf')
cf.read('../maConf.ini')

parent_dir = cf.get('output', 'parent_dir')
# parent_dir = os.getcwd()

##　输入数据路径
##三十六计
db = cf.get('db', 'db_36')
##狂暴之翼
db3 = cf.get('db', 'db3_kuangbao')
##厦门
db4 = cf.get('db', 'db4_xiamen')
##厦门，全。
db_xiamen_latest = cf.get('db', 'db_xiamen_latest')


def writeTo(path,file_name,pd_file):
    """
    根据文件夹路径，以及文件名和类型，保存DataFrame的数据。
    可以自动创建未存在的文件夹。
    默认存储格式为ｅｘｃｅｌ，适合windows用户打开
    
    :param path: 所存文件的父文件夹
    :param file_name: 文件名
    :param pd_file: DataFrame
    :return: 
    """
    full_path = os.path.join(parent_dir, path)
    os.makedirs(full_path, exist_ok=True)

    timeFlag = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    time_stamp = '_'.join(timeFlag.split())
    file_name = file_name + "_" +time_stamp
    full_path = os.path.join(full_path,file_name)

    full_path = full_path + ".xlsx"
    print(full_path)
    pd_file.to_excel(full_path, "sheet1", index=False,engine='xlsxwriter')


## 分离了 database 的管理
class DatabaseManager(object):
    def __init__(self, db):
        self.conn = sqlite3.connect(db)
        self.cur = self.conn.cursor()

    def query(self, arg):
        self.cur.execute(arg)
        return self.cur

    def __del__(self):
        self.conn.close()

def dataGen(db, query):
    dbms = DatabaseManager(db)

    for row in dbms.query(query):
        yield row

## 单独构造了生成指定间隔的生成器的类
class IntervalGenerator(object):

    def __init__(self, first_date, days=3):
        self.days = days
        self.first_date = datetime.fromtimestamp(first_date)
        self.begin_interval = -1
        self.end_interval = -1

    def daysGenerator(self):
        for i in range(self.days + 1):
            num_of_days = timedelta(i)
            interval = self.first_date + num_of_days
            during_secs = interval.second + 60 * interval.minute + interval.hour * 3600
            self.begin_interval = interval.timestamp() - during_secs
            self.end_interval = interval.timestamp() + (86400 - during_secs)
            yield self

    def hoursGenerator(self):
        num_of_hours = (self.days + 1) * 24
        for i in range(num_of_hours):
            first_date_timestamp = self.first_date.timestamp()
            # during_secs = interval.second + 60 * interval.minute + interval.hour * 3600
            self.begin_interval = first_date_timestamp + i*3600
            self.end_interval = first_date_timestamp + (i+1)*3600
            yield self

    def clearInterval(self):
        self.begin_interval = -1
        self.end_interval = -1

## 分离比较ACTION的逻辑
def compareAction(user_action, action_list):


    if user_action in action_list:
        return True
    else:
        return False


"""
    Repeating Interval Decorator
        used to process one function several days.
    在给定的天数内，对每天的数据进行相应的运算。
    被装饰的函数必须有`begin_date`,`end_date`这两个参数。
    
"""
def repeatByInterval(begin_date, during_days, upperDir,file_name):
    def decorate(func):
        def clocked(**para):
            intervalGen = IntervalGenerator(begin_date.timestamp(), days=during_days)
            i = 0
            res = pd.DataFrame()
            value_l = []
            for ig in intervalGen.daysGenerator():
                start_date = ig.begin_interval
                end_date = ig.end_interval
                print(datetime.fromtimestamp(start_date))
                print(datetime.fromtimestamp(end_date))

                ## run the decorated function
                dayOne = func(begin_date=start_date, end_date=end_date, **para)

                ## check the type of return value
                if isinstance(dayOne, pd.DataFrame):
                    if i == 0:
                        res.loc[:, 'Player'] = dayOne.iloc[:, 0]
                    name = str(i) + "Day"
                    res.loc[:, name] = dayOne.iloc[:, 1]
                else:
                    value_l.append(dayOne)
                i += 1

            if len(value_l) != 0:
                res.loc[:, 'ValueList'] = pd.Series(value_l)

            writeTo(upperDir, file_name, res)
            # print(fmt.format(**locals()))
            return res

        return clocked

    return decorate

if __name__ == "__main__":
    os.listdir(parent_dir)