import pandas as pd
import numpy as np
import sqlite3
import collections
import time
from datetime import datetime
from datetime import timedelta
import seaborn as sns
from sklearn import preprocessing

import warnings
warnings.simplefilter('ignore', np.RankWarning)

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

    for row in db.query(query):
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


def latestPlayDate(db):
    dbms = DatabaseManager(db)

    sqlstr = "SELECT player_id, happen_time FROM maidian ORDER BY player_id,happen_time ASC;"
    data_iterator = dataGen(dbms, sqlstr)

    latestDate = -1
    player = -1

    counter = 0
    # dist = collections.defaultdict(int)
    print("Enter!")
    for row in data_iterator:
        counter += 1
        if counter % 1000000 == 0:
            print("%s lines processed\n" % counter)
            # print(timestamp)

        player_id = row[0]
        timestamp = row[1]

        if timestamp > latestDate:
            latestDate = timestamp
            player = player_id

    print("Latest Player {0} - Time:{1}".format(player, latestDate))

def findMoneyUser(db):

    dbms = DatabaseManager(db)
    keyList = ["充值", "开服累冲", "购", "礼", "总"]

    sqlstr = "SELECT player_id, happen_time, minor_class, action, gold FROM maidian ORDER BY player_id,happen_time ASC;"
    data_iterator = dataGen(dbms, sqlstr)

    key_indicator = "开服活动"
    #在规定的步数以内元宝需要增长,才算是付费用户
    step_limt = 10
    #最少充值１００元宝！
    money_limit = 100

    current_player = -1
    origin_money = -1
    find_indicator = False
    steps = -1
    money_user = False
    money_ulist = []
    step_list = []
    total_index_dict = collections.defaultdict(list)

    starting_time = time.time()
    counter = 0

    for row in data_iterator:
        counter += 1
        if counter % 1000000 == 0:
            print("%s lines processed\n" % counter)
            # print(timestamp)

        player_id = row[0]
        timestamp = row[1]
        subject_action = row[2]
        action = row[3]
        gold = row[4]

        # 如果更换用户，初始化所有值
        if player_id != current_player:

            find_indicator = False
            money_user = False
            current_player = player_id

        if money_user:
            continue

        if find_indicator and steps <= step_limt:
            steps += 1
            present_money = gold
            if present_money - origin_money >= money_limit:
                money_user = True
                # print("Present Money: {0} - Original Money: {1}. Steps: {2}".format(present_money, origin_money, steps))
                step_list.append(steps)
                money_ulist.append(current_player)
                continue

        if sum([1 for e in keyList if e in subject_action]) > 0:
            find_indicator = True
            origin_money = gold
            steps = 0


    return pd.DataFrame({"user":money_ulist, "steps":step_list})


def clickFrequency(db):

    dbms = DatabaseManager(db)

    sqlstr = "SELECT player_id, happen_time, major_class FROM maidian ORDER BY player_id,happen_time ASC;"
    data_iterator = dataGen(dbms, sqlstr)

    # 如果两次动作的间隔时间大于３００秒，算两次连续动作
    interval_limit = 300
    first_action_time = -1
    last_action_time = -1
    times = -1.
    current_player = -1
    freq_list = []
    total_index_dict = collections.defaultdict(list)

    starting_time = time.time()
    counter = 0

    for row in data_iterator:
        counter += 1
        if counter % 1000000 == 0:
            print("%s lines processed\n" % counter)
            # print(timestamp)

        player_id = row[0]
        timestamp = row[1]
        major_action = row[2]

        #         ## 如果在启动阶段，用户的关键指标可能是０，因为需要从服务器取得数据。所以忽略。
        #         if major_action == None or major_action == "启动":
        #             continue

        # 如果更换用户，初始化所有值
        if player_id != current_player:

            ##有的用户只有一次连续操作的时间段
            if len(freq_list) == 0:
                diff = last_action_time - first_action_time
                if diff != 0:
                    freq = times / diff
                    freq_list.append(freq)

            if current_player != -1:
                total_index_dict[current_player] = np.median(freq_list)

            first_action_time = timestamp
            last_action_time = timestamp
            freq_list = []
            times = -1.
            current_player = player_id
            # print("================")
        #         if player_id == 36028840385154595:

        #             print("last " + str(last_action_time))
        #             print("first " + str(first_action_time))
        #             print("now " + str(timestamp))

        if timestamp - last_action_time > interval_limit:

            diff = last_action_time - first_action_time
            ## 连续时间必须大于５分钟才加入计算
            if diff >= 300:
                freq = times / diff
                freq_list.append(freq)

            first_action_time = timestamp
            times = -1.

        times += 1
        last_action_time = timestamp

    ##有的用户只有一次连续操作的时间段
    if len(freq_list) == 0:
        diff = last_action_time - first_action_time
        if diff != 0:
            freq = times / diff
            freq_list.append(freq)

    total_index_dict[current_player] = np.median(freq_list)

    pd_dist = pd.DataFrame(list(total_index_dict.items()), columns=['Player', 'Freq'])

    return pd_dist

def numOfRecords(x,begin_date, end_date):

    interval = x.loc[(x['happen_time'] >begin_date) & (x['happen_time'] < end_date),]
    return (interval['happen_time'].tail(1) - interval['happen_time'].head(1))

##在给定的日期内,算出连续操作的时间段的和
def realPlayTime(timeList, begin_date, end_date, threshold = 300):

    during_time = 0
    truncL = [t for t in timeList if t <= end_date and t >= begin_date]

    begin_time = truncL[0]
    last_time = begin_time
    for t in truncL:

        if t - last_time > threshold:
            during_time += (last_time - begin_time)
            begin_time = t

        last_time = t

    during_time += (last_time - begin_time)

    return during_time

## 分离比较ACTION的逻辑
def compareAction(user_action, action_list, auto=False):

    if not auto:
        if user_action in action_list:
            return True
        else:
            return False
    else:
        if user_action not in action_list:
            return True
        else:
            return False


def ui_stay_time(in_action, out_action, loss_window=20):

    dbms = DatabaseManager(db)
    query_sql = "SELECT player_id,action,happen_time, major_class FROM maidian ORDER BY player_id,happen_time ASC"
    data_iterator= dataGen(dbms, query_sql)

    stay_limit = 1500
    num_enter = 0
    num_of_correct_exit = 0

    current_player = -1

    find_flag = False
    correct_exit = True
    window_len = 0

    starting_time = time.time()
    counter = 0

    enter_time = 0
    total_exit_time = 0

    stay_time = 0
    duringDict = {}

    for row in data_iterator:

        counter += 1
        if counter % 1000000 == 0:
            print("%s lines processed\n" % counter)
            # print("total enter:" + str(num_enter))
            # print("correct exit :" + str(num_of_correct_exit))

        player_id = row[0]
        major_action = row[3]
        # 如果更换用户，初始化所有值，并且计算上一个用户的所有以及正确的停留的时间。
        if player_id != current_player:
            find_flag = False
            window_len = 0
            correct_exit = True

            if current_player != -1:
                stay_time = (total_exit_time - enter_time)
                duringDict[current_player] = stay_time

            current_player = player_id

            enter_time = 0
            total_exit_time = 0
            stay_time = 0

        action = row[1]
        timestamp = row[2]

        # ## 如果操作为启动状态,忽略.
        # if major_action == None or major_action == "启动":
        #     continue

        # 如果被标记，该用户所有剩余action都忽略
        if not correct_exit:
            continue

        # 如果对某一个用户，第一次找到相应的进入UI动作，记录总人数加一并标记。记录进入时间
        if (compareAction(action, in_action) and (not find_flag)):
            find_flag = True
            enter_time = timestamp
            total_exit_time = timestamp
            num_enter += 1
            continue

        # 如果已找到进入动作，判断此时动作是否为退出动作。如果是，正确退出加一，否则判断窗口长度是否大于
        # 标准值，如果是的话，此用户记录为非正常退出，并记录时间
        if find_flag:
            if compareAction(action, out_action):
                num_of_correct_exit += 1
                total_exit_time = timestamp
                correct_exit = False
            else:
                window_len += 1
                ##　如果步数大于阈值,且经历的时间小于规定，记录此时时间为退出时间.
                if window_len >= loss_window and (timestamp - enter_time) < stay_limit:
                    correct_exit = False
                    total_exit_time = timestamp
                ## 如果经历的时间大于规定,则记录退出时间等于开始时间.
                if (timestamp - enter_time) > stay_limit:
                    correct_exit = False
                    total_exit_time = enter_time

    stay_time = (total_exit_time - enter_time)
    duringDict[current_player] = stay_time

    pd_dist = pd.DataFrame(list(duringDict.items()), columns=['Player', 'TimeOnUI'])

    return pd_dist

def timePlayOneDay(db, begin_date, end_date, threshold):
    dbms = DatabaseManager(db)

    sqlstr = "SELECT player_id, happen_time, major_class FROM maidian ORDER BY player_id,happen_time ASC;"
    data_iterator = dataGen(dbms, sqlstr)

    current_player = -1
    #     total_index_dict = collections.defaultdict(list)
    timeDict = {}
    during_time = 0

    counter = 0
    # dist = collections.defaultdict(int)
    print("Enter!")
    for row in data_iterator:
        counter += 1
        if counter % 1000000 == 0:
            print("%s lines processed\n" % counter)
            # print(timestamp)

        player_id = row[0]
        timestamp = row[1]
        major_action = row[2]
        if player_id != current_player:

            if current_player != -1:
                during_time += (last_time - begin_time)
                timeDict[current_player] = during_time

            begin_time = timestamp
            last_time = timestamp

            during_time = 0
            current_player = player_id

        if timestamp > end_date or timestamp < begin_date:
            continue

        # ## 如果在启动阶段，用户的关键指标可能是０，因为需要从服务器取得数据。所以忽略。
        #         if major_action == None or major_action == "启动":
        #             continue

        if timestamp - last_time > threshold:
            during_time += (last_time - begin_time)
            begin_time = timestamp

        last_time = timestamp

    during_time += (last_time - begin_time)
    timeDict[current_player] = during_time
    pd_dist = pd.DataFrame(list(timeDict.items()), columns=['Player', 'Duration'])

    return pd_dist

def firstNonZeroTime(x):
    return x.loc[x['Duration']!=0,'Duration'].head(1).values

def lossUsers(x):
    return x.loc[x['Duration'] > 0,'Duration'].count()

def firstGetMoney(db):
    current_player = -1

    dbms = DatabaseManager(db)

    sqlstr = "SELECT player_id, happen_time, minor_class FROM maidian ORDER BY player_id,happen_time ASC;"
    data_iterator = dataGen(dbms, sqlstr)

    keyList = ["充值", "开服累冲", "购", "礼", "总"]
    find_flag = False
    correct_exit = True

    starting_time = time.time()

    counter = 0

    enter_time = 0
    first_access_time = 0

    stay_time = 0
    accessDict = {}

    for row in data_iterator:

        counter += 1
        if counter % 1000000 == 0:
            print("%s lines processed\n" % counter)
            # print("total enter:" + str(num_enter))
            # print("correct exit :" + str(num_of_correct_exit))

        player_id = row[0]
        timestamp = row[1]
        minor_action = row[2]

        # 如果更换用户，初始化所有值，并且计算上一个用户的所有以及正确的停留的时间。
        if player_id != current_player:

            if first_access_time > 0:
                stay_time = first_access_time - enter_time

            if current_player != -1:
                accessDict[current_player] = stay_time

            enter_time = timestamp
            stay_time = 0
            find_flag = False
            first_access_time = 0
            current_player = player_id

        # 如果被标记，该用户所有剩余action都忽略
        if find_flag:
            continue

        # 如果对某一个用户，看是否是充值活动,记录进入时间
        if sum([1 for e in keyList if e in minor_action]) > 0:
            find_flag = True
            first_access_time = timestamp
            continue

    if first_access_time > 0:
        stay_time = first_access_time - enter_time
    accessDict[current_player] = stay_time

    pd_dist = pd.DataFrame(list(accessDict.items()), columns=['Player', 'Time'])

    return pd_dist

def timePlayAllTime(db, threshold):

    begin_date = datetime.strptime("2017-05-18", "%Y-%m-%d")
    ##可以根据latestPlayDate函数找到玩家的玩的最晚一天，这里先写死
    during_days = 6

    intervalGen = IntervalGenerator(begin_date.timestamp(), days=during_days)
    final_df = []
    for ig in intervalGen.daysGenerator():
        begin_date = ig.begin_interval
        end_date = ig.end_interval
        dayOneDF = timePlayOneDay(db, begin_date, end_date, threshold)
        final_df.append(dayOneDF)

    result = pd.concat(final_df)

    firstPlayTime = result.groupby("Player").apply(lambda x: firstNonZeroTime(x))

    ## 1%的用户玩的时间很少，应该只点击了一两次左右，删掉
    print("第一天玩的记录时间为零的用户比例为：".format(firstPlayTime.apply(lambda x: len(x) == 0).sum() / firstPlayTime.shape[0]))
    print("剔除以上用户")

    ##找到流失玩家－－只玩了一天的玩家
    lossu = result.groupby("Player").apply(lambda x: lossUsers(x))

    resDF = lossu.to_frame("Count")
    resDF['Duration'] = firstPlayTime
    # resDF = resDF[resDF['Duration'].apply(lambda x: len(x) != 0)]
    # resDF['Duration'] = resDF['Duration'].apply(lambda x: x[0])
    resDF = resDF.reset_index()
    resDF['Loss'] = resDF['Count'] > 1
    resDF.drop('Count', axis=1, inplace=True)

    return resDF

def userProfile(db):

    moneyU = findMoneyUser(db)
    freq = clickFrequency(db)
    freq['money'] = freq['Player'].isin(moneyU['user'])

    resDF = timePlayAllTime(db, 300)
    resDF['Money'] = freq['money']
    resDF['Freq'] = freq['Freq']

    ##玩家首次尝试点击付费相关的界面元素时间
    firstMoney = firstGetMoney(db)
    resDF['FirstTime'] = firstMoney['Time']

    ##玩家玩法停留时间
    enter_action = "世界地图 / 世界地图 / 【主】创建部队"
    exit_action = "世界地图 / 世界地图 / 【部队】添加部队"

    ustay = ui_stay_time(enter_action, exit_action)
    resDF['TimeOnUI'] = ustay['TimeOnUI']

    ##清洗数据
    resDF = resDF[resDF['Duration'].apply(lambda x: len(x) != 0)]
    resDF['Duration'] = resDF['Duration'].apply(lambda x: x[0])

    cols = resDF.columns.tolist()
    new_cols = cols[:2] + cols[4:] + cols[2:4]
    resDF=resDF[new_cols]

    resDF.to_csv("用户画像.csv", index= False)

def poly_fit(l,poly_deg):
    y = l.values
    x = l.index.tolist()
    z = np.polyfit(x=x, y=y, deg=poly_deg)
    p = np.poly1d(z)
    return p.c

def timeSeriesModel(db, sqlstr, feature_name, poly_deg = 4):

    conn = sqlite3.connect(db)
    keyIndexData = pd.read_sql_query(sqlstr, conn)

    # keyIndex = keyIndexData.loc[keyIndexData['yonghu_id'] == 1039, 'jinbi']
    user_trend = keyIndexData.groupby("yonghu_id").apply(lambda x: poly_fit(x[feature_name],poly_deg))
    user_para = pd.DataFrame()

    for i in range(4 + 1):
        name = "poly_" + str(4 - i)
        user_para.loc[:, name] = user_trend.apply(lambda x: x[i])
        # user_para.loc[:, name] = preprocessing.StandardScaler().fit_transform(user_para[name].values.reshape(-1, 1))

    user_para.to_csv("userTrend.csv",index=False)

def randSample(db,sqlstr):
    conn = sqlite3.connect(db)
    keyIndexData = pd.read_sql_query(sqlstr, conn)

    keyIndexData.groupby('yonghu_id').apply(lambda x: x['jinbi'].sample(n=1).values)


def wanfa_analysis(in_action, out_action, data_iterator, loss_window=20):
    time_of_total_stay = 0

    stay_limit = 1500
    time_limit = 600
    num_enter = 0
    num_of_correct_exit = 0

    current_player = -1

    find_flag = False
    correct_exit = True
    window_len = 0
    time_list = []

    starting_time = time.time()
    counter = 0

    enter_time = 0
    total_exit_time = 0
    stay_time = 0

    first_action_time = -1
    last_action_time = -1

    duringDict = {}

    for row in data_iterator:

        counter += 1
        if counter % 1000000 == 0:
            print("%s lines processed\n" % counter)
            # print("total enter:" + str(num_enter))
            # print("correct exit :" + str(num_of_correct_exit))

        player_id = row[0]
        action = row[2]
        timestamp = row[1]

        # 如果更换用户，初始化所有值，并且计算上一个用户的所有以及正确的停留的时间。
        if player_id != current_player:

            find_flag = False
            window_len = 0

            if current_player != -1:
                time_of_total_stay += last_action_time - first_action_time

                duringDict[current_player] = [time_list, time_of_total_stay]

            current_player = player_id

            first_action_time = timestamp
            last_action_time = timestamp

            enter_time = 0
            total_exit_time = 0
            stay_time = 0
            time_of_total_stay = 0
            time_list = []

        ##　如果这次点击的时间跟上一次点击的时间的差小于阈值，则进入计算累计时间。否则重新定义
        # 连续点击的第一次。
        if timestamp - last_action_time > time_limit:
            time_of_total_stay += (last_action_time - first_action_time)
            # if player_id == 37902:
            #     print("total: {0}, current: {1},last_action: {2}, first_action{3}".
            #           format(time_of_total_stay, timestamp, last_action_time, first_action_time))
            first_action_time = timestamp

        # 如果对某一个用户，第一次找到相应的进入UI动作，记录总人数加一并标记。记录进入时间
        if (compareAction(action, in_action) and (not find_flag)):
            find_flag = True
            enter_time = timestamp
            total_exit_time = timestamp
            num_enter += 1
            last_action_time = timestamp
            window_len = 0
            continue

        # 如果已找到进入动作，判断此时动作是否为退出动作。如果是，正确退出加一，否则判断窗口长度是否大于
        # 标准值，如果是的话，此用户记录为非正常退出，并记录时间
        if find_flag:
            if compareAction(action, out_action):
                num_of_correct_exit += 1
                stay_time = (timestamp - enter_time)
                # if player_id == 37902:
                #     print("correct: {0}, current: {1},enter_time: {2}".
                #           format(stay_time, timestamp, enter_time))
                if timestamp - last_action_time <= time_limit:
                    time_list.append(stay_time)

                find_flag = False
            else:
                window_len += 1
                if window_len >= loss_window or (timestamp - last_action_time) > time_limit:

                    stay_time = (last_action_time - enter_time)
                    # if player_id == 37902:
                    #     print("non-correct: {0}, current: {1},enter_time: {2}".
                    #           format(stay_time, timestamp, enter_time))
                    time_list.append(stay_time)
                    find_flag = False

        last_action_time = timestamp

    time_of_total_stay += last_action_time - first_action_time

    duringDict[current_player] = [time_list, time_of_total_stay]

    pd_dist = pd.DataFrame(list(duringDict.items()), columns=['Player', 'TimeOnUI'])

    return pd_dist


if __name__ == "__main__":
    db = "/home/maoan/maidianAnalysis/level2-uianalysis/world_seven.db"
    db2 = "/home/maoan/maidianAnalysis/xiamen/xiamen_1.db"
    db3 = "/home/maoan/maidianAnalysis/xiamen/1308310007.db"
    db4 = "/home/maoan/maidianAnalysis/xiamen/xiamen_1b.db"

    feature = "jinbi"
    sqlstr = "SELECT yonghu_id, wanjia_id," + feature + " FROM maidian ORDER BY yonghu_id,timestamp ASC;"

    # clickFrequency(db)
    # enter_action = "世界地图 / 世界地图 / 【主】创建部队"
    # exit_action = "世界地图 / 世界地图 / 【部队】添加部队"
    # dbms = DatabaseManager(db)
    # query_sql = "SELECT player_id,action,happen_time FROM maidian ORDER BY player_id,happen_time ASC"
    #
    # ustay = ui_stay_time(enter_action,exit_action, dataGen(dbms,query_sql))
    # userProfile(db)
    timeSeriesModel(db4,sqlstr,"jinbi",)