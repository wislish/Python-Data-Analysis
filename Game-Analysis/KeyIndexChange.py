import pandas as pd
import numpy as np
import sqlite3
import collections
import time
from datetime import datetime
from datetime import timedelta
import seaborn as sns

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

def chaosIndex(sqlstr, interval_in_secs, db, file_name,player = None):
    num_enter = 0

    dbms = DatabaseManager(db)
    data_iterator = dataGen(dbms, sqlstr)

    current_player = -1
    first_action_time = -1
    last_action_time = -1
    last_key_value = -1
    one_block_time = 0
    time_limit = 300

    total_index_dict = collections.defaultdict(list)
    action_set = set()
    unique_acts = 0
    player_list = []
    starting_time = time.time()
    counter = 0
    # dist = collections.defaultdict(int)

    for row in data_iterator:

        counter += 1
        if counter % 1000000 == 0:
            print("%s lines processed\n" % counter)
            # print(timestamp)

        player_id = row[0]
        timestamp = row[1]
        action = row[2]

        # 如果更换用户，初始化所有值
        if player_id != current_player:
            first_action_time = timestamp
            last_action_time = timestamp
            one_block_time = 0
            ##忽略最后不满一小时的数据
            # player_list.append(last_key_value)

            if current_player != -1 and len(player_list) !=0:
                total_index_dict[current_player] = player_list

            player_list = []
            unique_acts = 0
            action_set.clear()
            current_player = player_id


        ##　如果这次点击的时间跟上一次点击的时间的差小于阈值，则进入计算累计时间。否则重新定义
        # 连续点击的第一次。
        if timestamp - last_action_time <= time_limit:
            during_time = last_action_time - first_action_time
            ## 如果这一次的时间差加上前面的累计时间大于阈值，则记录上一次
            # 的值为上一个时间段的关键指标的值
            if during_time + one_block_time >= interval_in_secs:

                player_list.append(unique_acts)

                first_action_time = timestamp
                one_block_time = 0
                unique_acts = 0

            ## 记录上一次的累计时间
        else:
            one_block_time += last_action_time - first_action_time
            first_action_time = timestamp


        last_action_time = timestamp
        last_action= action
        if action not in action_set:
            action_set.add(action)
            unique_acts += 1

    if len(player_list) != 0:
        player_list.append(unique_acts)

    total_index_dict[current_player] = player_list

    pd_dist = pd.DataFrame(list(total_index_dict.items()), columns=['Player', 'KeyFactor'])

    index = "混乱度"
    key_factor_list = pd_dist['KeyFactor'].values
    max_continue_hour = np.max([len(i) for i in key_factor_list])
    max_list = []
    min_list = []
    mean_list = []
    median_list = []
    num_list = []
    len_list = []

    for i in range(max_continue_hour):
        hour_list = [l[i] for l in key_factor_list if len(l) > i]
        print("For Interval {0}, there are {1} users ".format(i, len(hour_list)))
        len_list.append(len(hour_list))
        print("Minimum {0} is {1}".format(index, np.min(hour_list)))
        min_list.append(np.min(hour_list))
        print("Maximun {0} is {1}".format(index, np.max(hour_list)))
        max_list.append(np.max(hour_list))
        print("Mean {0} is {1:0.2f}".format(index, np.mean(hour_list)))
        mean_list.append(np.mean(hour_list))
        print("Median {0} is {1}".format(index, np.median(hour_list)))
        median_list.append(np.median(hour_list))
        print("================================\n")
        num_list.append(i)

    max_index_value = pd.Series(max_list)
    min_index_value = pd.Series(min_list)
    mean_index_value = pd.Series(mean_list)
    median_index_value = pd.Series(median_list)
    num_hour = pd.Series(num_list)
    num_user = pd.Series(len_list)
    plot_df = pd.DataFrame({"时间段": num_hour, "人数":num_user,
                            "最大指标值": max_index_value,
                            "最小指标值": min_index_value,
                            "指标均值": mean_index_value,
                            "指标中位数": median_index_value})
    name = file_name + "_" + str(interval_in_secs) + ".csv"

    plot_df.to_csv(name, encoding="utf_8", index=False)

    return pd_dist

def indexChangeCumul(db, sqlstr, begin_date, end_date, **kw):

    dbms = DatabaseManager(db)

    # sqlstr = "SELECT yonghu_id, " + intex + ", timestamp FROM maidian ORDER BY yonghu_id,timestamp ASC;"
    data_iterator = dataGen(dbms, sqlstr)
    changeF = 1

    if 'changeF' in kw:
        changeF = -1

    current_player = -1
    #     total_index_dict = collections.defaultdict(list)
    timeDict = {}
    cumulative_index = 0

    counter = 0
    # dist = collections.defaultdict(int)
    print("Enter!")
    for row in data_iterator:
        counter += 1
        if counter % 1000000 == 0:
            print("%s lines processed\n" % counter)
            # print(timestamp)

        player_id = row[0]
        timestamp = row[2]
        key_factor = row[1]

        if player_id != current_player:

            if current_player != -1:
                timeDict[current_player] = cumulative_index

            last_key_value = key_factor
            cumulative_index = 0
            current_player = player_id

        if timestamp > end_date or timestamp < begin_date:
            continue

        # ## 如果在启动阶段，用户的关键指标可能是０，因为需要从服务器取得数据。所以忽略。
        #         if major_action == None or major_action == "启动":
        #             continue

        if (key_factor-last_key_value) * changeF > 0:
            diff = (key_factor-last_key_value) * changeF

            cumulative_index += diff

        last_key_value = key_factor

    timeDict[current_player] = cumulative_index
    pd_dist = pd.DataFrame(list(timeDict.items()), columns=['Player', 'CumulativeChange'])

    return pd_dist

def timeGen(db, sqlstr, begin_date, during_days):

    # begin_date = datetime.strptime("2017-05-18", "%Y-%m-%d")
    ##可以根据latestPlayDate函数找到玩家的玩的最晚一天，这里先写死
    # during_days = 6

    intervalGen = IntervalGenerator(begin_date.timestamp(), days=during_days)
    final_df = []
    i=0
    res = pd.DataFrame()

    for ig in intervalGen.daysGenerator():
        begin_date = ig.begin_interval
        end_date = ig.end_interval
        dayOneDF = indexChangeCumul(db, sqlstr, begin_date, end_date, changeF=-1)
        if i == 0:
            res.loc[:, 'Player'] = dayOneDF['Player']
        name = str(i) + "Day"
        print(name)
        #     res = res.assign({name:dayOneDF['CumulativeChange']})
        res.loc[:, name] = dayOneDF['CumulativeChange']
        i += 1

    times = pd.DataFrame()
    times['Player'] = res['Player']
    for i in range(during_days):
        name = str(i) + "Day"
        times.loc[:, name] = res.apply(lambda x: dayPlay(x, i + 1), axis=1)

    times.to_csv("体力损耗分布.csv", index=False)
    return times

def clickTimes(sqlstr, interval_in_secs, db, feature_name, player = None):
    num_enter = 0

    dbms = DatabaseManager(db)
    data_iterator = dataGen(dbms, sqlstr)

    current_player = -1
    first_action_time = -1
    last_action_time = -1
    last_key_value = -1
    one_block_time = 0
    time_limit = 300

    total_index_dict = collections.defaultdict(list)
    action_set = set()
    unique_acts = 0
    player_list = []
    starting_time = time.time()
    counter = 0
    # dist = collections.defaultdict(int)

    for row in data_iterator:

        counter += 1
        if counter % 1000000 == 0:
            print("%s lines processed\n" % counter)
            # print(timestamp)

        player_id = row[0]
        timestamp = row[1]
        action = row[2]

        # 如果更换用户，初始化所有值
        if player_id != current_player:
            first_action_time = timestamp
            last_action_time = timestamp
            one_block_time = 0

            ##忽略最后不满一小时的数据
            # player_list.append(last_key_value)
            # if len(player_list) == 0:
            #     player_list.append(unique_acts)

            if current_player != -1 and len(player_list) != 0:
                total_index_dict[current_player] = player_list

            player_list = []
            action_set.clear()
            unique_acts = 0
            current_player = player_id


        ##　如果这次点击的时间跟上一次点击的时间的差小于阈值，则进入计算累计时间。否则重新定义
        # 连续点击的第一次。
        if timestamp - last_action_time <= time_limit:
            during_time = last_action_time - first_action_time
            ## 如果这一次的时间差加上前面的累计时间大于阈值，则记录上一次
            # 的值为上一个时间段的关键指标的值
            if during_time + one_block_time >= interval_in_secs:

                player_list.append(unique_acts)

                first_action_time = timestamp
                one_block_time = 0
                unique_acts = 0

            ## 记录上一次的累计时间
        else:
            one_block_time += last_action_time - first_action_time
            first_action_time = timestamp


        last_action_time = timestamp
        # last_action= action
        unique_acts += 1


    if len(player_list) != 0:
        player_list.append(unique_acts)

    total_index_dict[current_player] = player_list

    pd_dist = pd.DataFrame(list(total_index_dict.items()), columns=['Player', 'KeyFactor'])

    index = "点击次数"
    key_factor_list = pd_dist['KeyFactor'].values
    max_continue_hour = np.max([len(i) for i in key_factor_list])
    max_list = []
    min_list = []
    mean_list = []
    median_list = []
    num_list = []
    len_list = []

    for i in range(max_continue_hour):
        hour_list = [l[i] for l in key_factor_list if len(l) > i]
        print("For Interval {0}, there are {1} users ".format(i, len(hour_list)))
        len_list.append(len(hour_list))
        print("Minimum {0} is {1}".format(index, np.min(hour_list)))
        min_list.append(np.min(hour_list))
        print("Maximun {0} is {1}".format(index, np.max(hour_list)))
        max_list.append(np.max(hour_list))
        print("Mean {0} is {1:0.2f}".format(index, np.mean(hour_list)))
        mean_list.append(np.mean(hour_list))
        print("Median {0} is {1}".format(index, np.median(hour_list)))
        median_list.append(np.median(hour_list))
        print("================================\n")
        num_list.append(i)

    max_index_value = pd.Series(max_list)
    min_index_value = pd.Series(min_list)
    mean_index_value = pd.Series(mean_list)
    median_index_value = pd.Series(median_list)
    num_hour = pd.Series(num_list)
    num_user = pd.Series(len_list)
    plot_df = pd.DataFrame({"时间段": num_hour, "人数":num_user,
                            "最大指标值": max_index_value,
                            "最小指标值": min_index_value,
                            "指标均值": mean_index_value,
                            "指标中位数": median_index_value})
    name = feature_name + "_" + str(interval_in_secs) + ".csv"

    plot_df.to_csv(name, encoding="utf_8", index=False)

    return pd_dist

def dayPlay(x, day):
    nonzeroL = x.values[np.nonzero(x.values)]
    if len(nonzeroL) >= day+1:
        return nonzeroL[day]
    else:
        return 0


if __name__ == "__main__":
    db = "/home/maoan/maidianAnalysis/level2-uianalysis/world_seven.db"
    db2 = "/home/maoan/maidianAnalysis/xiamen/xiamen_1.db"
    db3= "/home/maoan/maidianAnalysis/xiamen/1308310007.db"

    feature = "tili"
    sqlstr = "SELECT yonghu_id, " + feature + ", timestamp, action FROM maidian ORDER BY yonghu_id,timestamp ASC;"
    sqls = "SELECT yonghu_id, timestamp, action FROM maidian ORDER BY yonghu_id,timestamp ASC;"

    sqls_kuangbaozhiyi = "SELECT user_id, riqi, action FROM maidian ORDER BY user_id,riqi ASC;"

    # chaosIndex(sqls_kuangbaozhiyi, interval_in_secs=3600, db=db3,file_name="混乱度_狂暴之翼")

    # keyIndexTimes(sqlstr=sqls_kuangbaozhiyi, interval_in_secs=60, db=db3)
    clickTimes(sqlstr=sqls_kuangbaozhiyi, interval_in_secs=3600, db=db3, feature_name="点击次数_狂暴之翼")

    begin_date = datetime.strptime("2016-10-10", "%Y-%m-%d")
    sqlstr_kbzy_tili = "SELECT user_id, tili, riqi FROM maidian ORDER BY user_id,riqi ASC;"

    # timeGen(db3,sqlstr_kbzy_tili,begin_date,3)
    # clickFrequency(db)
    # enter_action = "世界地图 / 世界地图 / 【主】创建部队"
    # exit_action = "世界地图 / 世界地图 / 【部队】添加部队"
    # dbms = DatabaseManager(db)
    # query_sql = "SELECT player_id,action,happen_time FROM maidian ORDER BY player_id,happen_time ASC"
    #
    # ustay = ui_stay_time(enter_action,exit_action, dataGen(dbms,query_sql))
    # userProfile(db)