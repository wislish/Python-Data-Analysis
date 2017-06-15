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


def interval_last(x, begin_date, end_date):
    return x.loc[(x['happen_time'] >begin_date) & (x['happen_time'] < end_date),].tail(1)['total_power'].values

def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    same valued added to a same key
    """
    result = collections.defaultdict(list)
    for dictionary in dict_args:
        for k in dictionary.keys():
            result[k].append(dictionary.get(k))
    return result

def keyIndexGrowthByActions(index, user_id, timestamp, action, interval_in_secs, db, growth = True, player=None):

    dbms = DatabaseManager(db)

    # sqlstr = "SELECT yonghu_id, " + index + ", timestamp, action FROM maidian ORDER BY yonghu_id,timestamp ASC;"
    sqlstr = "SELECT " + user_id + ", " + index + ", "+timestamp + ", " + action +" FROM maidian ORDER BY " +user_id+","+timestamp+ " ASC;"
    print(sqlstr)
    data_iterator = dataGen(dbms, sqlstr)

    if growth:
        changeF = 1
    else:
        changeF = -1

    current_player = -1
    first_action_time = -1
    last_action_time = -1
    last_key_value = -1
    one_block_time = 0
    num_growth = 0
    one_interval_clicks_list = 0
    time_limit = 300

    total_index_dict = collections.defaultdict(list)
    action_contribution_dict = collections.defaultdict(list)
    player_list = []
    starting_time = time.time()
    counter = 0
    # dist = collections.defaultdict(int)

    for row in data_iterator:
        counter += 1
        if counter % 100000 == 0:
            print("%s lines processed\n" % counter)
            # print(timestamp)

        key_factor = row[1]
        player_id = row[0]
        timestamp = row[2]
        action = row[3]
        # 如果更换用户，初始化所有值
        if player_id != current_player:

            temp_list = []
            for k, v in action_contribution_dict.items():
                temp_list.append([k, v])

            player_list.append(temp_list)

            first_action_time = timestamp
            last_action_time = timestamp
            one_block_time = 0
            ##忽略最后不满一小时的数据
            # player_list.append(last_key_value)

            if current_player != -1:
                total_index_dict[current_player] = player_list

            action_contribution_dict.clear()
            player_list = []
            last_key_value = key_factor
            last_action = action
            current_player = player_id

        # if major_action == None or major_action == "启动":
        #     continue

        if timestamp - last_action_time <= time_limit:
            during_time = last_action_time - first_action_time
            if during_time + one_block_time >= interval_in_secs:
                temp_list = []
                for k, v in action_contribution_dict.items():
                    temp_list.append([k, v])

                player_list.append(temp_list)
                first_action_time = timestamp
                one_block_time = 0
                action_contribution_dict.clear()
        else:
            one_block_time += last_action_time - first_action_time
            first_action_time = timestamp

            ## 如果关键指标增长了，记录增长量以及增长次数。
        if (key_factor-last_key_value) * changeF > 0:
            diff = (key_factor - last_key_value)*changeF
            # num_growth += 1
            temp_tuple = action_contribution_dict.get(last_action, [0, 0])
            temp_tuple[0] += diff
            temp_tuple[1] += 1
            action_contribution_dict[last_action] = temp_tuple

        last_action_time = timestamp
        last_key_value = key_factor
        last_action = action

    temp = pd.DataFrame(list(total_index_dict.items()), columns=['Player', 'KeyFactor'])

    # if player:
    #     single_user = temp.loc[temp['Player'] == player,]
    #     for i in range(3):
    #         single_user["actions"].apply(lambda x: x[i])
    #         sns.factorplot(x="who", y="survived", col="class", data=)
    #
    #     single_user['指标增长量最大值'] = single_user['Growth'].apply(lambda x: np.max([e[0] for e in x]))
    if player:
        single_user = temp.loc[temp['Player'] == player,]
        final_df = []
        increase_times = []
        for i in range(len(single_user['KeyFactor'].values[0])):
            actions = pd.Series(single_user["KeyFactor"].apply(lambda x: [e[0] for e in x[i]]).values[0])
            growth = pd.Series(single_user["KeyFactor"].apply(lambda x: [e[1][0] for e in x[i]]).values[0])
            times = pd.Series(single_user["KeyFactor"].apply(lambda x: [e[1][1] for e in x[i]]).values[0])
            plotDF = pd.DataFrame({'Actions': actions, 'Growth': growth, 'Times': times})
            plotDF["第几个小时"] = i
            final_df.append(plotDF)
            increase_times.append(sum(times.values))
            if i < 3:
                name = str(player) + "_" + str(i) + "_小时动作贡献表.csv"
            plotDF.to_csv(name, encoding="utf_8", index=False)

        name = str(player) + "_动作对关键指标贡献表.csv"
        result = pd.concat(final_df)
        # result.to_csv(name, encoding="utf_8", index=False)

        name2 = str(player) + "_战力提高次数分布.csv"
        pd.Series(increase_times).to_csv(name2)

    key_factor_list = temp['KeyFactor'].values
    max_continue_hour = np.max([len(i) for i in key_factor_list])

    final_df = []
    print("num " + str(max_continue_hour))
    for num_hour in range(max_continue_hour):

        hour_list = [l[num_hour] for l in key_factor_list if len(l) > num_hour]
        index_growth_list = []
        for j in range(len(hour_list)):
            ulist = hour_list[j]
            index_growth_list.append(dict((ele[0], (ele[1][0], ele[1][1])) for ele in ulist))

        growth_res = merge_dicts(*index_growth_list)
        pd_dist = pd.DataFrame(list(growth_res.items()), columns=['Action', 'Growth'])
        pd_dist['小时序列'] = num_hour
        pd_dist['人数'] = pd_dist['Growth'].apply(lambda x: len(x))
        pd_dist['指标变化量最大值'] = pd_dist['Growth'].apply(lambda x: np.max([e[0] for e in x]))
        pd_dist['指标变化量最小值'] = pd_dist['Growth'].apply(lambda x: np.min([e[0] for e in x]))
        pd_dist['指标变化量中位数'] = pd_dist['Growth'].apply(lambda x: np.median([e[0] for e in x]))
        pd_dist['指标变化量平均值'] = pd_dist['Growth'].apply(lambda x: np.mean([e[0] for e in x]))
        pd_dist['指标变化量总值'] = pd_dist['指标变化量平均值']* pd_dist['人数']
        pd_dist['指标出现次数最大值'] = pd_dist['Growth'].apply(lambda x: np.max([e[1] for e in x]))
        pd_dist['指标出现次数最小值'] = pd_dist['Growth'].apply(lambda x: np.min([e[1] for e in x]))
        pd_dist['指标出现次数中位数'] = pd_dist['Growth'].apply(lambda x: np.median([e[1] for e in x]))
        pd_dist['指标出现次数平均值'] = pd_dist['Growth'].apply(lambda x: np.mean([e[1] for e in x]))

        final_df.append(pd_dist)

    # for i in range(max_continue_hour):
    #     hour_list = [l[i] for l in key_factor_list if len(l) > i]
    #     print("For Hour {0}, there are {1} users ".format(i, len(hour_list)))
    #     print("Minimum {0} is {1}".format(index, np.min(hour_list)))
    #     print("Maximun {0} is {1}".format(index, np.max(hour_list)))
    #     print("Mean {0} is {1:0.2f}".format(index, np.mean(hour_list)))
    #     print("Median {0} is {1}".format(index, np.median(hour_list)))
    #     print("================================\n")
    result = pd.concat(final_df)
    result.to_csv("关键指标跟动作的关系.csv", encoding="utf_8", index=False)


def keyIndexGrowthTimes(sqlstr, interval_in_secs, db, feature_name):

    num_growth = 0

    dbms = DatabaseManager(db)
    # sqlstr = "SELECT player_id, " + index + ", happen_time, major_class FROM maidian ORDER BY player_id,happen_time ASC;"
    data_iterator = dataGen(dbms, sqlstr)

    current_player = -1
    first_action_time = -1
    last_action_time = -1
    last_key_value = -1
    one_block_time = 0

    total_index_dict = collections.defaultdict(list)
    player_list = []
    starting_time = time.time()
    counter = 0
    # dist = collections.defaultdict(int)

    for row in data_iterator:

        counter += 1
        if counter % 100000 == 0:
            print("%s lines processed\n" % counter)
            # print(timestamp)

        key_factor = row[2]
        player_id = row[0]
        timestamp = row[1]
        # major_action = row[3]

        # 如果更换用户，初始化所有值
        if player_id != current_player:
            first_action_time = timestamp
            last_action_time = timestamp
            one_block_time = 0
            ##忽略最后不满一小时的数据
            # player_list.append(last_key_value)


            if current_player != -1 and len(player_list) != 0:
                total_index_dict[current_player] = player_list

            player_list = []
            num_growth = 0
            current_player = player_id
            last_key_value = key_factor

        # ## 如果在启动阶段，用户的关键指标可能是０，因为需要从服务器取得数据。所以忽略。
        # if major_action == None or major_action == "启动":
        #     continue
        ##　如果这次点击的时间跟上一次点击的时间的差小于阈值，则进入计算累计时间。否则重新定义
        # 连续点击的第一次。
        if timestamp - last_action_time <= interval_in_secs:
            during_time = last_action_time - first_action_time
            ## 如果这一次的时间差加上前面的累计时间大于阈值，则记录上一次
            # 的值为上一个时间段的关键指标的值
            if during_time + one_block_time >= interval_in_secs:
                player_list.append(num_growth)
                first_action_time = timestamp
                one_block_time = 0
                num_growth = 0

            ## 记录上一次的累计时间
        else:
            one_block_time += last_action_time - first_action_time
            first_action_time = timestamp

        if key_factor > last_key_value:
            diff = key_factor - last_key_value
            num_growth += 1

            # temp_tuple = action_contribution_dict.get(last_action, [0, 0])
            # temp_tuple[0] += diff
            # temp_tuple[1] += 1
            # action_contribution_dict[last_action] = temp_tuple

        last_action_time = timestamp
        last_key_value = key_factor

    if len(player_list) != 0:
        player_list.append(num_growth)

    total_index_dict[current_player] = player_list

    pd_dist = pd.DataFrame(list(total_index_dict.items()), columns=['Player', 'KeyFactorTimes'])
    # pd_dist['KeyFactor'].apply(lambda x : len(x) != 0)
    # pd_dist.to_csv("关键指标小时增长情况.csv", encoding="utf_8", index=False)

    key_factor_list = pd_dist['KeyFactorTimes'].values
    max_continue_hour = np.max([len(i) for i in key_factor_list])

    max_list = []
    min_list = []
    mean_list =[]
    median_list = []
    num_list = []
    # num_list = []
    len_list = []

    for i in range(max_continue_hour):
        hour_list = [l[i] for l in key_factor_list if len(l) > i]
        print("For Interval {0}, there are {1} users ".format(i, len(hour_list)))
        print("================================\n")
        len_list.append(len(hour_list))
        min_list.append(np.min(hour_list))
        max_list.append(np.max(hour_list))
        mean_list.append(np.mean(hour_list))
        median_list.append(np.median(hour_list))
        # print("================================\n")
        num_list.append(i)

        max_index_value = pd.Series(max_list)
        min_index_value = pd.Series(min_list)
        mean_index_value = pd.Series(mean_list)
        median_index_value = pd.Series(median_list)
        num_hour = pd.Series(num_list)
        num_user = pd.Series(len_list)
        plot_df = pd.DataFrame({"时间段": num_hour, "人数": num_user,
                                "最大指标值": max_index_value,
                                "最小指标值": min_index_value,
                                "指标均值": mean_index_value,
                                "指标中位数": median_index_value})
        name = feature_name + "_" + str(interval_in_secs) + "_成长次数.csv"
        plot_df.to_csv(name, encoding="utf_8", index=False)

def keyIndexGrowthByHours(sqlstr, interval_in_secs, db, feature_name, player = None, days=3):
    num_enter = 0

    dbms = DatabaseManager(db)
    # sqlstr = "SELECT player_id, " + index + ", happen_time, major_class FROM maidian ORDER BY player_id,happen_time ASC;"
    data_iterator = dataGen(dbms, sqlstr)

    current_player = -1
    first_action_time = -1
    last_action_time = -1
    last_key_value = -1
    one_block_time = 0
    time_limit = 300

    total_index_dict = collections.defaultdict(list)
    player_list = []
    starting_time = time.time()
    counter = 0
    # dist = collections.defaultdict(int)

    for row in data_iterator:

        counter += 1
        if counter % 100000 == 0:
            print("%s lines processed\n" % counter)
            # print(timestamp)

        key_factor = row[2]
        player_id = row[0]
        timestamp = row[1]
        # major_action = row[3]

        # if player_id ==252201622525511796:
        #     continue

        # 如果更换用户，初始化所有值
        if player_id != current_player:
            first_action_time = timestamp
            last_action_time = timestamp
            one_block_time = 0
            ##忽略最后不满一小时的数据
            # player_list.append(last_key_value)
            if len(player_list) == 0:
                player_list.append(last_key_value)

            if current_player != -1:
                total_index_dict[current_player] = player_list

            player_list = []
            current_player = player_id

        # ## 如果在启动阶段，用户的关键指标可能是０，因为需要从服务器取得数据。所以忽略。
        # if major_action == None or major_action == "启动":
        #     continue

        ##　如果这次点击的时间跟上一次点击的时间的差小于阈值，则进入计算累计时间。否则重新定义
        # 连续点击的第一次。
        if timestamp - last_action_time <= time_limit:
            during_time = last_action_time - first_action_time
            ## 如果这一次的时间差加上前面的累计时间大于阈值，则记录上一次
            # 的值为上一个时间段的关键指标的值
            if during_time + one_block_time >= interval_in_secs:
                player_list.append(last_key_value)

                first_action_time = timestamp
                one_block_time = 0

            ## 记录上一次的累计时间
        else:
            one_block_time += last_action_time - first_action_time
            first_action_time = timestamp


        last_action_time = timestamp
        last_key_value = key_factor

    if len(player_list) == 0:
        player_list.append(last_key_value)

    total_index_dict[current_player] = player_list

    pd_dist = pd.DataFrame(list(total_index_dict.items()), columns=['Player', 'KeyFactor'])
    pd_dist['KeyFactor'].apply(lambda x : len(x) != 0)
    # pd_dist.to_csv("所有用户关键指标小时增长情况.csv", encoding="utf_8",index=False)

    if player:
        single_user = pd_dist.loc[pd_dist['Player'] == player,]
        hour_index = pd.Series(single_user['KeyFactor'].values[0])
        name = str(player) + "_关键指标小时增长.csv"

        hour_index.to_csv(name, encoding="utf_8")
        return 0


    key_factor_list = pd_dist['KeyFactor'].values
    max_continue_hour = np.max([len(i) for i in key_factor_list])

    max_list = []
    min_list = []
    mean_list =[]
    median_list = []
    num_list = []
    # num_list = []
    len_list = []
    for i in range(max_continue_hour):
        hour_list = [l[i] for l in key_factor_list if len(l) > i]
        print("For Hour {0}, there are {1} users ".format(i, len(hour_list)))
        len_list.append(len(hour_list))
        # print("Minimum {0} is {1}".format(index, np.min(hour_list)))
        min_list.append(np.min(hour_list))
        # print("Maximun {0} is {1}".format(index, np.max(hour_list)))
        max_list.append(np.max(hour_list))
        # print("Mean {0} is {1:0.2f}".format(index, np.mean(hour_list)))
        mean_list.append(np.mean(hour_list))
        # print("Median {0} is {1}".format(index, np.median(hour_list)))
        median_list.append(np.median(hour_list))
        # print("================================\n")
        num_list.append(i)

    max_index_value = pd.Series(max_list)
    min_index_value = pd.Series(min_list)
    mean_index_value = pd.Series(mean_list)
    median_index_value = pd.Series(median_list)
    num_hour = pd.Series(num_list)
    num_user = pd.Series(len_list)
    plot_df = pd.DataFrame({"时间段":num_hour, "人数":num_user,
                            "最大指标值":max_index_value,
                            "最小指标值": min_index_value,
                            "指标均值":mean_index_value,
                            "指标中位数": median_index_value})
    name = feature_name + "_" + str(interval_in_secs) + ".csv"
    plot_df.to_csv(name, encoding="utf_8", index=False)


def keyIndexGrowthByDays(index, conn, first_day_timestamp, player=None, days=3):

    sqlstr = "SELECT happen_time, player_id, " + index + " FROM maidian ORDER BY player_id,happen_time ASC;"
    indexDF = pd.read_sql_query(sqlstr, conn)

    userlist = indexDF['player_id'].unique()
    print("Total Users: " + str(len(userlist)))
    index_list = []
    indexDF.set_index("player_id", drop=False, inplace=True)

    intervalGen = IntervalGenerator(first_day_timestamp, days=days)

    ## 对于指定的N天内，循环每一天用户的指标成长情况。如果有不连续的情况，则忽略。
    for ig in intervalGen.daysGenerator():

        begin_date = ig.begin_interval
        end_date = ig.end_interval
        print(
            "Begin From {0} - To {1}".format(datetime.fromtimestamp(begin_date), datetime.fromtimestamp(end_date)))
        one_interval_list = []

        inter_data = indexDF.groupby('player_id').apply(lambda x: interval_last(x, begin_date, end_date))
        stay_row = inter_data.apply(lambda x: len(x) != 0)
        print(stay_row.sum())
        if stay_row.sum() == 0:
            print("oops...")
            break
        one_interval_list.extend(inter_data.loc[stay_row].apply(lambda x: x[0]).values)
        index_list.append(one_interval_list)
        stay_id = inter_data.loc[stay_row].index
        indexDF = indexDF.loc[stay_id]


    max_list = []
    min_list = []
    mean_list = []
    median_list = []
    num_list = []


    for j in range(len(index_list)):
        print("For Day {0}".format(j))
        print("Minimum {0} is {1}".format(index, np.min(index_list[j])))
        print("Maximun {0} is {1}".format(index, np.max(index_list[j])))
        print("Mean {0} is {1:0.2f}".format(index, np.mean(index_list[j])))
        print("Median {0} is {1}".format(index, np.median(index_list[j])))
        print("================================\n")

        min_list.append(np.min(index_list[j]))
        max_list.append(np.max(index_list[j]))
        median_list.append(np.mean(index_list[j]))
        mean_list.append(np.median(index_list[j]))
        num_list.append(j)

    max_index_value = pd.Series(max_list)
    min_index_value = pd.Series(min_list)
    mean_index_value = pd.Series(mean_list)
    median_index_value = pd.Series(median_list)
    num_hour = pd.Series(num_list)

    plot_df = pd.DataFrame({"天数": num_hour, "最大指标值": max_index_value,
                            "最小指标值": min_index_value,
                            "指标均值": mean_index_value,
                            "指标中位数": median_index_value})
    name = feature + "_按天增长_csv"
    plot_df.to_csv(name, encoding="utf_8", index=False)

        ######################################################
        ### 对于每一个用户循环，速度太慢，采用上面的 groupby 方法。###
        ######################################################
        # index_df_within_interval = indexDF.loc[(indexDF['happen_time'] > begin_date) & (indexDF['happen_time'] < end_date),]
        # index_df_within_interval.sort_values(by=["player_id","happen_time"], inplace=True, ascending=True)
        # for user in userlist:
        #     print(num)
        #
        #     udata = indexDF.loc[(indexDF['player_id'] == user)
        #                                          & (indexDF['happen_time'] > begin_date) & (indexDF['happen_time'] < end_date),]
        #
        #     if len(udata) > 0:
        #         last_index_value = udata.tail(1)[index]
        #         update_list.append(user)
        #         one_interval_list.extend(last_index_value.values)
        #
        #     num += 1
        #
        # if len(update_list) == 0:
        #     print("oops...")
        #     break
        # print(len(update_list))
        # index_list.append(one_interval_list)
        # userlist = list(update_list)


        # print(indexDF.head())



if __name__ == "__main__":
    # db = "/home/maoan/maidianAnalysis/level2-uianalysis/world_seven.db"
    db = "/home/maoan/maidianAnalysis/level2-uianalysis/world_seven.db"
    db2 = "/home/maoan/maidianAnalysis/xiamen/xiamen_1.db"
    db3= "/home/maoan/maidianAnalysis/xiamen/1308310007.db"
    db4 = "/home/maoan/maidianAnalysis/xiamen/xiamen_1b.db"

    feature = "tili"
    sqlstr = "SELECT yonghu_id, " + feature + ", timestamp, action FROM maidian ORDER BY yonghu_id,timestamp ASC;"
    sqls = "SELECT yonghu_id, timestamp, duiwu_zhanli FROM maidian ORDER BY yonghu_id,timestamp ASC;"

    sqls_kuangbaozhiyi = "SELECT user_id, riqi, zhanli FROM maidian ORDER BY user_id,riqi ASC;"
    # chaosIndex(sqls_kuangbaozhiyi, interval_in_secs=60, db=db3)
    # keyIndexTimes(sqlstr=sqls_kuangbaozhiyi, interval_in_secs=60, db=db3)

    begin_date = datetime.strptime("2016-10-10", "%Y-%m-%d")
    sqlstr_kbzy_tili = "SELECT user_id, tili, riqi FROM maidian ORDER BY user_id,riqi ASC;"

    # features = "player_level, peerage, gold, silver, food, wood, iron, exploit, hero_power, building_power, army_power, peerage_power, territory_power, total_power"
    feature = "zuanshi"
    userp = "yonghu_id"
    timep = "timestamp"
    actionp = "action"
    # new_date = datetime.strptime("2017-05-18", "%Y-%m-%d")
    # keyIndexGrowthByDays(feature, conn, new_date.timestamp())
    # keyIndexGrowthByHours(sqlstr=sqls_kuangbaozhiyi, interval_in_secs=60, db=db3, feature_name="战力")

    # keyIndexGrowthByActions(index=feature, user_id=userp, timestamp=timep, action=actionp, interval_in_secs=864000, db=db, growth=False)
    sql_diff_user = "SELECT yonghu_id, timestamp, duiwu_zhanli FROM maidian WHERE num_days_played = 3 ORDER BY yonghu_id,timestamp ASC;"
    # keyIndexGrowthTimes(sqlstr=sql_diff_user, interval_in_secs=3600, db=db4, feature_name="队伍战力-3天用户")

    keyIndexGrowthTimes(sqlstr=sqls_kuangbaozhiyi, interval_in_secs=3600, db=db3, feature_name="战力" )