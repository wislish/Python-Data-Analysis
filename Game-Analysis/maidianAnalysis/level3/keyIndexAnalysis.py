import pandas as pd
import numpy as np
import sqlite3
import collections
import time
from datetime import datetime
from datetime import timedelta
import seaborn as sns

from tools import *

## 输出路径设置,所有输出文件都在此父文件夹下。
parent_file = "关键指标分析结果"

def interval_last(x, begin_date, end_date,timestamp,key_index):
    return x.loc[(x[timestamp] >begin_date) & (x[timestamp] < end_date),].tail(1)[key_index].values

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

def keyIndexChangeByActions(data_iterator, interval_in_secs, growth = True, player=None):
    """
        对于给定的间隔时间，统计造成关键指标变化的action，可以选择是否统计单人，
    以及可以选择从指标增长或者减少任一一方面来统计
    
    :param data_iterator: data generator.[player_id,timestamp,action,key_index]
    :param interval_in_secs: 每隔多少秒统计一下动作分布
    :param growth: 是否统计关键指标增长状况，否则为减少
    :param player: 是否统计单人数据
    :return: 统计结果，DataFrame格式。
            　导出文件:
                所有人：“关键指标跟动作的关系_timestamp.xlsx”
                单人：“playerID_动作对关键指标贡献表_timestamp.xlsx”,“playerID_关键指标提高次数分布_timestamp.xlsx”
    
    """

    ## 1.代表增长
    ## 2.代表减少
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

    ## 如果超过这个阈值，则判断玩家离线
    time_limit = 300

    total_index_dict = collections.defaultdict(list)
    action_contribution_dict = collections.defaultdict(list)
    player_list = []
    starting_time = time.time()
    counter = 0
    # dist = collections.defaultdict(int)

    for row in data_iterator:
        counter += 1
        if counter % 1000000 == 0:
            print("%s lines processed\n" % counter)
            # print(timestamp)

        key_factor = row[3]
        player_id = row[0]
        timestamp = row[1]
        action = row[2]
        # 如果更换用户，初始化所有值
        if player_id != current_player:

            first_action_time = timestamp
            last_action_time = timestamp
            one_block_time = 0

            ## 忽略未达到规定时间的值
            if current_player != -1 and len(player_list) !=0:
                total_index_dict[current_player] = player_list

            action_contribution_dict.clear()
            player_list = []
            last_key_value = key_factor
            last_action = action
            current_player = player_id

        if timestamp - last_action_time <= time_limit:
            during_time = last_action_time - first_action_time
            ## 如果累计时间超过阈值
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

    if len(player_list) !=0:
        total_index_dict[current_player] = player_list

    temp = pd.DataFrame(list(total_index_dict.items()), columns=['Player', 'KeyFactor'])

    ##　如果只需要某一个人的指标增长情况。
    if player:
        if len(temp.loc[temp['Player'] == player,'KeyFactor']) == 0:
            print("该用户所玩时间在一个单位时间段内，无法统计")
            return -1
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
            # if i < 3:
            #     name = str(player) + "_" + str(i) + "_小时动作贡献表.csv"
            # plotDF.to_csv(name, encoding="utf_8", index=False)

        name = str(player) + "_动作对关键指标贡献表"
        result = pd.concat(final_df)
        writeTo(parent_file,name,result)

        name2 = str(player) + "_关键指标提高次数分布"
        writeTo(parent_file,name2,pd.DataFrame({"次数": pd.Series(increase_times)}))

    key_factor_list = temp['KeyFactor'].values
    max_continue_hour = np.max([len(i) for i in key_factor_list])
    final_df = []

    print("最大游戏时间段数:" + str(max_continue_hour))
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

    result = pd.concat(final_df)

    writeTo(parent_file,"关键指标跟动作的关系",result)

    return result


def keyIndexGrowthTimes(data_iterator,interval_in_secs,feature_name):
    """
    对给定的关键指标，统计其在给定的时间间隔里，成长的次数。

    :param data_iterator: 数据generator,[player_id,timestamp,key_index]
    :param interval_in_secs: 统计间隔时间
    :param feature_name: 关键指标名字
    :return: 统计结果, DataFrame 格式.
                导出文件：“feature_name_interval_in_secs_成长次数_timestamp.xlsx”
        
    """

    num_growth = 0

    current_player = -1
    first_action_time = -1
    last_action_time = -1
    last_key_value = -1
    one_block_time = 0
    time_limit = 300


    total_index_dict = collections.defaultdict(list)
    player_list = []
    counter = 0

    for row in data_iterator:

        counter += 1
        if counter % 1000000 == 0:
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

            if current_player != -1 and len(player_list) != 0:
                total_index_dict[current_player] = player_list

            player_list = []
            num_growth = 0
            current_player = player_id
            last_key_value = key_factor

        ##　如果这次点击的时间跟上一次点击的时间的差小于阈值，则进入计算累计时间。否则重新定义
        # 连续点击的第一次。
        if timestamp - last_action_time <= time_limit:
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

        last_action_time = timestamp
        last_key_value = key_factor

    if len(player_list) != 0:
        total_index_dict[current_player] = player_list

    pd_dist = pd.DataFrame(list(total_index_dict.items()), columns=['Player', 'KeyFactorTimes'])

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
        name = feature_name + "_" + str(interval_in_secs) + "_成长次数"

        writeTo(parent_file,name,plot_df)

        return plot_df

def keyIndexGrowthByHours(data_iterator, interval_in_secs,feature_name, player = None):
    """
    对于给定的关键指标，统计其每小时增长情况。
    但也可对任意的间隔时间，以秒为单位，统计增长情况。
    若输入了某单独用户的id，则只会统计一个用户的关键指标增长情况
    ==================================================
    
    :param data_iterator: 数据generator,可以循环,[player_id,timestamp,key_index]
    :param interval_in_secs: 统计间隔时间
    :param feature_name: 关键指标名
    :param player: 是否只统计单人数据
    :return: 最后统计结果, DataFrame格式. 
                导出文件：
                    单人：playerID _关键指标小时增长_timestamp.xlsx
                    所有人：feature_name_interval_in_secs_timestamp.xlsx
    
    """

    current_player = -1
    first_action_time = -1
    last_action_time = -1
    last_key_value = -1
    one_block_time = 0
    time_limit = 300

    total_index_dict = collections.defaultdict(list)
    player_list = []
    counter = 0

    for row in data_iterator:

        counter += 1
        if counter % 1000000 == 0:
            print("%s lines processed\n" % counter)
            # print(timestamp)

        key_factor = row[2]
        player_id = row[0]
        timestamp = row[1]

        # 如果更换用户，初始化所有值
        if player_id != current_player:
            first_action_time = timestamp
            last_action_time = timestamp
            one_block_time = 0

            if current_player != -1 and len(player_list) != 0:
                total_index_dict[current_player] = player_list

            player_list = []
            current_player = player_id

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

    if len(player_list) != 0:
        total_index_dict[current_player] = player_list

    pd_dist = pd.DataFrame(list(total_index_dict.items()), columns=['Player', 'KeyFactor'])
    pd_dist['KeyFactor'].apply(lambda x : len(x) != 0)

    if player:
        single_user = pd_dist.loc[pd_dist['Player'] == player,]
        if len(single_user['KeyFactor'])==0:
            print("该用户游戏时间小于一个单位时间段，无法统计。")
            return -1

        hour_index = pd.Series(single_user['KeyFactor'].values[0])
        name = str(player) + "_关键指标小时增长.csv"

        # hour_index.to_csv(name, encoding="utf_8")
        writeTo(parent_file,name,pd.DataFrame({"小时增长":hour_index}))

        return hour_index


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
    name = feature_name + "_" + str(interval_in_secs)

    writeTo(parent_file,name,plot_df)

    return plot_df


def keyIndexGrowthByDays(sqlstr, db, user_id, timestamp, key_index, first_day_timestamp, feature_name, days=3):
    """
    基于给定的关键指标,统计其从第一天开始，用户关键指标的连续变化情况。
    如果用户不是从第一天加入游戏，则不会参与统计。例如，第一天有５００个
    用户参与游戏，则第二天也只会从这５００个用户中统计第二天还接着玩儿的用户。
    ===========================================================
    
    :param sqlstr: 读取数据库的语句,[player_id,timestamp,key_index]
    :param db: 数据库路径
    :param user_id: 用户ＩＤ在数据库里的字段名称
    :param timestamp: 时间戳在数据库里的字段名称
    :param key_index: 关键指标在数据库里的字段名称
    :param first_day_timestamp: 需要统计的第一天的时间戳
    :param feature_name: 关键指标名，用于存储文件
    :param days: 持续统计天数，从给定的第一天开始,默认为3天
    :return: 最后统计结果, DataFrame格式.
                导出文件：feature_name_按天增长_timestamp.xlsx
    """

    conn = sqlite3.connect(db)
    indexDF = pd.read_sql_query(sqlstr, conn)

    userlist = indexDF[user_id].unique()
    print("Total Users: " + str(len(userlist)))
    index_list = []
    indexDF.set_index(user_id, drop=False, inplace=True)

    intervalGen = IntervalGenerator(first_day_timestamp, days=days)

    ## 对于指定的N天内，循环每一天用户的指标成长情况。如果有不连续的情况，则忽略。
    for ig in intervalGen.daysGenerator():

        begin_date = ig.begin_interval
        end_date = ig.end_interval
        print(
            "Begin From {0} - To {1}".format(datetime.fromtimestamp(begin_date), datetime.fromtimestamp(end_date)))
        one_interval_list = []
        inter_data = indexDF.groupby(user_id).apply(lambda x: interval_last(x, begin_date, end_date,timestamp,key_index))
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
        print("Minimum {0} is {1}".format(feature_name, np.min(index_list[j])))
        print("Maximun {0} is {1}".format(feature_name, np.max(index_list[j])))
        print("Mean {0} is {1:0.2f}".format(feature_name, np.mean(index_list[j])))
        print("Median {0} is {1}".format(feature_name, np.median(index_list[j])))
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
    name = feature_name + "_按天增长"

    writeTo(parent_file,name,plot_df)

    return plot_df
    # plot_df.to_csv(name, encoding="utf_8", index=False)

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

def chaosIndex(data_iterator, interval_in_secs):

    """
    混乱度,量度单位时间内，用户点击全新的动作的次数。
    :param data_iterator: 数据生成器，可循环。[player_id,timestamp,action]
    :param interval_in_secs: 间隔时间,秒.
    :return: 统计结果,DataFrame格式.
                导出文件:混乱度_interval_in_secs_timestamp.xlsx
    """

    num_enter = 0

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
    counter = 0

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
    name = index + "_" + str(interval_in_secs)

    writeTo(parent_file,name,plot_df)

    return plot_df


def indexChangeCumul(db, sqlstr,begin_date, end_date, growth = True):
    """
    统计关键指标在给定的时间区间内的累计变化量，对每个用户而言。
    
    :param data_iterator: 数据生成器，可循环.[player_id,timestamp,key_index]
    :param begin_date: 统计时间区间开始时间戳
    :param end_date: 统计时间区间结束时间戳
    :param growth: 是否统计关键指标增长量的累计变化
    :return: 统计结果, DataFrame格式。
                导出文件:关键指标累计变化_开始日期_结束日期_timestamp.xlsx
    """

    data_iterator = dataGen(db, sqlstr)
    changeF = 1

    if not growth:
        changeF = -1

    current_player = -1
    timeDict = {}
    cumulative_index = 0

    counter = 0
    for row in data_iterator:
        counter += 1
        if counter % 1000000 == 0:
            print("%s lines processed\n" % counter)
            # print(timestamp)

        player_id = row[0]
        timestamp = row[1]
        key_factor = row[2]

        if player_id != current_player:

            if current_player != -1:
                timeDict[current_player] = cumulative_index

            last_key_value = key_factor
            cumulative_index = 0
            current_player = player_id

        if timestamp > end_date or timestamp < begin_date:
            continue

        if (key_factor-last_key_value) * changeF > 0:
            diff = (key_factor-last_key_value) * changeF

            cumulative_index += diff

        last_key_value = key_factor

    timeDict[current_player] = cumulative_index
    pd_dist = pd.DataFrame(list(timeDict.items()), columns=['Player', 'CumulativeChange'])

    name = "关键指标累计变化_"+str(datetime.fromtimestamp(begin_date)) +"_"+ str(datetime.fromtimestamp(end_date))

    writeTo(parent_file,name,pd_dist)

    return pd_dist

def keyIndexChangeByLevel(data_iterator, feature_name):
    """
    根据等级的变化，统计关键指标相应的变化情况，并对第三个指标（一般是战力）
    进行统计。对每一行数据，前两个是用户id,timestamp,后面三个依次是等级,关键指标（例如钻石）
    第三指标(战力)。
    :param data_iterator: 数据生成器，可循环.[player_id,timestamp,level,key_index,zhanli]
    :param feature_name: 关键指标名字
    :return: 统计结果, DataFrame格式.
                导出文件:feature_name+随等级变化情况_timestamp.xlsx
    """

    current_player = -1
    current_level = -1

    increase_times = 0
    increase_vol = 0
    decrease_times = 0
    decrease_vol = 0
    zhanli_increase_times = 0
    zhanli_increase_vol = 0

    last_status = -1
    last_zhanli = -1
    last_key_factor = -1

    key_factor_dict = collections.defaultdict(list)

    counter = 0

    for row in data_iterator:
        counter += 1
        if counter % 1000000 == 0:
            print("%s lines processed\n" % counter)
            # print(timestamp)

        current_status = row[2]
        player_id = row[0]
        timestamp = row[1]
        key_factor = row[3]
        zhanli = row[4]

        # 如果更换用户，初始化所有值
        if player_id != current_player:

            if current_player != -1:
                key_factor_dict[last_status].append(
                    (increase_times, increase_vol, decrease_times, decrease_vol, last_key_factor,
                     zhanli_increase_times, zhanli_increase_vol, last_zhanli))

            increase_times = 0
            increase_vol = 0
            decrease_times = 0
            decrease_vol = 0
            zhanli_increase_times = 0
            zhanli_increase_vol = 0

            # current_status = order_factor
            last_status = current_status
            last_key_factor = key_factor
            last_zhanli = zhanli

            current_player = player_id

        ## 忽略极端值
        # if current_player == 164616 or current_player == 168419:
        #     continue

        if current_status != last_status:
            #             temp_list = key_factor_dict[last_status]
            key_factor_dict[last_status].append(
                (increase_times, increase_vol, decrease_times, decrease_vol, last_key_factor,
                 zhanli_increase_times, zhanli_increase_vol, last_zhanli))
            increase_times = 0
            increase_vol = 0
            decrease_times = 0
            decrease_vol = 0
            zhanli_increase_times = 0
            zhanli_increase_vol = 0

            last_zhanli = zhanli
            last_status = current_status
            last_key_factor = key_factor
            continue

        if key_factor - last_key_factor > 0:
            diff = key_factor - last_key_factor
            increase_times += 1
            increase_vol += diff

        elif key_factor - last_key_factor < 0:
            diff = last_key_factor - key_factor
            decrease_times += 1
            decrease_vol += diff

        if zhanli - last_zhanli > 0:
            diff = zhanli - last_zhanli
            zhanli_increase_times += 1
            zhanli_increase_vol += diff

        last_zhanli = zhanli
        last_status = current_status
        last_key_factor = key_factor

    key_factor_dict[last_status].append(
        (increase_times, increase_vol, decrease_times, decrease_vol, last_key_factor,
         zhanli_increase_times, zhanli_increase_vol, last_zhanli))

    pd_dist = pd.DataFrame(list(key_factor_dict.items()), columns=['Level', 'KeyFactor'])

    pd_dist['人数'] = pd_dist['KeyFactor'].apply(lambda x: len(x))
    pd_dist['指标增加次数-中位数'] = pd_dist['KeyFactor'].apply(lambda x: np.median([e[0] for e in x]))
    pd_dist['指标增加次数-平均数'] = pd_dist['KeyFactor'].apply(lambda x: np.mean([e[0] for e in x]))
    pd_dist['指标增加量-中位数'] = pd_dist['KeyFactor'].apply(lambda x: np.median([e[1] for e in x]))
    pd_dist['指标增加量-平均数'] = pd_dist['KeyFactor'].apply(lambda x: np.mean([e[1] for e in x]))
    pd_dist['指标消耗次数-中位数'] = pd_dist['KeyFactor'].apply(lambda x: np.median([e[2] for e in x]))
    pd_dist['指标消耗次数-平均数'] = pd_dist['KeyFactor'].apply(lambda x: np.mean([e[2] for e in x]))
    pd_dist['指标消耗量-中位数'] = pd_dist['KeyFactor'].apply(lambda x: np.median([e[3] for e in x]))
    pd_dist['指标消耗量-平均数'] = pd_dist['KeyFactor'].apply(lambda x: np.mean([e[3] for e in x]))
    pd_dist['指标留存值-中位数'] = pd_dist['KeyFactor'].apply(lambda x: np.median([e[4] for e in x]))
    pd_dist['指标留存值-平均数'] = pd_dist['KeyFactor'].apply(lambda x: np.mean([e[4] for e in x]))

    pd_dist['战力提升次数-中位数'] = pd_dist['KeyFactor'].apply(lambda x: np.median([e[5] for e in x]))
    pd_dist['战力提升次数-平均数'] = pd_dist['KeyFactor'].apply(lambda x: np.mean([e[5] for e in x]))
    pd_dist['战力提升量-中位数'] = pd_dist['KeyFactor'].apply(lambda x: np.median([e[6] for e in x]))
    pd_dist['战力提升量-平均数'] = pd_dist['KeyFactor'].apply(lambda x: np.mean([e[6] for e in x]))
    pd_dist['战力值-中位数'] = pd_dist['KeyFactor'].apply(lambda x: np.median([e[7] for e in x]))
    pd_dist['战力值-平均数'] = pd_dist['KeyFactor'].apply(lambda x: np.mean([e[7] for e in x]))

    pd_dist.drop("KeyFactor", axis=1,inplace=True)
    pd_dist.sort_values(by="Level",inplace=True)
    name = feature_name+"随等级变化情况"

    writeTo(parent_file,name,pd_dist)

    return pd_dist

def keyIndexChangeByTime(data_iterator, interval_in_secs,feature_name):
    """
    根据时间间隔的变化，统计关键指标相应的变化情况，并对第三个指标（一般是战力）
    进行统计。对每一行数据，前两个是用户id,timestamp,后面两个个依次是关键指标（例如钻石）,
    第三指标(战力)。
    :param data_iterator: 数据生成器，可循环.[player_id,timestamp,key_index,zhanli]
    :param interval_in_secs: 时间间隔，秒
    :param feature_name: 特征名称
    :return: 统计结果, DataFrame格式.
                导出文件:feature_name+随时间变化情况_timestamp.xlsx
    """

    current_player = -1
    current_level = -1

    increase_times = 0
    increase_vol = 0
    decrease_times = 0
    decrease_vol = 0
    zhanli_increase_times = 0
    zhanli_increase_vol = 0

    last_status = -1
    last_zhanli = -1
    last_key_factor = -1

    time_limit = 300
    time_interval = 0

    key_factor_dict = collections.defaultdict(list)

    counter = 0

    for row in data_iterator:
        counter += 1
        if counter % 1000000 == 0:
            print("%s lines processed\n" % counter)
            # print(timestamp)

        player_id = row[0]
        timestamp = row[1]
        key_factor = row[2]
        zhanli = row[3]

        # 如果更换用户，初始化所有值
        if player_id != current_player:
            first_action_time = timestamp
            last_action_time = timestamp
            one_block_time = 0

            increase_times = 0
            increase_vol = 0
            decrease_times = 0
            decrease_vol = 0
            zhanli_increase_times = 0
            zhanli_increase_vol = 0

            last_key_factor = key_factor
            last_zhanli = zhanli
            time_interval = 0

            current_player = player_id

        ## 忽略极端值
        # if current_player == 164616 or current_player == 168419:
        #     continue

        if timestamp - last_action_time <= time_limit:
            during_time = last_action_time - first_action_time
            if during_time + one_block_time >= interval_in_secs:
                time_interval += 1
                key_factor_dict[time_interval].append(
                    (increase_times, increase_vol, decrease_times, decrease_vol, last_key_factor,
                     zhanli_increase_times, zhanli_increase_vol, last_zhanli))

                increase_times = 0
                increase_vol = 0
                decrease_times = 0
                decrease_vol = 0
                zhanli_increase_times = 0
                zhanli_increase_vol = 0

                last_zhanli = zhanli
                last_key_factor = key_factor
                first_action_time = timestamp
                one_block_time = 0
                continue

        else:
            one_block_time += last_action_time - first_action_time
            first_action_time = timestamp

        ## 如果关键指标增长了，记录增长量以及增长次数。
        if key_factor - last_key_factor > 0:
            diff = key_factor - last_key_factor
            increase_times += 1
            increase_vol += diff
        elif key_factor - last_key_factor < 0:
            diff = last_key_factor - key_factor
            decrease_times += 1
            decrease_vol += diff

        if zhanli - last_zhanli > 0:
            diff = zhanli - last_zhanli
            zhanli_increase_times += 1
            zhanli_increase_vol += diff

        last_action_time = timestamp
        last_key_factor = key_factor
        last_zhanli = zhanli

    names = "时间段－" + str(interval_in_secs) + "s"
    pd_dist = pd.DataFrame(list(key_factor_dict.items()), columns=[names, 'KeyFactor'])

    pd_dist['人数'] = pd_dist['KeyFactor'].apply(lambda x: len(x))
    pd_dist['指标增加次数-中位数'] = pd_dist['KeyFactor'].apply(lambda x: np.median([e[0] for e in x]))
    pd_dist['指标增加次数-平均数'] = pd_dist['KeyFactor'].apply(lambda x: np.mean([e[0] for e in x]))
    pd_dist['指标增加量-中位数'] = pd_dist['KeyFactor'].apply(lambda x: np.median([e[1] for e in x]))
    pd_dist['指标增加量-平均数'] = pd_dist['KeyFactor'].apply(lambda x: np.mean([e[1] for e in x]))
    pd_dist['指标消耗次数-中位数'] = pd_dist['KeyFactor'].apply(lambda x: np.median([e[2] for e in x]))
    pd_dist['指标消耗次数-平均数'] = pd_dist['KeyFactor'].apply(lambda x: np.mean([e[2] for e in x]))
    pd_dist['指标消耗量-中位数'] = pd_dist['KeyFactor'].apply(lambda x: np.median([e[3] for e in x]))
    pd_dist['指标消耗量-平均数'] = pd_dist['KeyFactor'].apply(lambda x: np.mean([e[3] for e in x]))
    pd_dist['指标留存值-中位数'] = pd_dist['KeyFactor'].apply(lambda x: np.median([e[4] for e in x]))
    pd_dist['指标留存值-平均数'] = pd_dist['KeyFactor'].apply(lambda x: np.mean([e[4] for e in x]))

    pd_dist['战力提升次数-中位数'] = pd_dist['KeyFactor'].apply(lambda x: np.median([e[5] for e in x]))
    pd_dist['战力提升次数-平均数'] = pd_dist['KeyFactor'].apply(lambda x: np.mean([e[5] for e in x]))
    pd_dist['战力提升量-中位数'] = pd_dist['KeyFactor'].apply(lambda x: np.median([e[6] for e in x]))
    pd_dist['战力提升量-平均数'] = pd_dist['KeyFactor'].apply(lambda x: np.mean([e[6] for e in x]))
    pd_dist['战力值-中位数'] = pd_dist['KeyFactor'].apply(lambda x: np.median([e[7] for e in x]))
    pd_dist['战力值-平均数'] = pd_dist['KeyFactor'].apply(lambda x: np.mean([e[7] for e in x]))

    pd_dist.drop("KeyFactor", axis=1,inplace=True)

    name = feature_name+"随时间变化情况"
    writeTo(parent_file,name,pd_dist)

    return pd_dist

if __name__ == "__main__":

    # sql_kuangbao = "SELECT user_id, riqi, action,zuanshi FROM maidian ORDER BY user_id,riqi ASC;"
    # keyIndexChangeByActions(dataGen(db3, sql_kuangbao), interval_in_secs=3600)

    ## 以下测试,均采用厦门游戏的数据.

    #统计动作对关键指标增长的影响，以１小时分割
    sql_act_zhanli = "SELECT yonghu_id, timestamp, action,duiwu_zhanli FROM maidian ORDER BY yonghu_id,timestamp ASC;"
    keyIndexChangeByActions(dataGen(db4, sql_act_zhanli), interval_in_secs=3600)
    keyIndexChangeByActions(dataGen(db4, sql_act_zhanli), interval_in_secs=3600,player=1039)

    ##统计关键指标增长次数
    sql_zhanli = "SELECT yonghu_id, timestamp, duiwu_zhanli FROM maidian ORDER BY yonghu_id,timestamp ASC;"
    keyIndexGrowthTimes(dataGen(db4,sql_zhanli),interval_in_secs=3600, feature_name="队伍战力")

    ##统计关键指标按小时增长情况
    keyIndexGrowthByHours(dataGen(db4, sql_zhanli), interval_in_secs=3600, feature_name="队伍战力")
    keyIndexGrowthByHours(dataGen(db4, sql_zhanli), interval_in_secs=3600, feature_name="队伍战力",player=1039)

    ##统计关键指标按天增长情况
    begin_date = datetime.strptime("2017-06-01", "%Y-%m-%d")
    end_date = datetime.strptime("2017-06-03", "%Y-%m-%d")
    keyIndexGrowthByDays(sql_zhanli, db4, "yonghu_id", "timestamp", "duiwu_zhanli", begin_date.timestamp(), "队伍战力", days=3)

    ##统计游戏的混乱度。
    sql_act = "SELECT yonghu_id, timestamp, action,duiwu_zhanli FROM maidian ORDER BY yonghu_id,timestamp ASC;"
    chaosIndex(dataGen(db4, sql_act), 3600)

    ##对于每一个用户，统计关键指标累计变化情况，在给定的日期内。
    indexChangeCumul(db4, sql_zhanli, begin_date.timestamp(), end_date.timestamp())

    ##一般说来，可以跟repeatByInterval连用，对给定区间内每一天进行运算。
    indexChangeCumul = repeatByInterval(begin_date,2,parent_file,"关键指标累积变化")(indexChangeCumul)
    indexChangeCumul(db=db4, sqlstr=sql_zhanli)

    ##统计钻石随着等级的变化情况，以及对战力的影响
    sql_level_zuanshi = "SELECT yonghu_id, timestamp, duiwu_level, zuanshi, duiwu_zhanli FROM maidian ORDER BY yonghu_id,duiwu_level ASC;"
    keyIndexChangeByLevel(dataGen(db4, sql_level_zuanshi), "钻石")

    ##统计钻石随着时间的变化情况，以及对战力的影响,这里的时间阈值是１小时
    sql_time_zuanshi = "SELECT yonghu_id, timestamp, zuanshi, duiwu_zhanli FROM maidian ORDER BY yonghu_id,timestamp ASC;"
    keyIndexChangeByTime(dataGen(db4, sql_time_zuanshi),3600,"钻石")
