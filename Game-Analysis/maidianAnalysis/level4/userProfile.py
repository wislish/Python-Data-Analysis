import pandas as pd
import numpy as np
import sqlite3
import collections
import seaborn as sns

from tools import *

import warnings
warnings.simplefilter('ignore', np.RankWarning)

parent_file = "用户特征分析结果"


def latestPlayDate(data_iterator):
    """
    统计所用玩家中最新一次加入游戏的时间
    :param data_iterator: 数据生成器,[player_id,timestamp]
    :return: 
    """

    latestDate = -1
    player = -1

    counter = 0
    # dist = collections.defaultdict(int)
    for row in data_iterator:
        counter += 1
        if counter % 1000000 == 0:
            print("%s lines processed\n" % counter)

        player_id = row[0]
        timestamp = row[1]

        if timestamp > latestDate:
            latestDate = timestamp
            player = player_id

    print("Latest Player {0} - Time:{1}".format(player, latestDate))


def findMoneyUser(data_iterator,keyList,step_limt = 5,money_limit = 100):
    """
        通过action所含关键字选出备选充值动作，即action含有keyList关键字。
    如果这个动作导致重要货币指标的增长，且在规定的步数以内增长值超过阈值，
    则表明为充值动作，该用户为付费用户。
    
        对于每一行数据，前两个为用户id和时间戳，后两个依次为action和货币指标
        
    :param data_iterator: 数据生成器，可循环.[player_id,timestamp,action,money_index]
    :param keyList: 关键字列表
    :param step_limt: 在规定的步数以内货币指标需要增长,才算是付费用户
    :param money_limit: 重要货币指标增长下限
    :return: 统计结果,DataFrame格式.
                导出文件:付费用户_timestamp.xlsx
    
    """
    # keyList = ["充值", "开服累冲", "购", "礼", "总"]

    current_player = -1
    origin_money = -1
    find_indicator = False
    steps = -1
    money_user = False
    money_ulist = []
    step_list = []
    total_index_dict = collections.defaultdict(list)
    counter = 0

    for row in data_iterator:
        counter += 1
        if counter % 1000000 == 0:
            print("%s lines processed\n" % counter)
            # print(timestamp)

        player_id = row[0]
        timestamp = row[1]
        subject_action = row[2]
        gold = row[3]

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

    res = pd.DataFrame({"user":money_ulist, "steps":step_list})
    name = "付费用户"
    writeTo(parent_file,name,res)

    return res

def clickFrequency(data_iterator, interval_limit = 300,minimum_continue_time=300):
    """
    统计每个玩家的点击频率
    :param data_iterator: 数据生成器，循环.[player_id,timestamp]
    :param interval_limit: 如果两次动作的间隔时间大于这个阈值，不算连续动作
    :param minimum_continue_time: 最小连续时间，如果连续游戏时间小于这个值，则忽略计算。
    :return: 统计结果,DataFrame格式.
                导出文件:点击频率_timestamp.xlsx
    """

    first_action_time = -1
    last_action_time = -1
    times = -1.
    current_player = -1
    freq_list = []
    total_index_dict = collections.defaultdict(list)

    counter = 0

    for row in data_iterator:
        counter += 1
        if counter % 1000000 == 0:
            print("%s lines processed\n" % counter)
            # print(timestamp)

        player_id = row[0]
        timestamp = row[1]

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

        if timestamp - last_action_time > interval_limit:

            diff = last_action_time - first_action_time
            if diff >= minimum_continue_time:
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

    name="点击频率"
    writeTo(parent_file,name,pd_dist)
    return pd_dist

def numOfRecords(x,begin_date, end_date):

    interval = x.loc[(x['happen_time'] >begin_date) & (x['happen_time'] < end_date),]
    return (interval['happen_time'].tail(1) - interval['happen_time'].head(1))

# def ui_stay_time(in_action, out_action, loss_window=20):
#
#     dbms = DatabaseManager(db)
#     query_sql = "SELECT player_id,action,happen_time, major_class FROM maidian ORDER BY player_id,happen_time ASC"
#     data_iterator= dataGen(dbms, query_sql)
#
#     stay_limit = 1500
#     num_enter = 0
#     num_of_correct_exit = 0
#
#     current_player = -1
#
#     find_flag = False
#     correct_exit = True
#     window_len = 0
#
#     starting_time = time.time()
#     counter = 0
#
#     enter_time = 0
#     total_exit_time = 0
#
#     stay_time = 0
#     duringDict = {}
#
#     for row in data_iterator:
#
#         counter += 1
#         if counter % 1000000 == 0:
#             print("%s lines processed\n" % counter)
#             # print("total enter:" + str(num_enter))
#             # print("correct exit :" + str(num_of_correct_exit))
#
#         player_id = row[0]
#         major_action = row[3]
#         # 如果更换用户，初始化所有值，并且计算上一个用户的所有以及正确的停留的时间。
#         if player_id != current_player:
#             find_flag = False
#             window_len = 0
#             correct_exit = True
#
#             if current_player != -1:
#                 stay_time = (total_exit_time - enter_time)
#                 duringDict[current_player] = stay_time
#
#             current_player = player_id
#
#             enter_time = 0
#             total_exit_time = 0
#             stay_time = 0
#
#         action = row[1]
#         timestamp = row[2]
#
#         ## 如果操作为启动状态,忽略.
#         if major_action == None or major_action == "启动":
#             continue
#
#         # 如果被标记，该用户所有剩余action都忽略
#         if not correct_exit:
#             continue
#
#         # 如果对某一个用户，第一次找到相应的进入UI动作，记录总人数加一并标记。记录进入时间
#         if (compareAction(action, in_action) and (not find_flag)):
#             find_flag = True
#             enter_time = timestamp
#             total_exit_time = timestamp
#             num_enter += 1
#             continue
#
#         # 如果已找到进入动作，判断此时动作是否为退出动作。如果是，正确退出加一，否则判断窗口长度是否大于
#         # 标准值，如果是的话，此用户记录为非正常退出，并记录时间
#         if find_flag:
#             if compareAction(action, out_action):
#                 num_of_correct_exit += 1
#                 total_exit_time = timestamp
#                 correct_exit = False
#             else:
#                 window_len += 1
#                 ##　如果步数大于阈值,且经历的时间小于规定，记录此时时间为退出时间.
#                 if window_len >= loss_window and (timestamp - enter_time) < stay_limit:
#                     correct_exit = False
#                     total_exit_time = timestamp
#                 ## 如果经历的时间大于规定,则记录退出时间等于开始时间.
#                 if (timestamp - enter_time) > stay_limit:
#                     correct_exit = False
#                     total_exit_time = enter_time
#
#     stay_time = (total_exit_time - enter_time)
#     duringDict[current_player] = stay_time
#
#     pd_dist = pd.DataFrame(list(duringDict.items()), columns=['Player', 'TimeOnUI'])
#
#     return pd_dist

def timePlayOneDay(data_iterator, begin_date, end_date, threshold=300):
    """
    统计某一天玩家的真实游戏时长
    :param data_iterator: 数据生成器，可循环.[player_id,timestamp]
    :param begin_date: 统计时间区间开始时间戳
    :param end_date: 统计时间区间结束时间戳
    :param threshold: 间隔时间阈值，如果两次操作时间间隔大于这个值，则不当做连续动作
    :return: 统计结果,DataFrame格式.
                导出文件:真实游戏时间_开始日期_结束日期_timestamp.xlsx
    """

    current_player = -1
    #     total_index_dict = collections.defaultdict(list)
    timeDict = {}
    during_time = 0

    counter = 0
    # dist = collections.defaultdict(int)
    for row in data_iterator:
        counter += 1
        if counter % 1000000 == 0:
            print("%s lines processed\n" % counter)
            # print(timestamp)

        player_id = row[0]
        timestamp = row[1]
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

        if timestamp - last_time > threshold:
            during_time += (last_time - begin_time)
            begin_time = timestamp

        last_time = timestamp

    during_time += (last_time - begin_time)
    timeDict[current_player] = during_time
    pd_dist = pd.DataFrame(list(timeDict.items()), columns=['Player', 'Duration'])
    name = "真实游戏时间_" + str(datetime.fromtimestamp(begin_date)) + "_" + str(datetime.fromtimestamp(end_date))
    writeTo(parent_file,name,pd_dist)

    return pd_dist

def firstNonZeroTime(x):
    return x.loc[x['Duration']!=0,'Duration'].head(1).values

def lossUsers(x):
    return x.loc[x['Duration'] > 0,'Duration'].count()

def firstGetMoney(data_iterator,keyList):
    """
    用户第一次进入付费界面的时间
    :param data_iterator: 数据生成器.[player_id,timestamp,action]
    :param keyList: 付费关键字列表
    :return: 统计结果,DataFrame格式.
                导出文件:第一次付费时间_timestamp.xlsx
    """
    current_player = -1

    find_flag = False
    correct_exit = True

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
    name = "第一次付费时间"
    writeTo(parent_file,name,pd_dist)
    return pd_dist

def whichInterval(time):
    """
    根据时间戳判断属于一天中的什么时间段
    :param time: 
    :return: 
    """
    morning_b = 8
    morning_e = 12
    noon_b = 12
    noon_e = 14
    afternoon_b = 14
    afternoon_e = 18
    night_b = 20

    hour = datetime.fromtimestamp(time).hour

    if hour >= morning_b and hour <= morning_e:
        return "morning"
    elif hour > noon_b and hour <= noon_e:
        return "noon"
    elif hour > afternoon_b and hour <= afternoon_e:
        return "afternoon"
    else:
        return "night"

def playInterval(data_iterator,interval_limit = 3600):
    """
    找到玩家的最频繁游戏时间，分为早上，中午，下午，晚上
    :param data_iterator: 数据生成器，可循环. [player_id,timestamp]
    :param interval_limit: 如果两次操作时间间隔大于这个阈值，则从新量度所属区间
    :return: 统计结果,DataFrame格式.
                导出文件:最频繁游戏时间区间_timestamp.xlsx
    """

    first_action_time = -1
    last_action_time = -1

    current_player = -1
    interval_list = []
    total_index_dict = collections.defaultdict(list)
    counter = 0

    for row in data_iterator:
        counter += 1
        if counter % 1000000 == 0:
            print("%s lines processed\n" % counter)
            # print(timestamp)

        player_id = row[0]
        timestamp = row[1]
        # level = row[2]

        # 如果更换用户，初始化所有值
        if player_id != current_player:

            interval_list.append(whichInterval(first_action_time))
            most_freq_val = collections.Counter(interval_list).most_common(1)[0][0]
            if current_player != -1:
                total_index_dict[current_player] = most_freq_val

            first_action_time = timestamp
            last_action_time = timestamp
            interval_list = []
            current_player = player_id
            # print("================")

        # if level >= 1:
        #     continue

        if timestamp - last_action_time > interval_limit:
            interval_list.append(whichInterval(first_action_time))
            first_action_time = timestamp

        last_action_time = timestamp

    interval_list.append(whichInterval(first_action_time))
    most_freq_val = collections.Counter(interval_list).most_common(1)[0][0]
    total_index_dict[current_player] = most_freq_val

    pd_dist = pd.DataFrame(list(total_index_dict.items()), columns=['Player', 'Freq'])
    name = "最频繁游戏时间区间"
    writeTo(parent_file,name,pd_dist)

    return pd_dist

def actionDistribution(data_iterator):
    """
    统计用户action点击次数的分布
    :param data_iterator: 数据生成器,[player_id,timestamp,action]
    :return: 统计结果,DataFrame格式.
                导出文件:用户动作点击次数分布_timestamp.xlsx
    """
    current_player = -1

    total_index_dict = collections.defaultdict(list)
    action_contribution_dict = collections.defaultdict(int)
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

            temp_list = []
            for k, v in action_contribution_dict.items():
                temp_list.append([k, v])

                #             player_list.append(temp_list)

            if current_player != -1:
                total_index_dict[current_player] = temp_list

            action_contribution_dict.clear()
            current_player = player_id

        action_contribution_dict[action] += 1

    temp = pd.DataFrame(list(total_index_dict.items()), columns=['Player', 'KeyFactor'])
    name = "用户动作点击次数分布"

    writeTo(parent_file,name,temp)
    return temp

def targetActionTimes(data_iterator,target_action_lists):
    """
    统计指定action的点击次数
    :param data_iterator: 数据生成器,[player_id,timestamp,action]
    :param target_action_lists: 指定动作列表
    :return: 统计结果,DataFrame格式.
                导出文件:指定动作点击次数分布_timestamp.xlsx
    """

    current_player = -1

    total_index_dict = collections.defaultdict(list)
    action_contribution_dict = {}
    counter = 0

    for row in data_iterator:
        counter += 1
        if counter % 1000000 == 0:
            print("%s lines processed\n" % counter)
            # print(timestamp)

        player_id = row[0]
        timestamp = row[1]
        action = row[2]
        # vip = row[3]

        # 如果更换用户，初始化所有值
        if player_id != current_player:

            temp_list = []
            for k, v in action_contribution_dict.items():
                temp_list.append([k, v])

            if current_player != -1:
                total_index_dict[current_player] = temp_list

            action_contribution_dict.clear()

            for k in target_action_lists:
                action_contribution_dict[k] = 0
            player_list = []
            current_player = player_id

        # if vip >= 1:
        #     continue

        if action in target_action_lists:
            action_contribution_dict[action] += 1

    temp_list = []
    for k, v in action_contribution_dict.items():
        temp_list.append([k, v])
    total_index_dict[current_player] = temp_list

    temp = pd.DataFrame(list(total_index_dict.items()), columns=['Player', 'KeyFactor'])
    name = "指定动作点击次数分布"
    writeTo(parent_file,name,temp)

    return temp


def maxClickFreq(data_iterator,ton_n = 10, interval_limit = 300):
    """
        找到用户最大点击频率的操作段,即选出前top_n个高点击频率的操作段
    :param data_iterator: 数据生成器,[player_id,timestamp,action]
    :param ton_n: 选出前top_n个高点击频率的操作段
    :param interval_limit: 如果两次动作的间隔时间大于这个阈值，不算连续动作
    :return: 统计结果,DataFrame格式.
                导出文件:最大点击频率段TOP_n_timestamp.xlsx
    """
    first_action_time = -1
    last_action_time = -1
    times = -1.
    current_player = -1
    freq_list = []
    alist = []
    one_interval_list = []

    total_index_dict = collections.defaultdict(list)

    counter = 0

    for row in data_iterator:
        counter += 1
        if counter % 1000000 == 0:
            print("%s lines processed\n" % counter)
            # print(timestamp)

        player_id = row[0]
        timestamp = row[1]
        # level = row[2]
        action = row[2]

        # 如果更换用户，初始化所有值
        if player_id != current_player:

            diff = last_action_time - first_action_time
            if diff != 0:
                freq = times / diff
                freq_list.append(freq)
                alist.append(one_interval_list)

            if current_player != -1:
                indices = np.argsort(freq_list)[:-(ton_n+1):-1]
                max_freq_list = [freq_list[i] for i in indices]
                max_freq_action_list = [alist[i] for i in indices]
                total_index_dict[current_player] = [max_freq_list,max_freq_action_list]

            first_action_time = timestamp
            last_action_time = timestamp
            freq_list = []
            one_interval_list =[]
            alist = []
            times = -1.
            current_player = player_id
            # print("================")
        #         if player_id == 36028840385154595:

        #             print("last " + str(last_action_time))
        #             print("first " + str(first_action_time))
        #             print("now " + str(timestamp))
        # ignore actions that had been done after the vip registering.
        # if level >= 1:
        #     continue

        if timestamp - last_action_time > interval_limit:

            diff = last_action_time - first_action_time
            if diff != 0:
                freq = times / diff
                freq_list.append(freq)
                alist.append(one_interval_list)
            one_interval_list= []
            first_action_time = timestamp
            times = -1.

        times += 1
        last_action_time = timestamp
        one_interval_list.append(action)

    diff = last_action_time - first_action_time
    if diff != 0:
        freq = times / diff
        freq_list.append(freq)
        alist.append(one_interval_list)

    indices = np.argsort(freq_list)[:-(ton_n + 1):-1]
    max_freq_list = [freq_list[i] for i in indices]
    max_freq_action_list = [alist[i] for i in indices]
    total_index_dict[current_player] = [max_freq_list, max_freq_action_list]

    pd_dist = pd.DataFrame(list(total_index_dict.items()), columns=['Player', 'FreqAction'])

    name = "最大点击频率段_TOP"+str(ton_n)
    writeTo(parent_file,name,pd_dist)

    return pd_dist

def findVIPActions(data_iterator, in_action):
    """
    找到付费玩家变成VIP之前的动作。如果当前action在付费action列表中，
    且重要货币指标有增长，则此时用户变为付费用户，停止统计action。
    
    对于每一行数据，前两个是用户id和时间戳，后两个依次是
    action以及重要的货币指标（例如钻石）
    :param data_iterator: 数据生成器,[player_id,timestamp,action,money_index]
    :param in_action: 可能的付费action
    :return: 统计结果,DataFrame格式.
                导出文件:付费玩家付费前点击动作_timestamp.xlsx
    """

    current_player = -1
    counter = 0
    action_contribution_dict = collections.defaultdict(int)
    total_index_dict = collections.defaultdict(list)

    for row in data_iterator:
        counter += 1
        if counter % 1000000 == 0:
            print("%s lines processed\n" % counter)

        zuanshi = row[3]
        action = row[2]
        player_id = row[0]
        timestamp = row[1]

        # 如果更换用户，初始化所有值
        if player_id != current_player:

            vip = False

            temp_list = []
            for k, v in action_contribution_dict.items():
                temp_list.append([k, v])

            # player_list.append(temp_list)
            if current_player != -1:
                total_index_dict[current_player] = temp_list

            action_contribution_dict.clear()
            alist = []
            current_player = player_id
            last_zuanshi = zuanshi

        if vip:
            continue

        if (compareAction(action, in_action)):
            if zuanshi > last_zuanshi:
                vip = True
            continue

        action_contribution_dict[action] += 1
        last_zuanshi = zuanshi

    temp_list = []
    for k, v in action_contribution_dict.items():
        temp_list.append([k, v])
    total_index_dict[current_player] = temp_list

    pd_dist = pd.DataFrame(list(total_index_dict.items()), columns=['Player', 'Actions'])
    name = "付费玩家付费前点击动作"

    writeTo(parent_file,name,pd_dist)
    return pd_dist

def ui_time_ratio(in_action, out_action, data_iterator,time_limit = 600, loss_window=20):
    """
    在某个ＵＩ或者玩法的停留时间占所有游戏时间的比值。
    
    :param in_action: ui进入动作
    :param out_action: ui退出动作
    :param data_iterator: 数据生成器，[player_id,timestamp,action]
    :param time_limit: 如果两次点击的时间差小于这个阈值，则进入计算累计时间。否则重新定义连续点击的第一次。
    :param loss_window: 丢失窗口，如果距离进入ui的动作的步数大于这个窗口值，则计算这一次
                        目标ui的玩法停留比率，并开始寻找下一个ui的停留时间。
    :return: 统计结果,DataFrame格式.
                导出文件:重要ui停留时间比率_timestamp.xlsx
    """

    time_of_total_stay = 0

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
    pd_dist.loc[:, 'BattleRatio'] = pd_dist['TimeOnUI'].apply(lambda x: sum(x[0]) / (x[1] + 1))

    name ="重要ui停留时间比率"
    writeTo(parent_file,name,pd_dist)
    return pd_dist

def merge_counter(x,counters):
    temp = {ele[0]: ele[1] for ele in x}
    for key in temp.keys():
        counters[key].append(temp[key])

def findOutLierActions(res,nonliush,real_vip,topN=20):

    non_vip_stay_users = res.loc[nonliush['stay'],].loc[~real_vip, 'Actions']
    vip_stay_users = res.loc[nonliush['stay'],].loc[real_vip, 'Actions']
    stay_users = res.loc[nonliush['stay'], 'Actions']

    vip_stay_total_dist = collections.defaultdict(list)
    vip_stay_users.apply(lambda x: merge_counter(x, vip_stay_total_dist))
    vip_c = collections.Counter({k: np.median(vip_stay_total_dist[k]) for k in vip_stay_total_dist.keys()})
    vip_top_actions = [ele[0] for ele in vip_c.most_common(topN)]

    non_vip_stay_total_dist = collections.defaultdict(list)
    non_vip_stay_users.apply(lambda x: merge_counter(x, non_vip_stay_total_dist))
    non_vip_c = collections.Counter({k: np.median(non_vip_stay_total_dist[k]) for k in non_vip_stay_total_dist.keys()})
    non_vip_top_actions = [ele[0] for ele in non_vip_c.most_common(topN)]

    stay_total_dist = collections.defaultdict(list)
    stay_users.apply(lambda x: merge_counter(x, stay_total_dist))
    diff_action = collections.Counter({k: np.std(stay_total_dist[k]) for k in stay_total_dist.keys()})
    diff_top_actions = [ele[0] for ele in diff_action.most_common(topN//2)]

    name_vip = "非流失付费用户动作"
    name_non_vip = "非流失非付费用户动作"
    name_diff= "非流失方差大动作"

    writeTo(parent_file,name_vip)

    return nonliush

if __name__ == "__main__":

    ## 以下测试.
    sql_stand = "SELECT yonghu_id, timestamp, action FROM maidian ORDER BY yonghu_id,timestamp ASC;"
    sql_vip = "SELECT yonghu_id,timestamp,action,zuanshi FROM maidian ORDER BY wanjia_id,timestamp ASC;"
    sql_level = "SELECT yonghu_id,timestamp,action duiwu_level FROM maidian ORDER BY yonghu_id,timestamp ASC;"

    begin_date = datetime.strptime("2017-06-01", "%Y-%m-%d")
    end_date = datetime.strptime("2017-06-02", "%Y-%m-%d")

    enter_action = "UIRoot2D/ModalPanel/TeamEditWin/Btn_Start"
    exit_action = "UIRoot2D/NormalPanel/BattleSettlementWin/Img_Bg/Go_Btn/Btn_Quit"

    latestPlayDate(data_iterator=dataGen(db4,sql_stand))

    ## 三十六计
    vip_key_list = ["充值", "开服累冲", "购", "礼", "总"]
    sql_36 = "SELECT player_id, happen_time, minor_class, gold FROM maidian ORDER BY player_id,happen_time ASC;"
    findMoneyUser(dataGen(db,sql_36),keyList = vip_key_list)
    firstGetMoney(dataGen(db,sql_36),vip_key_list)

    timePlayOneDay(dataGen(db4,sql_stand),begin_date.timestamp(),end_date.timestamp())

    with open("./payActions", "r") as f:
        viplist = [ele.strip() for ele in f.readlines()]
    for i in range(15):
        viplist.append(viplist[5].format(i, i))
    del viplist[5]
    findVIPActions(dataGen(db4,sql_vip),viplist)

    ## 点击频率
    clickFrequency(dataGen(db4,sql_stand))
    maxClickFreq(dataGen(db4,sql_stand))

    ## 游戏时间段
    playInterval(dataGen(db4,sql_stand),interval_limit=3600)

    actionDistribution(dataGen(db4,sql_stand))

    with open("./diff_actions", 'r') as f:
        alist = {ele.strip() for ele in f.readlines()}
    alist = list(alist)

    targetActionTimes(dataGen(db4,sql_stand),alist)

    #ui停留时间比率
    # print(db4)
    ui_time_ratio(enter_action,exit_action,dataGen(db4,sql_level))
