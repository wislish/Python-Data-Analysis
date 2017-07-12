import os
import sqlite3
import codecs
import time
import statistics
import yaml
import collections
import pandas as pd
from operator import itemgetter
import numpy as np
from collections import Counter

from tools import *

## 输出路径设置,所有输出文件都在此父文件夹下。
parent_file = "ui分析结果"

### 研究每个用户的连续点击动作，并选出前几个动作进行
###　综合分析
def ui_continuing_click(data_iterator):
    """
    研究每个用户的连续点击动作，并选出前几个动作进行
    综合分析
    :param data_iterator: 数据生成器，可循环.[player_id,timestamp,action]
    :return: 统计结果,DataFrame格式.
                导出文件：连续动作分析_timestamp_xlsx.
    """

    num_enter = 0

    current_player = -1
    current_action = -1
    current_time = -1
    last_time = -1
    last_action = -1

    ##选出每个玩家的前三个动作
    top_n = 3
    time_limit = 600

    counter = 0
    dist = collections.defaultdict(int)
    time_dist = collections.defaultdict(float)
    total_dist = collections.defaultdict(list)
    last_action_dist = collections.defaultdict(str)
    max_action_num = 0

    for row in data_iterator:

        counter += 1
        if counter % 1000000 == 0:
            print("%s lines processed\n" % counter)
            # print(timestamp)

        action = row[2]
        player_id = row[0]
        timestamp = row[1]

        # 如果更换用户，初始化所有值
        if player_id != current_player:

            i = top_n
            # 对某一个用户，选出连续次数前top_n的action,并记录次数和时间
            s = [(k, dist[k]) for k in sorted(dist, key=dist.get, reverse=True)]
            for key, value in s:
                if i > 0:
                    value_pair = (value, time_dist.get(key), last_action_dist.get(key))
                    total_dist[key].append(value_pair)
                else:
                    break
                i -= 1

            dist.clear()
            time_dist.clear()
            max_action_num = 0
            current_action = -1
            current_time = -1
            last_time = -1
            current_player = player_id

        # 如果上一个动作等于现在的动作，记录次数。否则，判断上一个连续动作的
        # 次数是否超过最大值
        if current_action == action:
            max_action_num += 1
            if max_action_num ==1:
                current_time = timestamp
        else:
            ##　若不连续，得到目前动作的最大连续次数，
            ##　并更新最大数值，记录连续操作时间以及
            ##　连续动作开始前的一个动作。
            current_num = dist.get(current_action, 0)
            diff_time = last_time - current_time
            # current_time = timestamp

            ## 连续动作次数必须大于现有值，且间隔时间小于规定阈值（１０分钟）
            if max_action_num > current_num and diff_time < time_limit:
                dist[current_action] = max_action_num
                time_dist[current_action] = diff_time
                last_action_dist[current_action] = last_action
            max_action_num = 0
            last_action = current_action

        current_action = action
        last_time = timestamp

    ## 最后一个人
    i = top_n
    s = [(k, dist[k]) for k in sorted(dist, key=dist.get, reverse=True)]
    for key, value in s:
        if i > 0:
            value_pair = (value, time_dist.get(key), last_action_dist.get(key))
            total_dist[key].append(value_pair)
        else:
            break
        i -= 1
    # print(total_dist)
    # pd_dist = pd.Series(total_dist, name='OccurTimes')
    pd_dist = pd.DataFrame(list(total_dist.items()), columns=['Action', 'TopN'])

    # pd_dist.sort_values(by="OccurTimes", inplace=True, ascending=False)
    print(pd_dist.head(10))
    pd_dist['average_click'] = pd_dist['TopN'].apply(lambda l : np.mean([pair[0] for pair in l]))
    pd_dist['average_duration'] = pd_dist['TopN'].apply(lambda l : np.mean([pair[1] for pair in l]))
    pd_dist['median_click'] = pd_dist['TopN'].apply(lambda l: np.median([pair[0] for pair in l]))
    pd_dist['median_duration'] = pd_dist['TopN'].apply(lambda l: np.median([pair[1] for pair in l]))
    pd_dist['num_of_user'] = pd_dist['TopN'].apply(lambda l : len(l))
    pd_dist['last_action'] = pd_dist['TopN'].apply(lambda l : Counter([pair[2] for pair in l]).most_common(1)[0][0])
    pd_dist['TopN'] = pd_dist['TopN'].apply(lambda l : [(pair[0], pair[1]) for pair in l])

    pd_dist.drop(["TopN"], inplace=True,axis=1)
    pd_dist.sort_values(by=["num_of_user","average_click"], inplace=True, ascending=False)
    print(pd_dist.head(10))

    writeTo(parent_file,"连续动作分析",pd_dist)

    return total_dist

def ui_stay_click_distribution(in_action, out_action, data_iterator):
    """
    在某个ui期间，统计其中action的分布情况。
    :param in_action: ui入口
    :param out_action: ui出口
    :param data_iterator: 数据生成器，可循环。[player_id,timestamp,action]
    :return: 统计结果, DataFrame格式。
                导出文件: ui停留动作_timestamp.xlsx
    """

    num_enter = 0

    current_player = -1

    find_flag = False
    correct_exit = False

    # specific_act = "世界地图 / 世界地图 / 开始巡查"
    # max_occur_time = 0
    # max_id = ""
    # specific_act_occur_times = 0

    counter = 0
    dist = collections.defaultdict(int)
    total_dist = collections.defaultdict(list)

    for row in data_iterator:

        counter += 1
        if counter % 1000000 == 0:
            print("%s lines processed\n" % counter)

        action = row[2]
        player_id = row[0]
        timestamp = row[1]

        # 如果更换用户，初始化所有值
        if player_id != current_player:
            find_flag = False

            ## 如果是正常退出,记录其点击次数，并加和
            if correct_exit:
                for key, value in dist.items():
                    total_dist[key].append(value)

            correct_exit = False

            dist.clear()
            current_player = player_id

        # 如果被标记，该用户所有剩余action都忽略
        if correct_exit:
            continue

        # 如果对某一个用户，第一次找到相应的进入UI动作，记录总人数加一并标记。
        if (compareAction(action, in_action) and (not find_flag)):
            num_enter += 1
            find_flag = True
            continue

        # 如果已找到进入动作，判断此时动作是否为退出动作。如果是，正确退出加一，否则判断窗口长度是否大于
        # 标准值，如果是的话，此用户记录为非正常退出。
        if find_flag:
            if compareAction(action, out_action):
                correct_exit = True

                continue
            else:
                dist[action] += 1

    pd_dist = pd.DataFrame(list(total_dist.items()), columns=['Action', 'Times'])

    pd_dist['average_click'] = pd_dist['Times'].apply(lambda l : np.mean(l))
    pd_dist['median_click'] = pd_dist['Times'].apply(lambda l: np.median(l))
    pd_dist['num_of_user'] = pd_dist['Times'].apply(lambda l : len(l))

    pd_dist.sort_values(by=["num_of_user","median_click"], inplace=True, ascending=False)

    name = "ui停留动作"
    writeTo(parent_file,name,pd_dist)

    return pd_dist


def ui_loss_rate(in_action, out_action, data_iterator, loss_window = 20,stay_limit = 1500):
    """
    某个ui的流失率，若一个玩家只有进入动作，而无退出动作，
    记为流失玩家。
    :param in_action: ＵＩ进入动作
    :param out_action: ＵＩ退出动作
    :param data_iterator: 数据生成器，可循环.[player_id,timestamp,action]
    :param loss_window: 丢失窗口，如果进入ui后，action数量大于这个值，则判定为流失
    :param stay_limit: 如果某一个时间跟进入ui时间的差大于这个阈值，则判定为流失
    :return: loss_rate：流失率
            average_stay_click：停留期间，平均点击次数
            num_of_greater_3：停留期间，点击次数大于三的人数
    """

    num_enter = 0
    num_of_correct_exit = 0
    num_of_greater_3 = 0
    num_of_stay_click = 0

    current_player = -1

    find_flag = False
    correct_exit = True
    window_len = 0
    ui_stay_click = -1

    enter_time = 0
    total_exit_time = 0

    total_dist = collections.defaultdict(int)


    starting_time = time.time()
    counter = 0

    for row in data_iterator:

        counter += 1
        if counter % 100000 == 0:
            print("%s lines processed\n" % counter)
            print("total enter:" + str(num_enter))
            print("correct exit :" + str(num_of_correct_exit))

        action = row[2]
        player_id = row[0]
        timestamp = row[1]

        # 如果更换用户，初始化所有值
        if player_id != current_player:
            current_player = player_id
            find_flag = False
            window_len = 0
            correct_exit = True

            if ui_stay_click > 3:
                num_of_greater_3 += 1
            ## 记录了每个人的停留点击次数
            if ui_stay_click != -1:
                total_dist[current_player] = ui_stay_click

            num_of_stay_click += ui_stay_click
            ui_stay_click = -1

            enter_time = 0
            total_exit_time = 0

        # 如果被标记为，该用户所有剩余action都忽略
        if not correct_exit:
            continue


        # 如果对某一个用户，第一次找到相应的进入UI动作，记录总人数加一并标记。
        if (compareAction(action, in_action) and (not find_flag)):
            num_enter += 1
            find_flag = True
            enter_time = timestamp
            total_exit_time = timestamp
            continue

        # 如果已找到进入动作，判断此时动作是否为退出动作。如果是，正确退出加一，否则判断窗口长度是否大于
        # 标准值，如果是的话，此用户记录为非正常退出。
# if find_flag:
#     if compareAction(action, out_action):
#         num_of_correct_exit += 1
#         correct_exit = False
#         ui_stay_click -= 1
#     else:
#         window_len += 1
#         if window_len >= loss_window:
#             correct_exit = False
#     ui_stay_click += 1

        if find_flag:
            if compareAction(action, out_action):
                num_of_correct_exit += 1
                total_exit_time = timestamp
                correct_exit = False
                ui_stay_click -= 1
            else:
                window_len += 1
                if window_len >= loss_window and (timestamp - enter_time) < stay_limit:
                    correct_exit = False
                    total_exit_time = timestamp
                    continue
                if (timestamp - enter_time) > stay_limit:
                    correct_exit = False
                    total_exit_time = enter_time
                    continue
            ui_stay_click += 1

    if ui_stay_click > 3:
        num_of_greater_3 += 1
    ## 记录了每个人的停留点击次数
    if ui_stay_click != -1:
        total_dist[current_player] = ui_stay_click

    num_of_stay_click += ui_stay_click

    loss_rate = 1-(num_of_correct_exit/num_enter)
    average_stay_click = num_of_stay_click/num_enter
    print("loss rate is "+str(loss_rate))
    print("average stay click is  "+str(average_stay_click))
    print("num of users clicking bigger than 3  " + str(num_of_greater_3))
    end_time = time.time()
    print("script ran for %s secs" % ((end_time - starting_time)))

    return (loss_rate, average_stay_click, num_of_greater_3)


def ui_stay_time(in_action, out_action, data_iterator, loss_window=20):
    """
    在某个ui上的停留时间
    :param in_action: ui进入动作
    :param out_action: ui退出动作
    :param data_iterator: 数据生成器.[player_id,timestamp,action]
    :param loss_window: 丢失窗口，如果进入ui后，action数量大于这个值，则判定为流失
    :return: 统计结果, DataFrame 格式.
                导出文件：ui停留时间_timestamp.xlsx
    """

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
        # 如果更换用户，初始化所有值，并且计算上一个用户的所有以及正确的停留的时间。
        if player_id != current_player:
            find_flag = False
            window_len = 0
            correct_exit = True
            # 如果没有进入时间，则忽略计算。否则，判断下是否有整体退出时间。
            #             if enter_time != 0:
            #                 stay_time = (total_exit_time - enter_time)
            #                 duringDict[current_player] = stay_time
            #             else:
            #                 duringDict[current_player] = 0
            if current_player != -1:
                stay_time = (total_exit_time - enter_time)
                duringDict[current_player] = stay_time

            current_player = player_id

            enter_time = 0
            total_exit_time = 0
            stay_time = 0

        action = row[2]
        timestamp = row[1]

        #         if major_action == None or major_action == "启动":
        #             continue

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
                if window_len >= loss_window and (timestamp - enter_time) < stay_limit:
                    correct_exit = False
                    total_exit_time = timestamp
                if (timestamp - enter_time) > stay_limit:
                    correct_exit = False
                    total_exit_time = enter_time

    stay_time = (total_exit_time - enter_time)
    duringDict[current_player] = stay_time

    pd_dist = pd.DataFrame(list(duringDict.items()), columns=['Player', 'TimeOnUI'])

    writeTo(parent_file,"ui停留时间",pd_dist)

    return pd_dist

def ui_enter_ratio(data_iterator, in_action, begin_date, end_date):
    """
    针对某一个自然天，算出某个ＵＩ的进入率，即有多少人
    进入了这个ui
    :param data_iterator: 数据生成器，可循环.[player_id,timestamp,action]
    :param in_action: ui进入动作
    :param begin_date: 某一天的开始时间戳
    :param end_date: 某一天的结束时间戳
    :return: ratio，ui的进入率
    """

    num_enter = 0
    num_ui_enter = 0
    current_player = -1

    find_flag = False
    new_user = True
    counter = 0

    for row in data_iterator:
        counter += 1
        if counter % 1000000 == 0:
            print("%s lines processed\n" % counter)

        action = row[2]
        player_id = row[0]
        timestamp = row[1]

        # 如果更换用户，初始化所有值
        if player_id != current_player:
            current_player = player_id
            find_flag = False
            new_user = True

        # 如果此条记录在规定的自然天范围内，且是该用户在这一天第一次进入游戏，游戏总人数加1并标记.
        # 如果不在规定的自然天内，接着处理下一条
        if timestamp > begin_date and timestamp < end_date:
            if new_user:
                num_enter += 1
                new_user = False
        else:
            continue

        # 如果对某一个用户，第一次找到相应的进入UI动作，ui进入人数加一并标记。
        if (compareAction(action, in_action) and (not find_flag)):
            # if timestamp > begin_date and timestamp < end_date:
            num_ui_enter += 1
            find_flag = True


    if num_enter == 0:
        ratio = -1
    else:
        ratio =  (num_ui_enter / num_enter)

    print("{0} 进入率为 {1}".format(enter_action,ratio))

    return ratio


def ui_click_times(in_action, data_iterator):
    """
    同一动作的点击次数，和第一次点击时的等级。
    
    :param in_action: ui进入动作
    :param data_iterator: 数据生成器，可循环。[player_id,timestamp,action,level]
    :return: 统计结果，DataFrame格式。
                导出文件（次数不为０的用户）：action的最后一个字段_点击次数_timestamp.xlsx
    """

    current_player = -1

    ui_click = 0
    first_click_level = 0
    starting_time = time.time()
    counter = 0
    player_dict = collections.defaultdict(tuple)


    for row in data_iterator:

        counter += 1
        if counter % 1000000 == 0:
            print("%s lines processed\n" % counter)


        level = row[3]
        action = row[2]
        player_id = row[0]
        timestamp = row[1]

        # 如果更换用户，初始化所有值
        if player_id != current_player:

            if current_player != -1:
                val = (ui_click, first_click_level)
                player_dict[current_player] = val

            ui_click = 0
            first_click_level = 0
            find_flag = False
            current_player = player_id


        # 如果对某一个用户，找到相应的进入UI动作，记录总次数。
        if (compareAction(action, in_action)):
            ui_click += 1
            if not find_flag:
                find_flag = True
                first_click_level = level

    val = (ui_click, first_click_level)
    player_dict[current_player] = val

    end_time = time.time()
    print("script ran for %s secs" % ((end_time - starting_time)))

    pd_dist = pd.DataFrame(list(player_dict.items()), columns=['Player', 'Info'])
    pd_dist.loc[:, 'Times'] = pd_dist.Info.apply(lambda t: t[0])
    pd_dist.loc[:, 'Level'] = pd_dist.Info.apply(lambda t: t[1])
    # name = "./ui/" + action.split("/")[-1] + ".csv"
    name = in_action.split("/")[-1] + "_点击次数"

    writeTo(parent_file,name,pd_dist.loc[pd_dist['Times'] != 0,])

    return pd_dist


## test the script
if __name__ == "__main__":

    ### 以下测试,均采用厦门游戏的数据.

    enter_action = "UIRoot2D/ModalPanel/TeamEditWin/Btn_Start"
    exit_action = "UIRoot2D/NormalPanel/BattleSettlementWin/Img_Bg/Go_Btn/Btn_Quit"

    sqlstr = "SELECT yonghu_id, timestamp, action, duiwu_level FROM maidian ORDER BY yonghu_id,timestamp ASC;"
    sql_diff_user = "SELECT yonghu_id, timestamp, action, duiwu_level FROM maidian WHERE num_days_played = 2 ORDER BY yonghu_id,timestamp ASC;"

    ui_continuing_click(data_iterator=dataGen(db4,sqlstr))

    ui_stay_click_distribution(enter_action, exit_action, dataGen(db4, sqlstr))

    ui_loss_rate(enter_action, enter_action, dataGen(db4, sqlstr))

    ui_stay_time(enter_action, exit_action, dataGen(db4, sqlstr))

    ## 选择某一天，进行进入率的计算。
    begin_date = datetime.strptime("2017-06-01", "%Y-%m-%d")
    end_date = datetime.strptime("2017-06-03", "%Y-%m-%d")
    enter_rate = ui_enter_ratio(dataGen(db4, sqlstr), enter_action,begin_date.timestamp(),end_date.timestamp())

    ##对于有开始跟结束日期的方法，也可对多天进行统计，利用tools文件里的repeatByInterval
    #repeatByInterval()(ui_enter_ratio())

    ui_click_times(enter_action,dataGen(db4,sqlstr))

    # with open("alist.txt", 'r', encoding="utf_8") as f:
    #     for line in f:
    #         print(line)
    #         ui_click_times(line, dataGen(db2, sqlstr))
    # ui_click_times("UIRoot2D/NormalPanel/MainWin/Go_StarReward/Btn_Reward",dataGen(db4, sql_diff_user))
    # with codecs.open("world_seven_uianalysis.csv", 'a+', "utf_8_sig") as csvfile:
    #     # csvfile.write("loss_rate,average_normal_stay_time,average_total_stay_time,ui_enter_rate\n")
    #     #
    #     enter_action = "世界地图 / 城池菜单 / 点击城池"
    #     out_action = "世界地图 / 城池菜单 / 【菜单】点击前往"

        # csvfile.write("ui_loss_rate,average_stay_click,num_of_greater_3,ui_normal_stay_time,ui_total_stay_time,ui_enter_rate,enter_action,out_action\n")
        # with codecs.open("actionlist.txt", 'r', "utf_8") as f:
        #     uidict = yaml.load(f)
        #     for key in uidict.keys():
        #         enter_action = uidict.get(key).get("enter")
        #         out_action = uidict.get(key).get("exit")
        #         loss_rate, average_stay_click, num_of_greater_3 = ui_loss_rate(enter_action, out_action,
        #                                                                        dataGen(db,query_sql))
        #         average_normal_stay_time, average_total_stay_time = ui_stay_time(enter_action, out_action,
        #                                                                          dataGen(db,query_sql))
        #         new_date = datetime.strptime("2017-05-17", "%Y-%m-%d")
        #         # loss_rate = 0.343434
        #         # average_normal_stay_time = 3434.5555
        #         # average_total_stay_time = 34.444
        #         # enter_rate = 0.3434344
        #         print(enter_rate)
        #         csvfile.write("%s,%s,%s,%s,%s,%s,%s,%s\n" % (
        #         loss_rate, average_stay_click, num_of_greater_3, average_normal_stay_time, average_total_stay_time,
        #         enter_rate, enter_action, out_action))
    # ui_continuing_click(enter_action, dataGen(db, query_sql))

    # ui_click_times()


