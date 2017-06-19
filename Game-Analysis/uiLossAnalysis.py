import os
import sqlite3
import codecs
import time
import statistics
from datetime import datetime
from datetime import timedelta
import yaml
import collections
import pandas as pd
from operator import itemgetter
import numpy as np
from collections import Counter
from keyIndexAnalysis import IntervalGenerator

## 分离比较ACTION的逻辑
def compareAction(user_action, action_list):


    if user_action in action_list:
        return True
    else:
        return False



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

def dataGen(db_file, query):

    dbms = DatabaseManager(db_file)

    for row in dbms.query(query):
        yield row

def ui_continuing_click(in_action, data_iterator):
    num_enter = 0

    current_player = -1
    current_action = -1
    current_time = -1
    last_action = -1
    top_n = 3

    starting_time = time.time()
    counter = 0
    dist = collections.defaultdict(int)
    time_dist = collections.defaultdict(float)
    total_dist = collections.defaultdict(list)
    last_action_dist = collections.defaultdict(str)
    max_action_num = 0

    for row in data_iterator:

        counter += 1
        if counter % 100000 == 0:
            print("%s lines processed\n" % counter)
            # print(timestamp)

        action = row[1]
        player_id = row[0]
        timestamp = row[2]

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
            current_player = player_id

        # 如果上一个动作等于现在的动作，记录次数。否则，判断上一个连续动作的
        # 次数是否超过最大值
        if current_action == action:
            max_action_num += 1
        else:
            current_num = dist.get(current_action, 0)
            diff_time = timestamp - current_time
            current_time = timestamp
            if max_action_num > current_num:
                dist[current_action] = max_action_num
                time_dist[current_action] = diff_time
                last_action_dist[current_action] = last_action
            max_action_num = 0
            last_action = current_action

        current_action = action

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
    # print(pd_dist.columns)
    # print("max times "+str(max_occur_time))
    # print("max id "+ str(max_id))
    pd_dist.to_csv("连续点击动作-7.csv", encoding="utf_8",index=False, sep="\t")
    return total_dist


def ui_stay_click_distribution(in_action, out_action, data_iterator, loss_window = 20):

    num_enter = 0

    current_player = -1

    find_flag = False
    correct_exit = False

    # specific_act = "世界地图 / 世界地图 / 开始巡查"
    # max_occur_time = 0
    # max_id = ""
    # specific_act_occur_times = 0

    starting_time = time.time()
    counter = 0
    dist = collections.defaultdict(int)
    total_dist = collections.defaultdict(int)

    for row in data_iterator:

        counter += 1
        # if counter % 1000000 == 0:
        #     print("%s lines processed\n" % counter)
        #     print("total enter:" + str(num_enter))

        action = row[1]
        player_id = row[0]
        timestamp = row[2]

        # 如果更换用户，初始化所有值
        if player_id != current_player:
            find_flag = False
            if correct_exit:
                for key, value in dist.items():
                    total_dist[key] += value
                # if specific_act_occur_times > max_occur_time:
                #     max_occur_time = specific_act_occur_times
                #     max_id = current_player
            correct_exit = False
            # specific_act_occur_times = 0
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
                # print(action)
                # print(specific_act_occur_times)
                continue
            else:
                # if action == "世界地图 / 世界地图 / 开始巡查":
                #     specific_act_occur_times += 1
                dist[action] += 1

    # print(total_dist)
    # pd_dist = pd.Series(total_dist, name='OccurTimes')
    pd_dist = pd.DataFrame(list(total_dist.items()), columns=['Action', 'OccurTimes'])

    pd_dist.sort_values(by="OccurTimes", inplace=True, ascending=False)
    print(pd_dist.head(10))
    print(pd_dist.columns)
    # print("max times "+str(max_occur_time))
    # print("max id "+ str(max_id))

    return total_dist


# def ui_loss_rate_auto(in_action, data_iterator, loss_window = 20):


def ui_loss_rate(in_action, out_action, data_iterator, loss_window = 20):

    num_enter = 0
    num_of_correct_exit = 0
    num_of_greater_3 = 0
    num_of_stay_click = 0

    current_player = -1

    find_flag = False
    correct_exit = True
    window_len = 0
    ui_stay_click = 0

    starting_time = time.time()
    counter = 0

    for row in data_iterator:

        counter += 1
        if counter % 100000 == 0:
            print("%s lines processed\n" % counter)
            print("total enter:" + str(num_enter))
            print("correct exit :" + str(num_of_correct_exit))

        action = row[1]
        player_id = row[0]
        timestamp = row[2]

        # 如果更换用户，初始化所有值
        if player_id != current_player:
            current_player = player_id
            find_flag = False
            window_len = 0
            correct_exit = True
            if ui_stay_click > 3:
                num_of_greater_3 += 1
            num_of_stay_click += ui_stay_click
            ui_stay_click = -1

        # 如果被标记为，该用户所有剩余action都忽略
        if not correct_exit:
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
                num_of_correct_exit += 1
                correct_exit = False
                ui_stay_click -= 1
            else:
                window_len += 1
                if window_len >= loss_window:
                    correct_exit = False
            ui_stay_click += 1

    loss_rate = 1-(num_of_correct_exit/num_enter)
    average_stay_click = num_of_stay_click/num_enter
    print("loss rate is "+str(loss_rate))
    print("average stay click is  "+str(average_stay_click))
    print("num of users clicking bigger than 3  " + str(num_of_greater_3))
    end_time = time.time()
    print("script ran for %s secs" % ((end_time - starting_time)))

    return (loss_rate, average_stay_click, num_of_greater_3)



def ui_stay_time(in_action, out_action, data_iterator, loss_window = 20):

    time_of_normal_stay = 0
    time_of_total_stay = 0

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
    normal_exit_time = 0

    for row in data_iterator:

        counter += 1
        if counter % 100000 == 0:
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
            if enter_time != 0:
                if total_exit_time !=0:
                    time_of_total_stay += (total_exit_time - enter_time)
                    if normal_exit_time !=0:
                        if (normal_exit_time - enter_time) > 200:
                            print(current_player)
                        time_of_normal_stay += (normal_exit_time - enter_time)

                #如果为非正常退出，且窗口长度小于阈值，则取用户的最后一次时间戳为退出时间
                else:
                    time_of_total_stay += (timestamp - enter_time)

            current_player = player_id

            enter_time = 0
            total_exit_time = 0
            normal_exit_time = 0

        action = row[1]
        timestamp = row[2]

        # 如果被标记，该用户所有剩余action都忽略
        if not correct_exit:
            continue

        # 如果对某一个用户，第一次找到相应的进入UI动作，记录总人数加一并标记。记录进入时间
        if (compareAction(action, in_action) and (not find_flag)):
            find_flag = True
            enter_time = timestamp
            num_enter += 1
            continue

        # 如果已找到进入动作，判断此时动作是否为退出动作。如果是，正确退出加一，否则判断窗口长度是否大于
        # 标准值，如果是的话，此用户记录为非正常退出，并记录时间
        if find_flag:
            if compareAction(action, out_action):
                num_of_correct_exit += 1
                normal_exit_time = timestamp
                total_exit_time = timestamp
                correct_exit = False
            else:
                window_len += 1
                if window_len >= loss_window:
                    correct_exit = False
                    total_exit_time = timestamp


    # loss_rate = 1-(num_of_correct_exit/num_enter)
    average_normal_stay_time = time_of_normal_stay / num_of_correct_exit
    average_total_stay_time = time_of_total_stay / num_enter
    print("average normal stay time is "+str(average_normal_stay_time))
    # print("average total stay time is "+str(average_total_stay_time))
    end_time = time.time()
    print("正常进入退出："+str(num_of_correct_exit))
    print("script ran for %s secs" % ((end_time - starting_time)))

    return (average_normal_stay_time, average_total_stay_time)


def ui_enter_ratio(data_iterator, first_day_timestamp, n, in_action, loss_window = 20):

    num_of_days = timedelta(days=n)
    date = datetime.fromtimestamp(first_day_timestamp)
    print(date)

    range = date + num_of_days
    during_secs = range.second + 60 * range.minute + range.hour * 3600
    begin_date = range.timestamp() - during_secs
    end_date = range.timestamp() + (86400 - during_secs)
    print(begin_date)
    print(end_date)
    num_enter = 0
    num_ui_enter = 0
    current_player = -1

    find_flag = False
    new_user = True

    starting_time = time.time()
    counter = 0

    for row in data_iterator:
        counter += 1
        if counter % 100000 == 0:
            print("%s lines processed\n" % counter)

        action = row[1]
        player_id = row[0]
        timestamp = row[2]

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
        return -1
    else:
        return (num_ui_enter / num_enter)

def ui_click_times(in_action, data_iterator):

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



        # 如果对某一个用户，第一次找到相应的进入UI动作，记录总人数加一并标记。
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
    name = in_action.split("/")[-1] + ".xlsx"
    out_path = os.path.join(os.getcwd(), "ui",name)
    # writer = pd.ExcelWriter(out_path)
    pd_dist.loc[pd_dist['Times'] != 0,].to_excel(out_path, 'Sheet1',index=False, engine='xlsxwriter')

    return pd_dist

def timeGen(db, sqlstr, begin_date, during_days, feature_name):

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
        dayOneDF = ui_click_times(db, sqlstr, begin_date, end_date, changeF=-1)
        if i == 0:
            res.loc[:, 'Player'] = dayOneDF['Player']
        name = str(i) + "Day"
        print(name)
        #     res = res.assign({name:dayOneDF['CumulativeChange']})
        # res.loc[:, name] = dayOneDF['CumulativeChange']
        i += 1

    times = pd.DataFrame()
    times['Player'] = res['Player']


    name = feature_name + "_" + str(-1) + ".xlsx"
    times.to_excel(name, 'Sheet1', index=False, engine='xlsxwriter')
    # times.to_csv("体力损耗分布.csv", index=False)
    return times


if __name__ == "__main__":

    db_file = "./world_six.db"
    # db = DatabaseManager(db_file)
    # query_sql = "SELECT player_id,action,happen_time FROM maidian ORDER BY player_id,happen_time ASC"

    db = "/home/maoan/maidianAnalysis/level2-uianalysis/world_seven.db"
    db2 = "/home/maoan/maidianAnalysis/xiamen/xiamen_1.db"
    db3= "/home/maoan/maidianAnalysis/xiamen/1308310007.db"
    db4 = "/home/maoan/maidianAnalysis/xiamen/xiamen_1b.db"

    sqlstr = "SELECT yonghu_id, timestamp, action, duiwu_level FROM maidian ORDER BY yonghu_id,timestamp ASC;"

    # with open("alist.txt", 'r', encoding="utf_8") as f:
    #     for line in f:
    #         print(line)
    #         ui_click_times(line, dataGen(db2, sqlstr))
    sql_diff_user = "SELECT yonghu_id, timestamp, action, duiwu_level FROM maidian WHERE num_days_played = 2 ORDER BY yonghu_id,timestamp ASC;"
    ui_click_times("UIRoot2D/NormalPanel/MainWin/Go_StarReward/Btn_Reward",dataGen(db4, sql_diff_user))
    # ui_stay_click_distribution(enter_action, exit_action, dataGen(db, query_sql))
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
        #         enter_rate = ui_enter_ratio(dataGen(db,query_sql), new_date.timestamp(), 0, enter_action)
        #         # loss_rate = 0.343434
        #         # average_normal_stay_time = 3434.5555
        #         # average_total_stay_time = 34.444
        #         # enter_rate = 0.3434344
        #         print(enter_rate)
        #         csvfile.write("%s,%s,%s,%s,%s,%s,%s,%s\n" % (
        #         loss_rate, average_stay_click, num_of_greater_3, average_normal_stay_time, average_total_stay_time,
        #         enter_rate, enter_action, out_action))
    # ui_loss_rate(enter_action, enter_action, dataGen(db, query_sql))
    # ui_continuing_click(enter_action, dataGen(db, query_sql))

    # ui_click_times()