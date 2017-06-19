import pandas as pd
import numpy as np
import sqlite3
import collections
import time
from datetime import datetime
from datetime import timedelta
from keyIndexAnalysis import IntervalGenerator
from keyIndexAnalysis import DatabaseManager
# from keyIndexAnalysis import dataGen

def dataGen(db, query):
    dbms = DatabaseManager(db)
    for row in dbms.query(query):
        yield row

def keyIndexChangeByLevel(sqlstr, db):

    data_iterator = dataGen(db, sqlstr)

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
    # action_contribution_dict = collections.defaultdict(list)

    counter = 0
    # dist = collections.defaultdict(int)

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

    pd_dist.drop("KeyFactor", axis=1).to_excel("钻石随等级变化情况-厦门.xlsx", 'Sheet1', index=False, engine='xlsxwriter')
    return pd_dist


def keyIndexChangeByTime(sqlstr, db, interval_in_secs):
    data_iterator = dataGen(db, sqlstr)

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

    # action_contribution_dict = collections.defaultdict(list)

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

            ##忽略最后不满一小时的数据
            # player_list.append(last_key_value)

            # if current_player != -1:

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

        # if major_action == None or major_action == "启动":
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

    pd_dist.drop("KeyFactor", axis=1).to_excel("钻石随时间变化情况-厦门.xlsx", 'Sheet1', index=False, engine='xlsxwriter')

    return pd_dist


if __name__ == "__main__":
    db = "/home/maoan/maidianAnalysis/level2-uianalysis/world_seven.db"
    db2 = "/home/maoan/maidianAnalysis/xiamen/xiamen_1.db"
    db3 = "/home/maoan/maidianAnalysis/xiamen/1308310007.db"
    db4 = "/home/maoan/maidianAnalysis/xiamen/xiamen_1b.db"
    sql_time = "SELECT yonghu_id, timestamp, jinbi, duiwu_zhanli FROM maidian WHERE num_days_played > 1 ORDER BY yonghu_id,duiwu_level ASC;"
    sql_level = "SELECT yonghu_id, timestamp, duiwu_level, jinbi, duiwu_zhanli FROM maidian WHERE num_days_played > 1 ORDER BY yonghu_id,duiwu_level ASC;"

    sqls_kuangbaozhiyi_time = "SELECT user_id, riqi, zuanshi, zhanli FROM maidian ORDER BY user_id ASC;"
    sqls_kuangbaozhiyi_level = "SELECT user_id, riqi, dengji, zuanshi, zhanli FROM maidian ORDER BY user_id,dengji ASC;"

    keyIndexChangeByTime(sqlstr=sql_time, db=db4, interval_in_secs=600)
    keyIndexChangeByLevel(sqlstr=sql_level, db=db4)