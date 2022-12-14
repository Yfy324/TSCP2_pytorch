# -*- coding: utf-8 -*-

import os
import operator
import random
import scipy.io as sio
import pickle as pickle
import numpy as np
import pandas as pd
from collections import defaultdict


def make_phm_dataset():
    RUL_dict = {'Bearing1_1': 0, 'Bearing1_2': 0,
                'Bearing2_1': 0, 'Bearing2_2': 0,
                'Bearing3_1': 0, 'Bearing3_2': 0,
                'Bearing1_3': 573, 'Bearing1_4': 33.9, 'Bearing1_5': 161, 'Bearing1_6': 146, 'Bearing1_7': 757,
                'Bearing2_3': 753, 'Bearing2_4': 139, 'Bearing2_5': 309, 'Bearing2_6': 129, 'Bearing2_7': 58,
                'Bearing3_3': 82}
    phm_dataset = defaultdict(list)
    source_path = r'/data/yfy/FD-data/PHM2012/'
    for path_1 in ['Learning_set/', 'Full_Test_Set/']:
        bearings_names = os.listdir(source_path + path_1)
        bearings_names.sort()
        for bearings_name in bearings_names:
            file_names = os.listdir(source_path + path_1 + bearings_name + '/')
            file_names.sort()
            bearing_data = np.array([])
            for file_name in file_names:
                if 'acc' in file_name:
                    df = pd.read_csv(source_path + path_1 + bearings_name + '/' \
                                     + file_name, header=None)
                    data = np.array(df.loc[:, 4:6])
                    data = data[np.newaxis, :, :]
                    if bearing_data.size == 0:
                        bearing_data = data
                    else:
                        bearing_data = np.append(bearing_data, data, axis=0)

            phm_dataset[bearings_name] = bearing_data
            print(bearings_name, 'has been appended.')

    np.save('phm_dict.npy', phm_dataset)


def make_xjtu_dataset():
    '''
    csv_timestep = 32768  # 2^14 -> 32768 每个文件记录的data points数
    '''
    source_path = r"/data/yfy/FD-data/XJTU/"
    RUL_dict = {
        'Bearing1_1': [123, 2], 'Bearing1_2': [161, 2], 'Bearing1_3': [158, 2], 'Bearing1_4': [122, 3], 'Bearing1_5': [52, [1, 2]],
        'Bearing2_1': [491, 1], 'Bearing2_2': [161, 2], 'Bearing2_3': [533, 3], 'Bearing2_4': [42, 2], 'Bearing2_5': [339, 2],
        'Bearing3_1': [2538, 2], 'Bearing3_2': [2496, [1, 2, 3, 4]], 'Bearing3_3': [371, 1],  'Bearing3_4': [1515, 1], 'Bearing3_5': [114, 2]
                }  # [RUL, fault_label//(inner_race -> 1, outer_race -> 2, cage -> 3, ball -> 4 标签)]
    xjtu_dataset = defaultdict(list)
    # for path_1 in ["35Hz12kN/", "37.5Hz11kN/", "40Hz10kN/"]:
    for path_1 in ["37.5Hz11kN/"]:
        bearings_names = os.listdir(source_path + path_1)
        bearings_names.sort()
        for bearings_name in bearings_names:
            file_names = os.listdir(source_path + path_1 + bearings_name + '/')
            file_names.sort(key=lambda x1: int(x1.split('.')[0]))
            # dataset2.sort(key=lambda x1: int(x1.split('.')[0].split('_')[2]))
            bearing_data = np.array([])
            for file_name in file_names:
                # if RUL_dict[bearings_name][0] < 300:
                df = pd.read_csv(source_path + path_1 + bearings_name + '/' \
                                 + file_name, header=None).drop(0)
                data = np.array(df)
                data = data[np.newaxis, :30720, :]
                data = np.concatenate(np.split(data, 12, axis=1), axis=0)

                # elif RUL_dict[bearings_name][0] < 1000:
                #     df = pd.read_csv(source_path + path_1 + bearings_name + '/' \
                #                      + file_name, header=None).drop(0)
                #     data = np.array(df.iloc[::4, :])
                #     data = data[np.newaxis, :, :]
                #     data = np.concatenate(np.split(data, 4, axis=1), axis=0)
                # else:
                # df = pd.read_csv(source_path + path_1 + bearings_name + '/' \
                #                  + file_name, header=None).drop(0)
                # data = np.array(df.iloc[::16, :])
                # data = data[np.newaxis, :, :]
                    # data = np.concatenate(np.split(data, 16, axis=1), axis=0)
                if bearing_data.size == 0:
                    bearing_data = data
                else:
                    bearing_data = np.append(bearing_data, data, axis=0)

            xjtu_dataset[bearings_name] = bearing_data
            print(bearings_name, 'has been appended.')

    np.save('/data/yfy/FD-data/RUL/xjtu2_2560.npy', xjtu_dataset)


if __name__ == '__main__':
    # make_phm_dataset()
    # dataset = np.load('phm_dict.npy', allow_pickle=True).item()
    # dataset._save_info()

    make_xjtu_dataset()
    dataset = np.load('/data/yfy/FD-data/RUL/xjtu2_2560.npy', allow_pickle=True).item()
    # dataset._save_info()

    # make_paderborn_dataset()
    # make_ims_dataset()
    # dataset = DataSet.load_dataset('ims_data')
    print('1')
