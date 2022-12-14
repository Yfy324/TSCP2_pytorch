import numpy as np
from sklearn.model_selection import train_test_split


def load_bearing_ds(path, window, name='xjtu1', mode='train'):
    if mode == 'train':
        X, lbl = extract_windows_1(path, window, name)
        print('train samples : ', lbl.shape)
        return X, lbl

    # X_train, X_test, y_train, y_test = train_test_split(X, lbl,
    #                                                     train_size=0.8,
    #                                                     shuffle=True,
    #                                                     stratify=lbl,
    #                                                     random_state=13130132)
    # if mode == "train":
    #     print('train samples : ', y_train.shape)
    #     return X_train, y_train

    else:
        X, lbl = extract_windows(path, name)
        print(f'Test shape {X.shape}')
        return X, lbl


def extract_windows(path, name):
    load_path = path + name + '_2560.npy'
    dataset = np.load(load_path, allow_pickle=True).item()
    windows = dataset['Bearing1_1'].astype(float)
    lbl = np.zeros(windows.shape[0], dtype='int')
    print(f"number of samples : {len(windows)}"
          # f" number of samples with change point : {num_cp}"
          )
    return windows, lbl

def extract_windows_1(path, window_size, name):
    windows = []

    load_path = path + name + '.npy'
    dataset = np.load(load_path, allow_pickle=True).item()
    ts = dataset['Bearing1_1'].astype(float)
    ts = ts[:, :30720, :]

    for i in range(ts.shape[0]):
        for j in range(0, ts.shape[1] - window_size, 128):
            windows.append(ts[i, j:j+window_size, :])

    windows = np.array(windows)
    print(f"number of samples : {len(windows)}")
    lbl = np.zeros(windows.shape[0], dtype='int')

    return windows, lbl

def extract_windows_2(path, window_size, name):
    windows = np.array([])

    load_path = path + name + '.npy'
    dataset = np.load(load_path, allow_pickle=True).item()
    ts = dataset['Bearing1_1'].astype(float)
    length = ts.shape[0]
    r = length % 5
    # ts = ts.reshape(-1)

    for l in range(0, length-r, 5):
        data = np.concatenate(ts[l:l+5, :, :], axis=0)
        data = data[np.newaxis, :, :]
        data = np.concatenate(np.split(data, 64, axis=1), axis=0)
        if windows.size == 0:
            windows = data
        else:
            windows = np.append(windows, data, axis=0)

    data = np.concatenate(ts[-r:, :, :], axis=0)
    r1 = data.shape[0] % window_size
    data = data[np.newaxis, :-r1, :]
    data = np.concatenate(np.split(data, data.shape[1]//window_size, axis=1), axis=0)
    windows = np.append(windows, data, axis=0)

    windows = np.array(windows)
    lbl = np.zeros(windows.shape[0], dtype='int')
    print(f"number of samples : {len(windows)}"
          # f" number of samples with change point : {num_cp}"
          )
    return windows, lbl
