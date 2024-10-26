import logging
import time
import pickle as pk
import numpy as np
import pandas as pd

class Data:
    def __init__(self, data, is_split=False):
        self.srcs = data['src'].values
        self.dsts = data['dst'].values
        self.times = data['abs_time'].values
        self.trans_cascades = data['cas'].values
        self.pub_times = data['pub_time'].values
        self.labels = data['label'].values
        self.length = len(self.srcs)
        self.is_split = is_split
        if is_split:
            self.types = data['type'].values

    def loader(self, batch):
        for i in range(0, len(self.srcs), batch):
            right = min(i + batch, self.length)
            if self.is_split:
                yield (self.srcs[i:right], self.dsts[i:right], self.trans_cascades[i:right],
                       self.times[i:right], self.pub_times[i:right], self.types[i:right]), self.labels[i:right]
            else:
                yield (self.srcs[i:right], self.dsts[i:right], self.trans_cascades[i:right],
                       self.times[i:right], self.pub_times[i:right]), self.labels[i:right]


def get_label(x: pd.DataFrame, observe_time, label):
    # time是传播时间，oberve_time是指定的观察窗口
    id = np.searchsorted(x['time'], observe_time, side='left')
    # 因为本身就是按照cas分组传进来的dataframe，所以casid是唯一的
    casid = x['cas'].values[0]
    # 如果casid在label中，且id大于等于10，那么将label[casid] - id赋值给x['label'].iloc[length]
    if casid in label and id >= 10:
        length = min(id, 100) - 1
        # 相当于在观察时间之后有多少个级联？x['label'][11] = 96-12 = 84（返回前12条，并且记录未来发生了84个级联）
        x['label'].iloc[length] = label[casid] - id
        # 返回的最后一行的label值是在oberve_time之后有多少个级联（训练数据最少10条，最多100条，并且保留在oberve_time之后又发生多少级联）
        return [x.iloc[:length + 1, :]]
    else:
        return []


# pubtime是最早发布的时间，time是发布后传播的时间，time_unit是时间单位，min_time是所有pub_time中最早的时间，abs_time是pub_time + time
def data_transformation(dataset, data, time_unit, min_time, param):
    if dataset == 'aps':
        data['pub_time'] = (pd.to_datetime(data['pub_time']) - pd.to_datetime(min_time)).apply(lambda x: x.days)
    else:
        data['pub_time'] -= min_time
    data['abs_time'] = (data['pub_time'] + data['time']) / time_unit
    data['pub_time'] /= time_unit
    data['time'] /= time_unit
    data.sort_values(by=['abs_time', 'id'], inplace=True, ignore_index=True)
    # 所有的users
    users = list(set(data['src']) | set(data['dst']))
    ids = list(range(len(users)))
    user2id, id2user = dict(zip(users, ids)), dict(zip(ids, users))
    cases = list(set(data['cas']))
    ids = list(range(len(cases)))
    cas2id, id2cas = dict(zip(cases, ids)), dict(zip(ids, cases))
    data['src'] = data['src'].apply(lambda x: user2id[x])
    data['dst'] = data['dst'].apply(lambda x: user2id[x])
    data['cas'] = data['cas'].apply(lambda x: cas2id[x])
    param['node_num'] = {'user': max(max(data['src']), max(data['dst'])) + 1, 'cas': max(data['cas']) + 1}
    param['max_global_time'] = max(data['abs_time'])
    pk.dump({'user2id': user2id, 'id2user': id2user, 'cas2id': cas2id, 'id2cas': id2cas},
            open(f'data/{dataset}_idmap.pkl', 'wb'))


def get_split_data(dataset, observe_time, predict_time, time_unit, all_data, min_time, metadata, log, param, sub_test=False):
    def data_split(legal_cascades, train_portion=0.7, val_portion=0.15):
        """
        set cas type, 1 for train cas, 2 for val cas, 3 for test cas , and 0 for other cas that will be dropped
        """
        m_metadata = metadata[metadata['casid'].isin(set(legal_cascades))]
        all_idx, type_map = {}, {}
        if dataset == 'twitter':
            dt = pd.to_datetime(m_metadata['pub_time'], unit='s', utc=True).dt.tz_convert('Asia/Shanghai')
            # 在4.10之前的值为True，作为训练集
            idx = dt.apply(lambda x: not (x.month == 4 and x.day > 10)).values
        elif dataset == 'weibo':
            dt = pd.to_datetime(m_metadata['pub_time'], unit='s', utc=True).dt.tz_convert('Asia/Shanghai')
            # 8-18点之间的值为True，否则为False
            idx = dt.apply(lambda x: x.hour < 18 and x.hour >= 8).values
        elif dataset == 'aps':
            idx = pd.to_datetime(m_metadata['pub_time']).apply(lambda x: x.year <= 1997).values
        else:
            idx = np.array([True] * len(m_metadata))
        # 选取符合条件的casid（过滤器是idx）,我的理解可能是数组，数组里面有很多数值
        cas = m_metadata[idx]['casid'].values
        rng = np.random.default_rng(0)
        # 把casid打乱
        rng.shuffle(cas)
        train_pos, val_pos = int(train_portion * len(cas)), int((train_portion + val_portion) * len(cas))
        train_cas, val_cas, test_cas = np.split(cas, [train_pos, val_pos])
        all_idx['train'] = train_cas
        type_map.update(dict(zip(train_cas, [1] * len(train_cas))))
        all_idx['val'] = val_cas
        type_map.update(dict(zip(val_cas, [2] * len(val_cas))))
        all_idx['test'] = test_cas
        type_map.update(dict(zip(test_cas, [3] * len(test_cas))))
        reset_cas = set(metadata['casid']) - set(train_cas) - set(val_cas) - set(test_cas)
        type_map.update(dict(zip(list(reset_cas), [0] * len(reset_cas))))
        # 返回的是符合条件的casid集合（train/valid/test），以及每个casid：0/1/2的类型映射字典
        return all_idx, type_map

    all_label = all_data[all_data['time'] < predict_time * time_unit].groupby(by='cas', as_index=False)['id'].count()
    all_label = dict(zip(all_label['cas'], all_label['id']))
    m_data = []
    for cas, df in all_data.groupby(by='cas'):
        m_data.extend(get_label(df, observe_time * time_unit, all_label))
    all_data = pd.concat(m_data, axis=0)
    # 合法的casid的数组（级联大于10的，并且过滤了最多100条的，最后一位的label记录了observe_time之后有多少级联，所以label！=-1表明该记录是级联发生的最晚时间）
    all_idx, type_map = data_split(all_data[all_data['label'] != -1]['cas'].values)
    # type字段映射all_data中的每个cas属于哪个集（train/valid/test） 
    all_data['type'] = all_data['cas'].apply(lambda x: type_map[x])
    # 滤掉type等于0的行
    all_data = all_data[all_data['type'] != 0]
    # 对于要筛掉一部分数据的测试集单独筛选，all_idx['test']是一个数组，里面是casid
    # 根据casid在all_data中筛选，all_data肯定有一个筛选observe_time的字段，是time,所以这里的time是observe_time/2
    # 对于超过了observe_time/2的级联，将其label设置为-1，将最后一条不是-1的label值取出来，加上超过observe_time/2的级联的数量
    if sub_test:
        # all_data['cas']对应all_idx['test']的每一项
        for cas in all_idx['test']:
            cases = all_data[all_data['cas'] == cas]
            # 计算总数
            total = len(cases)
            # label是cases[label]中不是-1的那个数值
            number = 0
            for index in range(total):
                if cases.iloc[index]['label'] == -1 and cases.iloc[index]['time'] > observe_time / 2:
                    number += 1
                    cases.at[index, 'label'] = -2

            # 更新label列，不等于-1的部分
            cases.loc[cases['label'] != -1, 'label'] += number

            # 删除label等于-2的行
            cases = cases[cases['label'] != -2]

            # 更新all_data中的数据
            all_data.loc[all_data['cas'] == cas] = cases
            

    """all_idx is used for baselines to select the cascade id, so it don't need to be remapped"""
    # 对于all_data中的pub_time,abs_time和time进行转换，然后按照abs_time和id原地排序，然后对于src,dst,cas进行映射
    data_transformation(dataset, all_data, time_unit, min_time, param)
    all_data.to_csv(f'data/{dataset}_split.csv', index=False)
    pk.dump(all_idx, open(f'data/{dataset}_idx2.pkl', 'wb'))
    log.info(
        f"Total Trans num is {len(all_data)}, Train cas num is {len(all_idx['train'])}, "
        f"Val cas num is {len(all_idx['val'])}, Test cas num is {len(all_idx['test'])}")
    return Data(all_data, is_split=True)


def get_data(dataset, observe_time, predict_time, train_time, val_time, test_time, time_unit,
             log: logging.Logger, param):
    a = time.time()
    """
    data stores all diffusion behaviors, in the form of (id,src,dst,cas,time). The `id` refers to the
    id of the interaction; `src`,`dst`,`cas`,`time` means that user `dst` forwards the message `cas` from `src`
    after `time` time has elapsed since the publication of cascade `cas`. 
    -----------------
    metadata stores the metadata of cascades, including the publication time, publication user, etc.
    """
    data: pd.DataFrame = pd.read_csv(f'data/{dataset}.csv')
    metadata = pd.read_csv(f'data/{dataset}_metadata.csv')
    min_time = min(metadata['pub_time'])
    data = pd.merge(data, metadata, left_on='cas', right_on='casid')
    data = data[['id', 'src', 'dst', 'cas', 'time', 'pub_time']]
    # observe_time是针对不同的数据集在文件中写死的，不是参数传进来的，要看time_unit
    param['max_time'] = {'user': 1, 'cas': param['observe_time']}
    data['label'] = -1
    data.sort_values(by='id', inplace=True, ignore_index=True)
    log.info(
        f"Min time is {min_time}, Train time is {train_time}, Val time is {val_time}, Test time is {test_time}, Time unit is {time_unit}")
    return_data = get_split_data(dataset, observe_time, predict_time, time_unit, data, min_time,
                                 metadata, log, param, sub_test=param['subtest'])
    b = time.time()
    log.info(f"Time cost for loading data is {b - a}s")
    return return_data
