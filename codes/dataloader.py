from torch.functional import split
from torch.utils.data.dataset import Dataset
from codes import configs
import torch
import pandas as pd
import h5py
import random
from torch.utils.data import  random_split
import obspy
import numpy as np
import re


dtfl = h5py.File(configs.hdf5_path, 'r')
#   官方github改的
def make_stream(start,end,dataset):
    '''
    input: hdf5 dataset
    output: obspy stream
    
    '''
    data = np.array(dataset)
    tr_E = obspy.Trace(data=data[start:end, 0])
    tr_E.stats.delta = 0.01
    tr_E.stats.channel = dataset.attrs['receiver_type']+'E'
    tr_E.stats.station = dataset.attrs['receiver_code']
    tr_E.stats.network = dataset.attrs['network_code']

    tr_N = obspy.Trace(data=data[start:end, 1])
    tr_N.stats.delta = 0.01
    tr_N.stats.channel = dataset.attrs['receiver_type']+'N'
    tr_N.stats.station = dataset.attrs['receiver_code']
    tr_N.stats.network = dataset.attrs['network_code']

    tr_Z = obspy.Trace(data=data[start:end, 2])
    tr_Z.stats.delta = 0.01
    tr_Z.stats.channel = dataset.attrs['receiver_type']+'Z'
    tr_Z.stats.station = dataset.attrs['receiver_code']
    tr_Z.stats.network = dataset.attrs['network_code']

    stream = obspy.Stream([tr_E, tr_N, tr_Z])

    return stream

#   对hd5中str格式的信噪比转换，若大于config文件中的阈值则返回true
def snr_convert(stri):
    stri = str(stri)
    if(stri == "nan"):
        return False
    snr = re.sub(r'(^\[\s*)|(\s*\]$)', "", stri)
    snr = re.sub(r'\s+',",",snr)
    snr = snr.split(',')
    if((float(snr[0]) > configs.snr_DB_gt) & (float(snr[1]) > configs.snr_DB_gt) & (float(snr[2]) > configs.snr_DB_gt)):
        return True
    else:
        return False
    
#   直接从hdf5文件中获得数据列表的方式
def init_dataset():
    df = pd.read_csv(configs.csv_path)
    print(f'total events in csv file: {len(df)}')
    # filterering the dataframe
    #df = df[(df.trace_category == configs.trace_category) & (df.source_magnitude < configs.source_magnitude_lt) & (df.p_arrival_sample > 400) & (list(map(snr_convert,df['snr_db'])))]
    df = df[(df.trace_category == configs.trace_category) & (df.source_magnitude < configs.source_magnitude_lt) & (df.p_arrival_sample > 400) & (df.s_arrival_sample - df.p_arrival_sample < 1000) & (df.p_status == 'manual') & (df.s_status == 'manual')]
    print(f'total events selected: {len(df)}')

    ev_list = df['trace_name'].to_list()
    train_set, val_set, _ = random_split(
                dataset=ev_list,
                lengths=[configs.data_for_train,configs.data_for_val,ev_list.__len__()-configs.data_for_train-configs.data_for_val],
                generator=torch.Generator().manual_seed(12))

    # 读取噪声数据
    df = pd.read_csv(configs.csv_path)
    df = df[df.trace_category == 'noise']
    print(f'total noise selected: {len(df)}')
    noise_set = df['trace_name'].to_list()
    return train_set,val_set,noise_set



class STEAD_Dataset(Dataset):
    def __init__(self,ev_list):
        self.ev_list = ev_list
        #self.dtfl = dtfl   #只能用全局变量，否则无法使用多线程，不知道为啥

    # 高斯标签
    def GaussianLabel(self,lenth,miu):
        # sigma = 10
        # denominator = 1.25331415
        label = torch.rand(lenth)
        for index in range(lenth):
            label[index] = (1/2*np.sqrt(2*np.pi))*np.exp(-((index-miu)**2/(2*4)))*(1/1.25331415)
        return label

    def __getitem__(self,index):
        dataset = dtfl.get('data/'+str(self.ev_list[index]))
        #   随机截取16秒数据
        random_start = random.randint(9,int(dataset.attrs['p_arrival_sample']))
        p_start = int(dataset.attrs['p_arrival_sample']) - random_start
        s_start = int(dataset.attrs['s_arrival_sample']) - random_start
        #coda_end = int(dataset.attrs['coda_end_sample']) - random_start
        coda_end = 0    #not needed
        stream = make_stream(random_start,random_start+1600,dataset)
        stream = torch.Tensor(stream)
        # stream[0] = stream[0] / (stream[0].max() - stream[0].min() + 0.000001)
        # stream[1] = stream[1] / (stream[1].max() - stream[1].min() + 0.000001)
        # stream[2] = stream[2] / (stream[2].max() - stream[2].min() + 0.000001)
        mean = torch.mean(stream, dim=1)
        std = torch.std(stream, dim=1)
        stream[0] = (stream[0]-mean[0])/(std[0]+0.000001)
        stream[1] = (stream[1]-mean[1])/(std[1]+0.000001)
        stream[2] = (stream[2]-mean[2])/(std[2]+0.000001)

        label_p = self.GaussianLabel(1600,p_start)
        label_s = self.GaussianLabel(1600,s_start)
        return stream,label_p,label_s,p_start,s_start,coda_end

    def __len__(self):
        return self.ev_list.__len__()


#   噪声
class STEAD_Dataset_noised(Dataset):
    def __init__(self,ev_list,noise_list):
        self.ev_list = ev_list
        self.noise_list = noise_list
        self.noise_count = noise_list.__len__()
        #self.dtfl = dtfl   #只能用全局变量，否则无法使用多线程，不知道为啥
    def __getitem__(self,index):
        puredata = dtfl.get('data/'+str(self.ev_list[index]))
        noisedata = dtfl.get('data/'+str(self.noise_list[(index+1)%self.noise_count - 1]))

        #   随机截取30秒数据
        random_start = random.randint(9,int(puredata.attrs['p_arrival_sample']))
        # p_start = int(puredata.attrs['p_arrival_sample']) - random_start
        # s_start = int(puredata.attrs['s_arrival_sample']) - random_start
        # coda_end = int(puredata.attrs['coda_end_sample']) - random_start
        stream_pure = torch.Tensor(make_stream(random_start-1,random_start+3001,puredata))
        stream_noise = torch.Tensor(make_stream(random_start-1,random_start+3001,noisedata))
        #   normalization
        stream_pure[0] = (stream_pure[0] - np.average(stream_pure[0]))/(max(stream_pure[0])-min(stream_pure[0]))
        stream_pure[1] = (stream_pure[1] - np.average(stream_pure[1]))/(max(stream_pure[1])-min(stream_pure[1]))
        stream_pure[2] = (stream_pure[2] - np.average(stream_pure[2]))/(max(stream_pure[2])-min(stream_pure[2]))

        stream_noise[0] = 0.2*(stream_noise[0] - np.average(stream_noise[0]))/(max(stream_noise[0])-min(stream_noise[0]))
        stream_noise[1] = 0.2*(stream_noise[1] - np.average(stream_noise[1]))/(max(stream_noise[1])-min(stream_noise[1]))
        stream_noise[2] = 0.2*(stream_noise[2] - np.average(stream_noise[2]))/(max(stream_noise[2])-min(stream_noise[2]))        

        # 噪音有瑕疵数据
        if torch.isnan(stream_noise[0].sum()):
            stream_noise[0] = torch.zeros(1,3002)
        if torch.isnan(stream_noise[1].sum()):
            stream_noise[1] = torch.zeros(1,3002)
        if torch.isnan(stream_noise[2].sum()):
            stream_noise[2] = torch.zeros(1,3002)

        stream = stream_pure + stream_noise      
        return stream,stream_pure,stream_noise

    def __len__(self):
        return self.ev_list.__len__()
