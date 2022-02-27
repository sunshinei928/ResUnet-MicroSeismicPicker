import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
from codes.model import ResUnet_LSTM
from pandas.core.indexes.base import Index
from codes import configs
from copy import deepcopy
import os.path
import os
from queue import Queue
import threading

class Evaluator():
    def __init__(self):
        self.default1 = {
            "th0.1":False,"th0.2":False,"th0.3":False,"th0.4":False,
            "th0.5":False,"th0.6":False,"th0.7":False,"th0.8":False,
            "th0.9":False,"th0.95":False
        }
        self.default2 = {
            "th0.1":-1,"th0.2":-1,"th0.3":-1,"th0.4":-1,
            "th0.5":-1,"th0.6":-1,"th0.7":-1,"th0.8":-1,
            "th0.9":-1,"th0.95":-1
        }
        self.default3 = {
            "th0.1":0,"th0.2":0,"th0.3":0,"th0.4":0,
            "th0.5":0,"th0.6":0,"th0.7":0,"th0.8":0,
            "th0.9":0,"th0.95":0
        }

        self.thresholds = {
            "th0.1":0.1,"th0.2":0.2,"th0.3":0.3,"th0.4":0.4,
            "th0.5":0.5,"th0.6":0.6,"th0.7":0.7,"th0.8":0.8,
            "th0.9":0.9,"th0.95":0.95
        }

        self.TP_p = deepcopy(self.default3)
        self.FP_p = deepcopy(self.default3)
        self.FN_p = deepcopy(self.default3)
        self.TP_s = deepcopy(self.default3)
        self.FP_s = deepcopy(self.default3)
        self.FN_s = deepcopy(self.default3)

        self.P_detected = deepcopy(self.default1)
        self.P_maxprob = deepcopy(self.default2)
        self.P_maxprob_index = deepcopy(self.default2) #  拾取时间下标

        self.S_detected = deepcopy(self.default1)
        self.S_maxprob = deepcopy(self.default2)
        self.S_maxprob_index = deepcopy(self.default2) #  拾取时间下标

        self.call_out_p = deepcopy(self.default1)
        self.call_out_s = deepcopy(self.default1)
  
        #   若某次拾取时间判定误差time_bias，
        #   认为该条数据中的P/S到达事件已被有效识别
        #   否则 FN += 1（每条数据只有一个P/S波到达）  
    def next_sample(self):
        for key in list(self.call_out_p.keys()):
            if self.call_out_p[key] == False:
                self.FN_p[key] += 1

        for key in list(self.call_out_s.keys()):
            if self.call_out_s[key] == False:
                self.FN_s[key] += 1        


        self.call_out_p = deepcopy(self.default1)
        self.call_out_s = deepcopy(self.default1)    

    def feed_prob(self, index, p_prob, s_prob, p_start, s_start, time_bias):
        '''
        评估方式：
        一段模型预测连续大于threshold为一次拾取事件，取其中峰值为拾取点，
        计算其与真实值的差距，若偏差小于time_bias则为TP，否则为FP。
        若没有任何有效拾取符合条件，则记录一次FN
        '''
        for key in list(self.thresholds.keys()):

            #   评估P波拾取
            if p_prob >= self.thresholds[key]:
                if self.P_detected[key] ^ True:
                    #   有效拾取
                    self.P_detected[key] = True
                if p_prob > self.P_maxprob[key]:
                    self.P_maxprob[key] = p_prob
                    self.P_maxprob_index[key] = index
            else:
                if self.P_detected[key] ^ False:
                    #   拾取事件结束
                    if abs(self.P_maxprob_index[key] - p_start) <= time_bias:
                        self.TP_p[key] += 1
                        self.call_out_p[key] = True
                    else:
                        self.FP_p[key] += 1

                    #   重置，等待下次拾取
                    self.P_detected[key] = False
                    self.P_maxprob[key] = -1  # reset maxprob
                    self.P_maxprob_index[key] = -1 # reset index                    

            #   评估S波拾取
            if s_prob >= self.thresholds[key]:
                if self.S_detected[key] ^ True:
                    #   有效拾取
                    self.S_detected[key] = True
                if s_prob > self.S_maxprob[key]:
                    self.S_maxprob[key] = s_prob
                    self.S_maxprob_index[key] = index
            else:
                if self.S_detected[key] ^ False:
                    #   拾取事件结束
                    if abs(self.S_maxprob_index[key] - s_start) <= time_bias:
                        self.TP_s[key] += 1
                        self.call_out_s[key] = True
                    else:
                        self.FP_s[key] += 1

                    #   重置，等待下次拾取
                    self.S_detected[key] = False
                    self.S_maxprob[key] = -1  # reset maxprob
                    self.S_maxprob_index[key] = -1 # reset index    

    '''
    多线程 将其他对象的评估结果加过来
    '''
    def merge_evaluator(self,evaluator):
        for key in list(self.thresholds.keys()):
            self.TP_p[key] += evaluator.TP_p[key]
            self.FP_p[key] += evaluator.FP_p[key]
            self.FN_p[key] += evaluator.FN_p[key]
            self.TP_s[key] += evaluator.TP_s[key]
            self.FP_s[key] += evaluator.FP_s[key]
            self.FN_s[key] += evaluator.FN_s[key]

    def get_result_print(self):
        e = 10e-7
        #   返回评估结果
        self.Precision_p = deepcopy(self.default2)
        self.Recall_p = deepcopy(self.default2)
        self.Precision_s = deepcopy(self.default2)
        self.Recall_s = deepcopy(self.default2)

        self.F1_score_p = deepcopy(self.default2)
        self.F1_score_s = deepcopy(self.default2)

        for key in list(self.thresholds.keys()):
            self.Precision_p[key] = self.TP_p[key]/(self.TP_p[key] + self.FP_p[key]+e)
            self.Recall_p[key] = self.TP_p[key]/(self.TP_p[key] + self.FN_p[key]+e)
            self.Precision_s[key] = self.TP_s[key]/(self.TP_s[key] + self.FP_s[key]+e)
            self.Recall_s[key] = self.TP_s[key]/(self.TP_s[key] + self.FN_s[key]+e)

            self.F1_score_p[key] = 2*(self.Precision_p[key]*self.Recall_p[key])/(self.Precision_p[key]+self.Recall_p[key]+e)
            self.F1_score_s[key] = 2*(self.Precision_s[key]*self.Recall_s[key])/(self.Precision_s[key]+self.Recall_s[key]+e)

        #   打印结果
        print("")
        print("\033[0;36m",end='')
        print("*******************************Evaluation result(p_wave):*******************************")
        print("{:^20}\t{:^20}\t{:^20}\t{:^20}".format("threshold:","Precision:","Recall:","F1_score:"))
        print("{:^20}\t{:^20}\t{:^20}\t{:^20}".format("0.1:",round(self.Precision_p["th0.1"],2),round(self.Recall_p["th0.1"],2),round(self.F1_score_p["th0.1"],2)))
        print("{:^20}\t{:^20}\t{:^20}\t{:^20}".format("0.2:",round(self.Precision_p["th0.2"],2),round(self.Recall_p["th0.2"],2),round(self.F1_score_p["th0.2"],2)))
        print("{:^20}\t{:^20}\t{:^20}\t{:^20}".format("0.3:",round(self.Precision_p["th0.3"],2),round(self.Recall_p["th0.3"],2),round(self.F1_score_p["th0.3"],2)))
        print("{:^20}\t{:^20}\t{:^20}\t{:^20}".format("0.4:",round(self.Precision_p["th0.4"],2),round(self.Recall_p["th0.4"],2),round(self.F1_score_p["th0.4"],2)))
        print("{:^20}\t{:^20}\t{:^20}\t{:^20}".format("0.5:",round(self.Precision_p["th0.5"],2),round(self.Recall_p["th0.5"],2),round(self.F1_score_p["th0.5"],2)))
        print("{:^20}\t{:^20}\t{:^20}\t{:^20}".format("0.6:",round(self.Precision_p["th0.6"],2),round(self.Recall_p["th0.6"],2),round(self.F1_score_p["th0.6"],2)))
        print("{:^20}\t{:^20}\t{:^20}\t{:^20}".format("0.7:",round(self.Precision_p["th0.7"],2),round(self.Recall_p["th0.7"],2),round(self.F1_score_p["th0.7"],2)))
        print("{:^20}\t{:^20}\t{:^20}\t{:^20}".format("0.8:",round(self.Precision_p["th0.8"],2),round(self.Recall_p["th0.8"],2),round(self.F1_score_p["th0.8"],2)))
        print("{:^20}\t{:^20}\t{:^20}\t{:^20}".format("0.9:",round(self.Precision_p["th0.9"],2),round(self.Recall_p["th0.9"],2),round(self.F1_score_p["th0.9"],2)))
        print("{:^20}\t{:^20}\t{:^20}\t{:^20}".format("0.95:",round(self.Precision_p["th0.95"],2),round(self.Recall_p["th0.95"],2),round(self.F1_score_p["th0.95"],2)))
        print("****************************************************************************************")
        print("\033[0m")

        print("")
        print("\033[0;36m",end='')
        print("*******************************Evaluation result(s_wave):*******************************")
        print("{:^20}\t{:^20}\t{:^20}\t{:^20}".format("threshold:","Precision:","Recall:","F1_score:"))
        print("{:^20}\t{:^20}\t{:^20}\t{:^20}".format("0.1:",round(self.Precision_s["th0.1"],2),round(self.Recall_s["th0.1"],2),round(self.F1_score_s["th0.1"],2)))
        print("{:^20}\t{:^20}\t{:^20}\t{:^20}".format("0.2:",round(self.Precision_s["th0.2"],2),round(self.Recall_s["th0.2"],2),round(self.F1_score_s["th0.2"],2)))
        print("{:^20}\t{:^20}\t{:^20}\t{:^20}".format("0.3:",round(self.Precision_s["th0.3"],2),round(self.Recall_s["th0.3"],2),round(self.F1_score_s["th0.3"],2)))
        print("{:^20}\t{:^20}\t{:^20}\t{:^20}".format("0.4:",round(self.Precision_s["th0.4"],2),round(self.Recall_s["th0.4"],2),round(self.F1_score_s["th0.4"],2)))
        print("{:^20}\t{:^20}\t{:^20}\t{:^20}".format("0.5:",round(self.Precision_s["th0.5"],2),round(self.Recall_s["th0.5"],2),round(self.F1_score_s["th0.5"],2)))
        print("{:^20}\t{:^20}\t{:^20}\t{:^20}".format("0.6:",round(self.Precision_s["th0.6"],2),round(self.Recall_s["th0.6"],2),round(self.F1_score_s["th0.6"],2)))
        print("{:^20}\t{:^20}\t{:^20}\t{:^20}".format("0.7:",round(self.Precision_s["th0.7"],2),round(self.Recall_s["th0.7"],2),round(self.F1_score_s["th0.7"],2)))
        print("{:^20}\t{:^20}\t{:^20}\t{:^20}".format("0.8:",round(self.Precision_s["th0.8"],2),round(self.Recall_s["th0.8"],2),round(self.F1_score_s["th0.8"],2)))
        print("{:^20}\t{:^20}\t{:^20}\t{:^20}".format("0.9:",round(self.Precision_s["th0.9"],2),round(self.Recall_s["th0.9"],2),round(self.F1_score_s["th0.9"],2)))
        print("{:^20}\t{:^20}\t{:^20}\t{:^20}".format("0.95:",round(self.Precision_s["th0.95"],2),round(self.Recall_s["th0.95"],2),round(self.F1_score_s["th0.95"],2)))
        print("****************************************************************************************")
        print("\033[0m")

    def save_result_dataframe(self, model_path, info):
        header = ['model_path','info','(P_th0.1)Precision','(P_th0.2)Precision',
        '(P_th0.3)Precision','(P_th0.4)Precision','(P_th0.5)Precision','(P_th0.6)Precision','(P_th0.7)Precision',
        '(P_th0.8)Precision','(P_th0.9)Precision','(P_th0.95)Precision','(P_th0.1)Recall','(P_th0.2)Recall',
        '(P_th0.3)Recall','(P_th0.4)Recall','(P_th0.5)Recall','(P_th0.6)Recall','(P_th0.7)Recall',
        '(P_th0.8)Recall','(P_th0.9)Recall','(P_th0.95)Recall','(P_th0.1)F1_score','(P_th0.2)F1_score',
        '(P_th0.3)F1_score','(P_th0.4)F1_score','(P_th0.5)F1_score','(P_th0.6)F1_score','(P_th0.7)F1_score',
        '(P_th0.8)F1_score','(P_th0.9)F1_score','(P_th0.95)F1_score','(S_th0.1)Precision','(S_th0.2)Precision',
        '(S_th0.3)Precision','(S_th0.4)Precision','(S_th0.5)Precision','(S_th0.6)Precision','(S_th0.7)Precision',
        '(S_th0.8)Precision','(S_th0.9)Precision','(S_th0.95)Precision','(S_th0.1)Recall','(S_th0.2)Recall',
        '(S_th0.3)Recall','(S_th0.4)Recall','(S_th0.5)Recall','(S_th0.6)Recall','(S_th0.7)Recall',
        '(S_th0.8)Recall','(S_th0.9)Recall','(S_th0.95)Recall','(S_th0.1)F1_score','(S_th0.2)F1_score',
        '(S_th0.3)F1_score','(S_th0.4)F1_score','(S_th0.5)F1_score','(S_th0.6)F1_score','(S_th0.7)F1_score',
        '(S_th0.8)F1_score','(S_th0.9)F1_score','(S_th0.95)F1_score']

        eva_df_old = pd.DataFrame([],columns=header,index=None)



        if os.path.isfile(configs.evaluation_csv_path):
            eva_df_old = pd.read_csv(configs.evaluation_csv_path)
        else:
            #  若csv文件不存在,则创建一个只有表头的空表
            eva_df_old.to_csv(configs.evaluation_csv_path,index=None)


        df_list = []
        df_list.append({
            "model_path":model_path,
            "info":info,
            "(P_th0.1)Precision":self.Precision_p['th0.1'],
            "(P_th0.2)Precision":self.Precision_p['th0.2'],
            "(P_th0.3)Precision":self.Precision_p['th0.3'],
            "(P_th0.4)Precision":self.Precision_p['th0.4'],
            "(P_th0.5)Precision":self.Precision_p['th0.5'],
            "(P_th0.6)Precision":self.Precision_p['th0.6'],
            "(P_th0.7)Precision":self.Precision_p['th0.7'],
            "(P_th0.8)Precision":self.Precision_p['th0.8'],
            "(P_th0.9)Precision":self.Precision_p['th0.9'],
            "(P_th0.95)Precision":self.Precision_p['th0.95'],
            "(P_th0.1)Recall":self.Recall_p['th0.1'],
            "(P_th0.2)Recall":self.Recall_p['th0.2'],
            "(P_th0.3)Recall":self.Recall_p['th0.3'],
            "(P_th0.4)Recall":self.Recall_p['th0.4'],
            "(P_th0.5)Recall":self.Recall_p['th0.5'],
            "(P_th0.6)Recall":self.Recall_p['th0.6'],
            "(P_th0.7)Recall":self.Recall_p['th0.7'],
            "(P_th0.8)Recall":self.Recall_p['th0.8'],
            "(P_th0.9)Recall":self.Recall_p['th0.9'],
            "(P_th0.95)Recall":self.Recall_p['th0.95'],
            "(P_th0.1)F1_score":self.F1_score_p['th0.1'],
            "(P_th0.2)F1_score":self.F1_score_p['th0.2'],
            "(P_th0.3)F1_score":self.F1_score_p['th0.3'],
            "(P_th0.4)F1_score":self.F1_score_p['th0.4'],
            "(P_th0.5)F1_score":self.F1_score_p['th0.5'],
            "(P_th0.6)F1_score":self.F1_score_p['th0.6'],
            "(P_th0.7)F1_score":self.F1_score_p['th0.7'],
            "(P_th0.8)F1_score":self.F1_score_p['th0.8'],
            "(P_th0.9)F1_score":self.F1_score_p['th0.9'],
            "(P_th0.95)F1_score":self.F1_score_p['th0.95'],
            "(S_th0.1)Precision":self.Precision_s['th0.1'],
            "(S_th0.2)Precision":self.Precision_s['th0.2'],
            "(S_th0.3)Precision":self.Precision_s['th0.3'],
            "(S_th0.4)Precision":self.Precision_s['th0.4'],
            "(S_th0.5)Precision":self.Precision_s['th0.5'],
            "(S_th0.6)Precision":self.Precision_s['th0.6'],
            "(S_th0.7)Precision":self.Precision_s['th0.7'],
            "(S_th0.8)Precision":self.Precision_s['th0.8'],
            "(S_th0.9)Precision":self.Precision_s['th0.9'],
            "(S_th0.95)Precision":self.Precision_s['th0.95'],
            "(S_th0.1)Recall":self.Recall_s['th0.1'],
            "(S_th0.2)Recall":self.Recall_s['th0.2'],
            "(S_th0.3)Recall":self.Recall_s['th0.3'],
            "(S_th0.4)Recall":self.Recall_s['th0.4'],
            "(S_th0.5)Recall":self.Recall_s['th0.5'],
            "(S_th0.6)Recall":self.Recall_s['th0.6'],
            "(S_th0.7)Recall":self.Recall_s['th0.7'],
            "(S_th0.8)Recall":self.Recall_s['th0.8'],
            "(S_th0.9)Recall":self.Recall_s['th0.9'],
            "(S_th0.95)Recall":self.Recall_s['th0.95'],
            "(S_th0.1)F1_score":self.F1_score_s['th0.1'],
            "(S_th0.2)F1_score":self.F1_score_s['th0.2'],
            "(S_th0.3)F1_score":self.F1_score_s['th0.3'],
            "(S_th0.4)F1_score":self.F1_score_s['th0.4'],
            "(S_th0.5)F1_score":self.F1_score_s['th0.5'],
            "(S_th0.6)F1_score":self.F1_score_s['th0.6'],
            "(S_th0.7)F1_score":self.F1_score_s['th0.7'],
            "(S_th0.8)F1_score":self.F1_score_s['th0.8'],
            "(S_th0.9)F1_score":self.F1_score_s['th0.9'],
            "(S_th0.95)F1_score":self.F1_score_s['th0.95']
        })

        eva_df_new = pd.DataFrame(df_list,columns=header,index=None)
        eva_df_old.append(eva_df_new).to_csv(configs.evaluation_csv_path,index=None)


# 显示波形
def showWave(Title,wave):
    fig = plt.figure()
    plt.rcParams["figure.figsize"] = (20, 3)
    plt.title(Title)
    plt.plot(wave)


#   可视化原始波形和标签
def showRawWave(wave,p_start,s_start,coda_end):
    fig = plt.figure()

    fig_E = fig.add_subplot(311)
    plt.title("E")
    plt.plot(wave[0],color='#336699')
    plt.rcParams["figure.figsize"] = (15, 10)
    ymin, ymax = fig_E.get_ylim()
    pl = plt.vlines(p_start,ymin,ymax,color='b', linewidth=2, label='P-arrival')
    sl = plt.vlines(s_start,ymin,ymax,color='#993399', linewidth=2, label='S-arrival')
    el = plt.vlines(coda_end,ymin,ymax,color='#663300', linewidth=2, label='coda_end')
    plt.legend(handles=[pl, sl, el], loc = 'upper right', borderaxespad=0., prop={'weight':'bold'})#在轴上放置图例  

    fig_N = fig.add_subplot(312)
    plt.title("N")
    plt.plot(wave[1],color='#336699')
    plt.rcParams["figure.figsize"] = (15, 10)
    ymin, ymax = fig_N.get_ylim()
    pl = plt.vlines(p_start,ymin,ymax,color='b', linewidth=2, label='P-arrival')
    sl = plt.vlines(s_start,ymin,ymax,color='#993399', linewidth=2, label='S-arrival')
    el = plt.vlines(coda_end,ymin,ymax,color='#663300', linewidth=2, label='coda_end')
    plt.legend(handles=[pl, sl, el], loc = 'upper right', borderaxespad=0., prop={'weight':'bold'})#在轴上放置图例  

    fig_Z = fig.add_subplot(313)
    plt.title("Z")
    plt.plot(wave[2],color='#336699')
    plt.rcParams["figure.figsize"] = (15, 10)
    ymin, ymax = fig_Z.get_ylim()
    pl = plt.vlines(p_start,ymin,ymax,color='b', linewidth=2, label='P-arrival')
    sl = plt.vlines(s_start,ymin,ymax,color='#993399', linewidth=2, label='S-arrival')
    el = plt.vlines(coda_end,ymin,ymax,color='#663300', linewidth=2, label='coda_end')
    plt.legend(handles=[pl, sl, el], loc = 'upper right', borderaxespad=0., prop={'weight':'bold'})#在轴上放置图例  

    plt.show()

def showProb(wave,p_start,s_start,coda_end):
    fig = plt.figure()

    fig_E = fig.add_subplot(311)
    plt.title("E")
    plt.plot(wave[0],color='#336699')
    plt.rcParams["figure.figsize"] = (15, 10)
    ymin, ymax = fig_E.get_ylim()
    pl = plt.vlines(p_start,ymin,ymax,color='b', linewidth=2, label='P-arrival')
    sl = plt.vlines(s_start,ymin,ymax,color='#993399', linewidth=2, label='S-arrival')
    el = plt.vlines(coda_end,ymin,ymax,color='#663300', linewidth=2, label='coda_end')
    plt.legend(handles=[pl, sl, el], loc = 'upper right', borderaxespad=0., prop={'weight':'bold'})#在轴上放置图例  

    fig_N = fig.add_subplot(312)
    plt.title("N")
    plt.plot(wave[1],color='#336699')
    plt.rcParams["figure.figsize"] = (15, 10)
    ymin, ymax = fig_N.get_ylim()
    pl = plt.vlines(p_start,ymin,ymax,color='b', linewidth=2, label='P-arrival')
    sl = plt.vlines(s_start,ymin,ymax,color='#993399', linewidth=2, label='S-arrival')
    el = plt.vlines(coda_end,ymin,ymax,color='#663300', linewidth=2, label='coda_end')
    plt.legend(handles=[pl, sl, el], loc = 'upper right', borderaxespad=0., prop={'weight':'bold'})#在轴上放置图例  

    plt.show()

#   。。。
def showDeltaWave(wave,p_start,s_start,coda_end):
    delta0 = torch.Tensor([wave[0,i+1]-wave[0,i] for i in range(3001)]).unsqueeze(0)
    delta1 = torch.Tensor([wave[1,i+1]-wave[1,i] for i in range(3001)]).unsqueeze(0)
    delta2 = torch.Tensor([wave[2,i+1]-wave[2,i] for i in range(3001)]).unsqueeze(0)
    delta = torch.cat((delta0,delta1,delta2),0)
    fig = plt.figure()

    fig_E = fig.add_subplot(311)
    plt.title("E")
    plt.plot(delta[0],color='#336699')
    plt.rcParams["figure.figsize"] = (15, 10)
    ymin, ymax = fig_E.get_ylim()
    pl = plt.vlines(p_start,ymin,ymax,color='b', linewidth=2, label='P-arrival')
    sl = plt.vlines(s_start,ymin,ymax,color='#993399', linewidth=2, label='S-arrival')
    el = plt.vlines(coda_end,ymin,ymax,color='#663300', linewidth=2, label='coda_end')
    plt.legend(handles=[pl, sl, el], loc = 'upper right', borderaxespad=0., prop={'weight':'bold'})#在轴上放置图例  

    fig_N = fig.add_subplot(312)
    plt.title("N")
    plt.plot(delta[1],color='#336699')
    plt.rcParams["figure.figsize"] = (15, 10)
    ymin, ymax = fig_N.get_ylim()
    pl = plt.vlines(p_start,ymin,ymax,color='b', linewidth=2, label='P-arrival')
    sl = plt.vlines(s_start,ymin,ymax,color='#993399', linewidth=2, label='S-arrival')
    el = plt.vlines(coda_end,ymin,ymax,color='#663300', linewidth=2, label='coda_end')
    plt.legend(handles=[pl, sl, el], loc = 'upper right', borderaxespad=0., prop={'weight':'bold'})#在轴上放置图例  

    fig_Z = fig.add_subplot(313)
    plt.title("Z")
    plt.plot(delta[2],color='#336699')
    plt.rcParams["figure.figsize"] = (15, 10)
    ymin, ymax = fig_Z.get_ylim()
    pl = plt.vlines(p_start,ymin,ymax,color='b', linewidth=2, label='P-arrival')
    sl = plt.vlines(s_start,ymin,ymax,color='#993399', linewidth=2, label='S-arrival')
    el = plt.vlines(coda_end,ymin,ymax,color='#663300', linewidth=2, label='coda_end')
    plt.legend(handles=[pl, sl, el], loc = 'upper right', borderaxespad=0., prop={'weight':'bold'})#在轴上放置图例  

    plt.show()


#   可视化预测结果与真实值对比
def showresult(prob_p,prob_s,p_start,s_start,coda_end):
    fig = plt.figure()

    fig_pwave_prob = fig.add_subplot(211)
    plt.title("Probability of p-wave arrived")
    plt.plot(prob_p,'g')
    plt.rcParams["figure.figsize"] = (15, 10)
    ymin, ymax = fig_pwave_prob.get_ylim()
    pl = plt.vlines(p_start,ymin,ymax,color='b', linewidth=2, label='P-arrival')
    sl = plt.vlines(s_start,ymin,ymax,color='#993399', linewidth=2, label='S-arrival')
    el = plt.vlines(coda_end,ymin,ymax,color='#663300', linewidth=2, label='coda_end')
    plt.legend(handles=[pl, sl, el], loc = 'upper right', borderaxespad=0., prop={'weight':'bold'})#在轴上放置图例  

    fig_swave_prob = fig.add_subplot(212)
    plt.title("Probability of s-wave arrived")
    plt.plot(prob_s,'r')
    plt.rcParams["figure.figsize"] = (15, 10)
    ymin, ymax = fig_swave_prob.get_ylim()
    pl = plt.vlines(p_start,ymin,ymax,color='b', linewidth=2, label='P-arrival')
    sl = plt.vlines(s_start,ymin,ymax,color='#993399', linewidth=2, label='S-arrival')
    el = plt.vlines(coda_end,ymin,ymax,color='#663300', linewidth=2, label='coda_end')
    plt.legend(handles=[pl, sl, el], loc = 'upper right', borderaxespad=0., prop={'weight':'bold'})#在轴上放置图例  

    plt.show()


#   计算波形SI-SNR，用作损失函数
def SI_SNR(v_pred,v_real):
    # 将batch，channel维度去除，当一维向量直接计算
    v_pred = v_pred.view((1,-1))
    v_real = v_real.view((1,-1))
    X_t = (torch.mul(v_pred,v_real).sum()/torch.mul(v_real,v_real).sum())*v_real
    X_e = v_pred - X_t
    si_sdr = 10*torch.log10((X_t*X_t).sum()/(X_e*X_e).sum())
    return si_sdr


def th_calculator(dataloader,model,queue, th_num):
    total_waves = dataloader.__len__()
    for index,data in enumerate(dataloader,0):
        print("\rEvaluating...    {}%".format(int((index+1)*100/total_waves)),end='')
        stream,label_p,label_s,p_start,s_start,coda_end = data
        queue.put({
            "data":model(stream.to(configs.device)).squeeze(0),
            "p_start":p_start,
            "s_start":s_start
        })

    for i in range(th_num):
        queue.put("done")

def th_evaluator(evaluator,time_bias,queue,model_path, info):
    while True:
        output = queue.get()
        if output == "done":
            return
        for i in range(output['data'].shape[1]):
            p_prob = output['data'][0][i]
            s_prob = output['data'][1][i]
            evaluator.feed_prob(i,p_prob,s_prob,output['p_start'],output['s_start'],time_bias)
            
        evaluator.next_sample()


def evaluate(model_path, info, eval_loader, time_bias, th_num):
    q = Queue(th_num + 1)
    model = ResUnet_LSTM(False)
    model.load_state_dict(torch.load(model_path))
    model.eval().to(configs.device)

    evaluator_list = []
    thread_list = []
    for i in range(th_num):
        evaluator_list.append(Evaluator())

    for i in range(th_num):
        thread_list.append(threading.Thread(target=th_evaluator, args=(evaluator_list[i],time_bias,q,model_path,info)))

    '''
    评估方式：
    一段模型预测连续大于threshold为一次拾取事件，取其中峰值为拾取点，
    计算其与真实值的差距，若偏差小于time_bias则为TP，否则为FP。
    若没有任何有效拾取符合条件，则记录一次FN
    '''
    th_feeder = threading.Thread(target=th_calculator, args=(eval_loader,model_eval,q,th_num))
    th_feeder.start()

    for i in range(th_num):
        thread_list[i].start()
    for i in range(th_num):
        thread_list[i].join()

    for i in range(1,th_num):
        evaluator_list[0].merge_evaluator(evaluator_list[i])

    #   返回评估结果
    evaluator_list[0].get_result_print()
    evaluator_list[0].save_result_dataframe(model_path, info)