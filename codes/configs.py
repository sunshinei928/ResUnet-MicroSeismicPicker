
# hdf5_path = "/mnt/STEAD/merged.hdf5"
# csv_path = "/mnt/STEAD/merged.csv"

# evaluation_csv_path = "/mnt/result/evaluation.csv"
# model_save_dir = "/mnt/result/ckpt/"     #   文件夹保留后面的斜线



hdf5_path = "F:/earthquakedata/merged.hdf5"
csv_path = "F:/earthquakedata/merged.csv"

evaluation_csv_path = "E:/ResUnet-MicroSeismicPicker/result/eval/evaluation.csv"
#model_save_dir = "E:/ResUnet-MicroSeismicPicker/result/models/"     #   文件夹保留后面的斜线
result_dir = "E:/ResUnet-MicroSeismicPicker/result/"

#   模型类型
#model_type = 'ResUnet'
model_type = 'ResUnet_BiLSTM'
#model_type = 'ResUnet_SelfAttention'

#   数据筛选配置
trace_category = 'earthquake_local'
source_magnitude_lt = 2
data_for_train = 300000
data_for_val = 2000


label = "std_norm" #给本次测试起个标签，在csv文件中以此区分一次训练结果，不要重名
device = "cuda"
EPOCH_NUM = 5
BATCH_SIZE = 64
NUM_WORKERS = 4
LEARNING_RATE = 0.001
EPS = 1e-08
WEIGHT_DECAY = 0


LOSS_RECORD_ITER = 100
#CHECKPOINT_ITER = 100

#   标签分布偏差
label_bias = 20

