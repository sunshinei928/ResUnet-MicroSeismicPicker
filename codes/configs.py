
hdf5_path = "F:\\earthquakedata\\merged.hdf5"
csv_path = "F:\\earthquakedata\\merged.csv"

evaluation_csv_path = "F:\\result\\evaluation.csv"
model_save_dir = "F:\\result\\ckpt\\"     #   文件夹保留后面的斜线


#   数据筛选配置
trace_category = 'earthquake_local'
source_magnitude_lt = 2
data_for_train = 200000
data_for_val = 2000


label = "raw" #给本次测试起个标签，在csv文件中以此区分一次训练结果，不要重名
device = "cuda"
EPOCH_NUM = 5
BATCH_SIZE = 32
NUM_WORKERS = 4
LEARNING_RATE = 0.001
CHECKPOINT_ITER = 1000

#   标签分布偏差
label_bias = 20

