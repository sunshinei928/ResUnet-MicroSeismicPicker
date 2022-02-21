
hdf5_path = "D:\\merged.hdf5"
csv_path = "D:\\merged.csv"

evaluation_csv_path = "D:\\result\\evaluation.csv"
model_save_dir = "D:\\result\\"     #   文件夹保留后面的斜线


#   数据筛选配置
trace_category = 'earthquake_local'
source_magnitude_lt = 2


label = "raw" #给本次测试起个标签，在csv文件中以此区分一次训练结果，不要重名
device = "cuda"
EPOCH_NUM = 5
BATCH_SIZE = 16
NUM_WORKERS = 4
LEARNING_RATE = 0.001
CHECKPOINT_ITER = 1000

#   标签分布偏差
label_bias = 20

