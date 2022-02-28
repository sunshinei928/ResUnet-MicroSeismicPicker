import torch
from torch.utils.data import DataLoader
from codes.dataloader import init_dataset, STEAD_Dataset
from codes.visualize_and_evaluate import evaluate
from codes import configs

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

train_set,val_set,noise_set = init_dataset()
#train_set = STEAD_Dataset(train_set)
val_set = STEAD_Dataset(val_set)

#train_dataloader = DataLoader(train_set, shuffle=True, batch_size=configs.BATCH_SIZE, num_workers=64, drop_last=True)
eval_dataloader = DataLoader(val_set, shuffle=True,batch_size=1, num_workers=4, drop_last=True)

model_path = "C:\\Users\\29147\\source\\repos\\ResUnet-MicroSeismicPicker\\QuackPicker_lstm_iter25000.pt"

if __name__ == '__main__':
    evaluate(model_path,"stm_iter25000_b0.5",eval_dataloader,50,16)