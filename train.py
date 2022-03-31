import torch
from codes.model import ResUnet_LSTM_L, ResUnet, Loss_With_Weight
from torch.utils.data import DataLoader
from codes.dataloader import init_dataset, STEAD_Dataset
from codes.visualize_and_evaluate import evaluate
from codes import configs

if configs.device == "cuda":
    device = torch.device("cuda")
else:
    device = torch.device("cpu")



if __name__ == "__main__":
    train_set,val_set,noise_set = init_dataset()
    train_set = STEAD_Dataset(train_set)
    val_set = STEAD_Dataset(val_set)

    train_dataloader = DataLoader(train_set, shuffle=True, batch_size=configs.BATCH_SIZE, num_workers=configs.NUM_WORKERS, drop_last=True)
    eval_dataloader = DataLoader(val_set, shuffle=True,batch_size=1, num_workers=configs.NUM_WORKERS, drop_last=True)

    mseLoss = torch.nn.MSELoss()
    #QuackNet = ResUnet().to(device)# 不带lstm的版本
    if configs.model_type == 'ResUnet_BiLSTM':
        QuackNet = ResUnet_LSTM_L(True).to(device)# 带lstm的版本
    if configs.model_type == 'ResUnet':
        QuackNet = ResUnet().to(device)

    optimizer = torch.optim.Adam(QuackNet.parameters(),
                    lr=configs.LEARNING_RATE,
                    betas=(0.9, 0.999),
                    eps=configs.EPS,
                    weight_decay=configs.WEIGHT_DECAY,
                    amsgrad=False)

    total_step = 0
    loss_sum = 0
    loss_list = []

    for epoch in range(configs.EPOCH_NUM):
        # training
        for index,data in enumerate(train_dataloader,0):
            stream,label_p,label_s,p_start,s_start,coda_end = data

            # 清空累加梯度
            QuackNet.zero_grad()
            output = QuackNet(stream.to(device))
            loss = 0
            loss += Loss_With_Weight(output[:,0,:], label_p.to(device))
            loss += Loss_With_Weight(output[:,1,:], label_s.to(device))
            loss.backward()
            loss_sum += loss.item()
            optimizer.step()
            total_step += 1
            if total_step%configs.LOSS_RECORD_ITER == 0:
                loss_list.append(
                    "{},".format(total_step) + str(round(loss.item(),2))+'\n'
                )
                print("total_step:{} loss:{}".format(total_step,loss_sum/configs.LOSS_RECORD_ITER))
                loss_sum = 0


        torch.save(QuackNet.state_dict(),configs.result_dir+"models/{}_iter{}.pt".format(configs.label,total_step))
        QuackNet.train()

    f=open(configs.result_dir+"loss/{}_loss.txt".format(configs.label),"w")
    f.writelines(loss_list)
    f.close()
    QuackNet.eval()
    evaluate(configs.result_dir+"models/{}_iter{}.pt".format(configs.label,total_step),"batchsize:{} steps:{}".format(configs.BATCH_SIZE,total_step),eval_dataloader,50,45)   