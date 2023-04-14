# -*- coding: utf-8 -*-

 
#%%
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
from datetime import datetime
import matplotlib.pyplot as plt
import features_create
import pdb
import os
from Conv_Blocks import Inception_Block_V1, Inception_Block_V2
pd.options.mode.chained_assignment = None
from torch.utils.data import Dataset
import pdb
import berrar_rating
import pickle
#%% model training parameter
testing=False
server=True



valid_set="exact"
encoder_list=["TE"]      #"LSTM","GRU","TE"

if testing==True:
    print("In testing mode")

#%%%
if server!=True:
    data_path="C:/Users/calvi/Desktop/2023 soccer prediction challenge/"
    result_path="C:/Users/calvi/Desktop/2023 soccer prediction challenge/"
elif  server==True:
    data_path="/data_pool_1/soccer-pred-23/"
    result_path="/home/c_yeung/workspace6/python/soccer_challenge_23/"

#%% hyperparameters
if testing==False:
    inception_version=1
    window_size=5 #num of previous match considered
    features_per_window=8
    epochs=42 #num of epochs for model training
    batch_size,num_workers=100 ,2 #num of samlpe per update, num of cpu
    error_batch_size=50
else:

    window_size=5 #num of previous match considered
    features_per_window=8
    epochs=42 #num of epochs for model training
    batch_size,num_workers=100 ,2 #num of samlpe per update, num of cpu
    error_batch_size=50 #report error per error_batch_size batch
#%% layers hyperparameters
num_layer=1
p=0

embedding_dim=5 #[1,2,4,8,16]

inception_version=1 #[1,2]
d_model=1
d_f=1 #[1,2,4]

LSTM_input_size=features_per_window+embedding_dim+embedding_dim
LSTM_hidden_size=features_per_window+embedding_dim+embedding_dim
LSTM_num_layers=1 #[1,2,4,8,16,24]
LSTM_dropout=0 #[0,0.1]
LSTM_bidirectional=False

GRU_input_size=features_per_window+embedding_dim+embedding_dim
GRU_hidden_size=features_per_window+embedding_dim+embedding_dim
GRU_num_layers=1 #[1,2,4,8,16,24]
GRU_dropout=0 #[0,0.1]
GRU_bidirectional=False

TE_d_model=features_per_window+embedding_dim+embedding_dim
TE_nhead=1
TE_dim_feedforward=2048 #[1,8,64,512,2048,4096]
TE_dropout=0.1 #[0,0.1]

LR_in_features=features_per_window+embedding_dim+embedding_dim
LR_out_features=3
#%%

flag="5yr"
df=pd.read_csv("/data_pool_1/soccer-pred-23/final/dataset/trainset_22-23_exact_44.csv")
df=df.drop(columns=('Round'))
df=df.drop(columns=('index'))

additional_df=pd.read_csv("/data_pool_1/soccer-pred-23/final/dataset/Matches2Add.csv")
additional_df=additional_df.dropna()

# pdb.set_trace()
df1=pd.concat([df, additional_df], axis=0)
df1=df1.reset_index(drop=True) #reset index
df1=df1.reset_index()
df1,Berrar_Recencyfeature_team_dict=features_create.Berrar_Recencyfeature_train_final(df1,n=5)
trainset=df1[:]
trainset.to_csv("/home/c_yeung/workspace6/python/soccer_challenge_23/final_model_2/prediction/features_data.csv",index=False)
##get the ratings dict and hyperparameters
team_ratings_dict={}
berr_para={}
for i in trainset.Lge.unique():
    with open(f"/data_pool_1/soccer-pred-23/final/berrar_ratings/{i}_team_ratings_dict.pickle", 'rb') as f:
        team_ratings_dict[i]= pickle.load(f)
    with open(f"/data_pool_1/soccer-pred-23/final/berrar_ratings/{i}_berrarratings_hyperparameters.pickle", 'rb') as f:
        berr_para[i]= pickle.load(f)


#%% validation set
validation_set=pd.read_csv("/data_pool_1/soccer-pred-23/final/dataset/validset_22-23_44.csv")
# validation_set_1=pd.read_csv(data_path+f"split dataset/validset_{year}_{valid_set}_round1.csv")
# validation_set_2=pd.read_csv(data_path+f"split dataset/validset_{year}_{valid_set}_round2.csv")
# validation_set_1=validation_set_1.drop(["index"],axis=1)
# validation_set_2=validation_set_2.drop(["index"],axis=1)
# validation_set_1=validation_set_1.reset_index()
# validation_set_2=validation_set_2.reset_index()
# valid1=validation_set_1[:]
# valid2=validation_set_2[:]
team_dict=Berrar_Recencyfeature_team_dict.copy()
# valid1=features_create.Berrar_Recencyfeature_valid(valid1,team_dict,n=window_size)
# team_dict=features_create.Berrar_Recencyfeature_update(valid1,team_dict,n=window_size)
# valid2=features_create.Berrar_Recencyfeature_valid(valid2,team_dict,n=window_size)
#%% features scaling

# minmax= MinMaxScaler(feature_range = (0,1))
# minmax_deltaT=MinMaxScaler(feature_range = (0,1))
#%% encode WDL and Team name 
char2idx = {'W': 0, 'D': 1, 'L': 2}
trainset["WDL"].replace(char2idx,inplace=True)
# valid1["WDL"].replace(char2idx,inplace=True)
# valid2["WDL"].replace(char2idx,inplace=True)


team_to_index = {team_name: i for i, team_name in enumerate(np.unique(np.concatenate((df['HT'].tolist(), df['AT'].tolist()), axis=0)))}
trainset["HT_num"]=trainset["HT"]
trainset["AT_num"]=trainset["AT"]
trainset["HT_num"].replace(team_to_index ,inplace=True)
trainset["AT_num"].replace(team_to_index ,inplace=True)
# valid1["HT_num"]=valid1["HT"]
# valid2["HT_num"]=valid2["HT"]
# valid1["AT_num"]=valid1["AT"]
# valid2["AT_num"]=valid2["AT"]
# valid1["HT_num"].replace(team_to_index ,inplace=True)
# valid2["HT_num"].replace(team_to_index ,inplace=True)
# valid1["AT_num"].replace(team_to_index ,inplace=True)
# valid2["AT_num"].replace(team_to_index ,inplace=True)
#%%
# for i in valid1["HT"].tolist()+valid1["AT"].tolist()+valid2["HT"].tolist()+valid2["AT"].tolist():
#     if i not in team_to_index.keys():
#         print(i)

#%% drop row with nan for training set and fill it with 0 for validation set for now
trainset=trainset.dropna()
# valid1=valid1.fillna(0)
# valid2=valid2.fillna(0)
#%% Specify variables of interest

input_vars= ["HT_num", "AT_num", 'attacking_stength_1_HT', 'defensive_stength_1_HT',
       'strength_opposition_1_HT', 'home_advantage_1_HT',
       'attacking_stength_1_AT', 'defensive_stength_1_AT',
       'strength_opposition_1_AT', 'home_advantage_1_AT',
       'attacking_stength_2_HT', 'defensive_stength_2_HT',
       'strength_opposition_2_HT', 'home_advantage_2_HT',
       'attacking_stength_2_AT', 'defensive_stength_2_AT',
       'strength_opposition_2_AT', 'home_advantage_2_AT',
       'attacking_stength_3_HT', 'defensive_stength_3_HT',
       'strength_opposition_3_HT', 'home_advantage_3_HT',
       'attacking_stength_3_AT', 'defensive_stength_3_AT',
       'strength_opposition_3_AT', 'home_advantage_3_AT',
       'attacking_stength_4_HT', 'defensive_stength_4_HT',
       'strength_opposition_4_HT', 'home_advantage_4_HT',
       'attacking_stength_4_AT', 'defensive_stength_4_AT',
       'strength_opposition_4_AT', 'home_advantage_4_AT',
       'attacking_stength_5_HT', 'defensive_stength_5_HT',
       'strength_opposition_5_HT', 'home_advantage_5_HT',
       'attacking_stength_5_AT', 'defensive_stength_5_AT',
       'strength_opposition_5_AT', 'home_advantage_5_AT']
target_vars = ["WDL"]
# target_vars = ['HS', 'AS']

#%% get the first window_size match index of all team
# team_first_n_match_dict={}
# team_match_id={}
# for i in np.unique(trainset[['HT', 'AT']].values):
#     team_first_n_match_dict[i]=[]
#     team_match_id[i]=[]
# for j in range(len(trainset)):
#     row=trainset.iloc[j]
#     #pdb.set_trace()
#     if len(team_first_n_match_dict[row.HT])<window_size:
#         team_first_n_match_dict[row.HT].append(row["index"])
#     if len(team_first_n_match_dict[row.AT])<window_size:
#         team_first_n_match_dict[row.AT].append(row["index"])
#     team_match_id[row.HT].append(row["index"])
#     team_match_id[row.AT].append(row["index"])
# team_first_n_match = [item for sublist in list(team_first_n_match_dict.values()) for item in sublist]


# idx_all_train = np.repeat(True,len(trainset))
# valid_slice_idx=np.repeat(True,len(trainset))
# for k in team_first_n_match:
#     valid_slice_idx[k]=False
#%% Data class
idx_all_train = np.repeat(True,len(trainset))
class train_data():
    def __init__(self,idx=idx_all_train):
        self.idx = idx      
        self.valid_idx = trainset.index
    def __len__(self):
        return int(np.sum(idx_all_train))
    def __getitem__(self, i):
        j = self.valid_idx[i]
        row=trainset[trainset["index"]==j]
        x = row[input_vars]
        x = torch.from_numpy(x.to_numpy(dtype="float64"))
        y = row[target_vars]
        y = torch.from_numpy(y.to_numpy(dtype="float64"))
        return x,  y


class valid_data(Dataset):
    def __init__(self, row, input_vars, target_vars):
        self.row = row
        self.input_vars = input_vars
        self.target_vars = target_vars

    def __len__(self):
        return 1

    def __getitem__(self, i):
        x = torch.from_numpy(self.row[self.input_vars].to_numpy(dtype="float64"))
        y = 0
        return x, y


# idx_all_valid1 = np.repeat(True,len(valid1))
# class valid1_data():
#     def __init__(self,idx=idx_all_train):
#         self.idx = idx      
#         self.valid_idx = valid1.index
#     def __len__(self):
#         return int(np.sum(idx_all_valid1))
#     def __getitem__(self, i):
#         j = self.valid_idx[i]
#         row=valid1[valid1["index"]==j]
#         x = row[input_vars]
#         x = torch.from_numpy(x.to_numpy(dtype="float64"))
#         y = row[target_vars]
#         y = torch.from_numpy(y.to_numpy(dtype="float64"))
#         return x,  y
    
# idx_all_valid2 = np.repeat(True,len(valid2))
# class valid2_data():
#     def __init__(self,idx=idx_all_train):
#         self.idx = idx      
#         self.valid_idx = valid2.index
#     def __len__(self):
#         return int(np.sum(idx_all_valid2))
#     def __getitem__(self, i):
#         j = self.valid_idx[i]
#         row=valid2[valid2["index"]==j]
#         x = row[input_vars]
#         x = torch.from_numpy(x.to_numpy(dtype="float64"))
#         y = row[target_vars]
#         y = torch.from_numpy(y.to_numpy(dtype="float64"))
#         return x,  y
#%% Model
def positional_encoding(src):
  # src = X_cat0; d_model = 15

  pos_encoding = torch.zeros_like(src)
  seq_len = pos_encoding.shape[0]
  d_model = pos_encoding.shape[1]

  for i in range(d_model):
    for pos in range(seq_len):
      if i % 2 == 0:
        pos_encoding[pos,i] = np.sin(pos/100**(2*i/d_model))
      else:
        pos_encoding[pos,i] = np.cos(pos/100**(2*i/d_model))
  # plt.imshow(pos_encoding.cpu().numpy())
  return pos_encoding.float()

class timesnet(nn.Module):
    def __init__(self):  #pick up all specification vars from the global environment
        super(timesnet, self).__init__()    
        # for action one-hot
        if embedding_dim!=0:
            self.emb=nn.Embedding(len(team_to_index.keys()), embedding_dim)
        if encoder=="LSTM":
            self.encoder = nn.LSTM(input_size=LSTM_input_size,hidden_size=LSTM_hidden_size,num_layers=LSTM_num_layers,dropout=LSTM_dropout,bidirectional=LSTM_bidirectional,batch_first=True)
        elif encoder=="GRU":
            self.encoder = nn.GRU(input_size=GRU_input_size,hidden_size=GRU_hidden_size,num_layers=GRU_num_layers,dropout=GRU_dropout,bidirectional=GRU_bidirectional,batch_first=True)
        elif encoder=="TE":
            self.encoder = nn.TransformerEncoderLayer(d_model=TE_d_model, nhead=TE_nhead, dim_feedforward=TE_dim_feedforward, dropout=TE_dropout,batch_first=True).to(device)
        self.lin = nn.Linear(LR_in_features, LR_out_features, bias=True, device=None, dtype=None)
        if inception_version==1:
            self.conv = nn.Sequential(Inception_Block_V1(d_model, d_ff),
                                      nn.GELU(),Inception_Block_V1(d_ff, d_model))
        elif inception_version==2:
            self.conv= nn.Sequential(Inception_Block_V2(d_model, d_ff),
                                      nn.GELU(),Inception_Block_V2(d_ff, d_model))

        self.NN= nn.ModuleList()
        for num_layer in range(1):
            self.NN.append(nn.Linear(LR_in_features, LR_in_features, bias=True))
        self.dropout = nn.Dropout(p)
        # print(self)        

    def forward(self, X):
        #print(X.shape)        
        X_recent_features=torch.reshape(X[:,:,2:],(X.shape[0],window_size,features_per_window)) #batch_size(100),5,8
        X_HT=X[:,:,0]
        X_AT=X[:,:,1]
        if embedding_dim!=0:
            X_HT=self.emb(X_HT.int()) 
            X_AT=self.emb(X_AT.int())
            #import pdb;pdb.set_trace()
            X_HT=X_HT.repeat(1,window_size,1)
            X_AT=X_AT.repeat(1,window_size,1)
        
            X_cat=torch.cat([X_recent_features, X_HT,X_AT], dim=2)
        else:
            X_cat=X_recent_features
        X_cat=X_cat.unsqueeze(1)
        X_cat=self.conv(X_cat)
        X_cat=X_cat.squeeze(1)
        X_cat=X_cat.float()
        src = X_cat+ positional_encoding(X_cat).to(device)
        
        if encoder=="TE":
            X_encoded = self.encoder(src)
        else:
            X_encoded,_ = self.encoder(src)
        
        X_encoded=X_encoded[:,-1,:]
        for layer in self.NN[:]:
            X_encoded=layer(X_encoded)
            X_encoded=F.tanh(X_encoded) 
            X_encoded=self.dropout(X_encoded)
        out=self.lin(X_encoded)
        #import pdb;pdb.set_trace()
        out=F.softmax(out, dim=1)
        return out,X_encoded

#%% cost function

# def rmse(prd_HS,prd_AS,HS,AS):
#     prd_HS=round(prd_HS)
#     prd_AS=round(prd_AS)
#     loss=(0.5*((HS-prd_HS)**2+(AS-prd_AS)**2))**0.5
#     return loss


def RPS(pred,Y):
    #import pdb;pdb.set_trace()
    pr_w=pred[:,0].float()
    pr_d=pred[:,1].float()
    result=Y.squeeze()
    num_classes = 3
    one_hot = torch.zeros(len(result), num_classes)
    i=0
    for val in result:
        one_hot[i, val.long()] = 1
        i+=1
    one_hot=one_hot.to(device)
    #import pdb;pdb.set_trace()
    loss=0.5*((pr_w-one_hot[:,0])**2+((pr_w-one_hot[:,0])+(pr_d-one_hot[:,1]))**2)
    loss=torch.mean(loss) 
    return loss

#%% dataloader 

train_dataset = train_data()
train_loader = DataLoader(train_dataset,shuffle=True,batch_size=batch_size,num_workers=num_workers,drop_last=True)
# valid1_dataset = valid1_data()      
# valid1_loader = DataLoader(valid1_dataset,shuffle=False,batch_size=batch_size,num_workers=num_workers,drop_last=False)
# valid2_dataset = valid2_data()  
# valid2_loader = DataLoader(valid2_dataset,shuffle=False,batch_size=batch_size,num_workers=num_workers,drop_last=False)

#%% training,valid,test one epoch
def model_epoch(dataloader, model, optimiser, scheduler,epochtype):
    #pdb.set_trace()
    if epochtype=="train":
        model.train()     #turn training off if (val or test)
    else:
        model.eval()
     
    
    size = len(dataloader.dataset)
    loss_rollingmean = 0.
    # if epochtype != "train":
    #     pdb.set_trace()
    for batch, (X, Y) in enumerate(dataloader):
        batch_total=len(dataloader)
        #print("batch",batch+1,"/", batch_total)
        X, Y = X.to(device), Y.to(device)
        
        pred,_ = model(X)
        Loss = RPS(pred,Y)
        loss_rollingmean = loss_rollingmean+(Loss-loss_rollingmean)/(1+batch)
        
        if epochtype=="train":
            optimiser.zero_grad()
            Loss.backward()
            optimiser.step()
    
        if batch % error_batch_size == 0:
            #print("batch",batch,"/", batch_total)
            loss=Loss
            loss, current = loss.item(), batch * X.shape[0]
            print(f"loss: {loss:>7f}| batch: {batch}/{batch_total} | sample: [{current:>5d}/{size:>5d}] | lr: {optimiser.param_groups[0]['lr']}")
    loss_rollingmean= loss_rollingmean.detach().cpu().numpy().item()
    print("epoch ended")
    print(f"Epoch loss:    mean: {loss_rollingmean:>7f}")

    return loss_rollingmean

#%% training,valid,test,mutiple epoch

def model_train(epochs):
  torch.cuda.empty_cache(); import gc; gc.collect()
  global model 
  model = timesnet().to(device)
  #optimiser  = optim.RMSprop(model.parameters(),lr=0.01,eps=1e-16)
  optimiser  = optim.Adam(model.parameters(),lr=0.01,eps=1e-16)
  scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser,factor=.1,patience=3,verbose=True)

  #model.load_state_dict(torch.load("/content/gdrive/MyDrive/COMP6200project/Soccer/Data/1processed/MDLstate_20210730am_working_1GRU20210816star3"))

  trainloss_hist = pd.DataFrame(columns=["epoch","trn_L","valid1","valid2","valid"])
  time_start = datetime.now()
  model_params={}
  
  for t in range(epochs):
      torch.cuda.empty_cache(); import gc; gc.collect()
      print(f"Epoch {t}\n-------------------------------")
      trainloss = model_epoch(train_loader, model, optimiser, scheduler,"train")
      with torch.no_grad():
          valid1_loss = 0
          valid2_loss = 0
      epochloss = pd.DataFrame(np.concatenate( ( np.array([t]) , np.asarray([trainloss]),np.array([valid1_loss]),np.array([valid2_loss]),np.array([(valid1_loss+valid2_loss)/2]) ))).T
      epochloss.columns = trainloss_hist.columns
      trainloss_hist = pd.concat([trainloss_hist, epochloss], ignore_index=True)
      model_params[t]=model.state_dict()
      
      
      if optimiser.param_groups[0]["lr"] < 1E-7:
        break
  time_end = datetime.now()
  trainable_params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
  train_time=time_end-time_start
  train_time=train_time.total_seconds()
  
  '''
  save the models when looping and read it back and perform validation
  '''
  
  
  # with torch.no_grad():
  #   torch.cuda.empty_cache(); import gc; gc.collect()
  #   valid_loss=model_epoch(valid_loader, model, optimiser, scheduler,"valid")
  # with torch.no_grad():
  #   torch.cuda.empty_cache(); import gc; gc.collect()
  #   test_loss=model_epoch(test_loader, model, optimiser, scheduler,"test")
  return trainloss_hist,train_time,trainable_params_num,model_params #,valid_loss,test_loss
#%%
def predict(dataloader_x):
    with torch.no_grad():
        model.eval()                     
        for batch, (X, Y) in enumerate(dataloader_x):
            X = X.unsqueeze(0)
            X = X.to(device)
            pred,_ = model(X)
            all_pred = pred.detach().cpu()
    return all_pred


#%%
# if __name__ == '__main__': 
    # hyper parameters from the timesnet paper
    #model
    #for i in [0]:
inception_version=1
d_ff=4
embedding_dim=1
TE_dim_feedforward=1
TE_dropout=0
encoder="TE"
num_layer=10
p=0.2                                 
LSTM_input_size=features_per_window+embedding_dim+embedding_dim
LSTM_hidden_size=features_per_window+embedding_dim+embedding_dim
GRU_input_size=features_per_window+embedding_dim+embedding_dim
GRU_hidden_size=features_per_window+embedding_dim+embedding_dim
TE_d_model=features_per_window+embedding_dim+embedding_dim
LR_in_features=features_per_window+embedding_dim+embedding_dim

#load model
device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = timesnet().to(device)
model.load_state_dict(torch.load("/home/c_yeung/workspace6/python/soccer_challenge_23/final_model_2/model_44_param/final_44_41.pt"))

data_out=pd.DataFrame(columns=validation_set.columns)
#get the features for first row in prediction set
for index, row in validation_set.iterrows():
    #predict the result 
    print(row.ID)
    row_out=row.copy()
    row=features_create.Berrar_Recencyfeature_valid_final(row,team_dict,n=window_size)
    # import pdb;pdb.set_trace()
    row["HT_num"]=row["HT"]
    row["AT_num"]=row["AT"]
    row["HT_num"]=team_to_index[row["HT_num"]]
    row["AT_num"]=team_to_index[row["AT_num"]]

    row=row.fillna(0)

    valid_dataset=valid_data(row,input_vars,target_vars)
    valid_loader = DataLoader(valid_dataset,shuffle=False,batch_size=1,num_workers=num_workers,drop_last=False)

    result_prediction=predict(valid_loader)
    result_prediction=result_prediction.tolist()[0]

    row_out.prd_W=result_prediction[0]
    row_out.prd_D=result_prediction[1]
    row_out.prd_L=result_prediction[2]
    max_index = result_prediction.index(max(result_prediction))
    if max_index==0:
        row_out.WDL="W"
    elif max_index==1:
        row_out.WDL="D"
    elif max_index==2:
        row_out.WDL="L"
    #predict the goal

    G_H_hat,G_A_hat=berrar_rating.berrar_rating_valid_final(row_out,team_ratings_dict[row_out.Lge],x=berr_para[row_out.Lge])
    
    G_H_hat,G_A_hat=int(G_H_hat),int(G_A_hat)

    row_out['HS'] = G_H_hat
    row_out['AS'] = G_A_hat
    row_out['GD'] = G_H_hat-G_A_hat
    row_out.prd_HS=G_H_hat
    row_out.prd_AS=G_A_hat
    #write the value in row out
    # update 
    data_out=pd.concat([data_out,row_out.to_frame().transpose()])
    row=row_out.copy()
    row["WDL"]=char2idx[row["WDL"]]
    team_dict=features_create.Berrar_Recencyfeature_update_final(row,team_dict,n=window_size)





    #loop

data_out['HS'] = -1
data_out['AS'] = -1
data_out['GD'] =  0

data_out.to_csv("/home/c_yeung/workspace6/python/soccer_challenge_23/final_model_2/prediction/prediction_final.csv",index=False)
    
            
            
        



