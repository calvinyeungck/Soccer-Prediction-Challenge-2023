# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 10:47:03 2023

@author: calvi
"""
#%%
import pandas as pd
import features_create
import numpy as np


df=pd.read_excel("TrainingSet-FINAL.xlsx")
df=df[df["Sea"].isin(df["Sea"].unique()[-5:])] #use only 5 season


valid_set=pd.read_excel("PredictionSet-FINAL.xlsx")
valid_set1=pd.read_excel("PredictionSet_2023_01_31.xlsx")
required_league=list(valid_set.Lge.unique())
required_league1=list(valid_set1.Lge.unique())
required_league1.remove("NOR1")

df1=df[df["Lge"].isin(required_league)]
df1=df1.reset_index(drop=True)
df1=df1.reset_index()


df1=features_create.Round_train(df1)
df1.Sea=" "+df1.Sea

df1=df1.reset_index(drop=True) #reset index
df1=df1.drop(columns=('index')) #drop the column index
df1=df1.reset_index()
df1.to_csv("trainset_22-23_exact_44.csv", index=False)


df2=df[df["Lge"].isin(required_league1)]
df2=df2.reset_index(drop=True)
df2=df2.reset_index()


df2=features_create.Round_train(df2)
df2.Sea=" "+df2.Sea

df2=df2.reset_index(drop=True) #reset index
df2=df2.drop(columns=('index')) #drop the column index
df2=df2.reset_index()
df1.to_csv("trainset_22-23_exact_34.csv", index=False)
#%%
#get two validation set and split into round1 and round2
# def Round_valid():
    
#     df1=df
#     df2=valid_set
#     df_merged = df1.append(df2, ignore_index=True)
#     df_merged=features_create.Round_train(df_merged)
#     df_merged=df_merged[len(df1):]
#     return df_merged

valid_set_out=valid_set

#%% split for 34 and 44 league
valid_set_out_34=pd.DataFrame(columns=list(valid_set_out.columns))
valid_set_out_44=pd.DataFrame(columns=list(valid_set_out.columns))

# import pdb;pdb.set_trace()
for i in valid_set_out.Lge.unique():
    print(i)
    rows=valid_set_out[valid_set_out["Lge"]==i]
    if i in required_league1:

        valid_set_out_34=pd.concat([rows,valid_set_out_34])
        valid_set_out_44=pd.concat([rows,valid_set_out_44])

    else:

        valid_set_out_44=pd.concat([rows,valid_set_out_44])
        
valid_set_out_34=valid_set_out_34.sort_values("ID")
valid_set_out_34=valid_set_out_34.reset_index(drop=True)
valid_set_out_44=valid_set_out_44.sort_values("ID")

valid_set_out_34.to_csv("validset_22-23_34.csv", index=False)
valid_set_out_44.to_csv("validset_22-23_44.csv", index=False)

