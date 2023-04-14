# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 17:41:20 2023

@author: calvi
"""

#%%
import pandas as pd
import numpy as np

def Berrar_Recencyfeature_train_final(dataframe,n=9):
    df1=dataframe
    time_dict_attacking_stength_HT={} #create dict for each features
    time_dict_attacking_stength_AT={}
    time_dict_defensive_stength_HT={}
    time_dict_defensive_stength_AT={}
    time_dict_strength_opposition_HT={}
    time_dict_strength_opposition_AT={}
    time_dict_home_advantage_HT={}
    time_dict_home_advantage_AT={}
    Berrar_Recencyfeature_team_dict={}
    
    
    for num in range(n): #sub_dict for each time n
        time_dict_attacking_stength_HT[num+1]=[np.nan]*len(df1)
        time_dict_attacking_stength_AT[num+1]=[np.nan]*len(df1)
        time_dict_defensive_stength_HT[num+1]=[np.nan]*len(df1)
        time_dict_defensive_stength_AT[num+1]=[np.nan]*len(df1)
        time_dict_strength_opposition_HT[num+1]=[np.nan]*len(df1)
        time_dict_strength_opposition_AT[num+1]=[np.nan]*len(df1)
        time_dict_home_advantage_HT[num+1]=[np.nan]*len(df1)
        time_dict_home_advantage_AT[num+1]=[np.nan]*len(df1)
    
    
    df2=df1[:] 
    #loop for (super) leagues
    for i in ['ARG1', 'AUS1',"AUT1", 'BEL1', 'CHE1', 'CHL1','DNK1' ,'DZA1',"ECU1", ['ENG1', 'ENG2','ENG3', 'ENG4', 'ENG5'], ['FRA1', 'FRA2', 'FRA3'], ['GER1', 'GER2','GER3'],'GRE1', 'HOL1','ISR1', ['ITA1', 'ITA2'],['JPN1', 'JPN2'],"KOR1", 'MAR1', 'MEX1', 'POR1',['RUS1', 'RUS2'], ['SCO1', 'SCO2', 'SCO3', 'SCO4'], ['SPA1', 'SPA2'],"TUN1",['USA1', 'USA2'], "VEN1", 'ZAF1']:
        #print(i)
        if type(i)==str:    
            temp_df=df2[df2["Lge"]==i]
        else:
            temp_df=df2[df2["Lge"].isin(i)]
        team_list=np.unique(temp_df[['HT', 'AT']].values)
        team_info={}
        goal_scored=[np.nan]*n
        goal_conceded=[np.nan]*n
        home_advantage=[np.nan]*n
        strength_opp=[np.nan]*n
        for k in team_list:
            team_info[k] = [0,goal_scored.copy(),goal_conceded.copy(),home_advantage.copy(),strength_opp.copy()] #team appearance, goal scored, goal conceded, home advantage, strength_opposition
        for j in range(len(temp_df)):
            temp_df2=temp_df.iloc[j] #loop the df
            if team_info[temp_df2.HT][0]<=n or team_info[temp_df2.AT][0]<=n or np.nan in team_info[temp_df2.HT][4] or np.nan in team_info[temp_df2.HT][4] or np.nan in team_info[temp_df2.AT][4] or np.nan in team_info[temp_df2.AT][4]:
                if np.nan not in team_info[temp_df2.AT][1] and np.nan not in team_info[temp_df2.AT][2]:
                    team_info[temp_df2.HT][4].insert(0,sum([x - y for x, y in zip(team_info[temp_df2.AT][1], team_info[temp_df2.AT][2])])/n)
                    team_info[temp_df2.HT][4].pop(-1)
                if np.nan not in team_info[temp_df2.HT][1] and np.nan not in team_info[temp_df2.HT][2]:
                    team_info[temp_df2.AT][4].insert(0,sum([x - y for x, y in zip(team_info[temp_df2.HT][1], team_info[temp_df2.HT][2])])/n)
                    team_info[temp_df2.AT][4].pop(-1)
                #import pdb; pdb.set_trace()
                #print(temp_df2.HT,team_info[temp_df2.HT])
                team_info[temp_df2.HT][1].insert(0,temp_df2.HS)
                team_info[temp_df2.HT][1].pop(-1)
                team_info[temp_df2.HT][2].insert(0,temp_df2.AS)
                team_info[temp_df2.HT][2].pop(-1)
                team_info[temp_df2.HT][3].insert(0,1)
                team_info[temp_df2.HT][3].pop(-1)
                team_info[temp_df2.AT][1].insert(0,temp_df2.AS)
                team_info[temp_df2.AT][1].pop(-1)
                team_info[temp_df2.AT][2].insert(0,temp_df2.HS)
                team_info[temp_df2.AT][2].pop(-1)
                team_info[temp_df2.AT][3].insert(0,-1)
                team_info[temp_df2.AT][3].pop(-1)
                team_info[temp_df2.HT][0]+=1
                team_info[temp_df2.AT][0]+=1
            else:
                if np.nan in [team_info[temp_df2.HT][1],team_info[temp_df2.HT][2],team_info[temp_df2.HT][3]]:
                    print("error")
                for l in range(n):
                    time_dict_attacking_stength_HT[l+1][temp_df2["index"]]=team_info[temp_df2.HT][1][l]
                    time_dict_defensive_stength_HT[l+1][temp_df2["index"]]=team_info[temp_df2.HT][2][l]
                    time_dict_home_advantage_HT[l+1][temp_df2["index"]]=team_info[temp_df2.HT][3][l]
                    time_dict_attacking_stength_AT[l+1][temp_df2["index"]]=team_info[temp_df2.AT][1][l]
                    time_dict_defensive_stength_AT[l+1][temp_df2["index"]]=team_info[temp_df2.AT][2][l]
                    time_dict_home_advantage_AT[l+1][temp_df2["index"]]=team_info[temp_df2.AT][3][l]
                    time_dict_strength_opposition_HT[l+1][temp_df2["index"]]=team_info[temp_df2.AT][4][l]
                    time_dict_strength_opposition_AT[l+1][temp_df2["index"]]=team_info[temp_df2.AT][4][l]
  
                team_info[temp_df2.HT][4].insert(0,sum([x - y for x, y in zip(team_info[temp_df2.AT][1], team_info[temp_df2.AT][2])])/n)
                team_info[temp_df2.HT][4].pop(-1)
                team_info[temp_df2.AT][4].insert(0,sum([x - y for x, y in zip(team_info[temp_df2.HT][1], team_info[temp_df2.HT][2])])/n)
                team_info[temp_df2.AT][4].pop(-1)
                team_info[temp_df2.HT][1].insert(0,temp_df2.HS)
                team_info[temp_df2.HT][1].pop(-1)
                team_info[temp_df2.HT][2].insert(0,temp_df2.AS)
                team_info[temp_df2.HT][2].pop(-1)
                team_info[temp_df2.HT][3].insert(0,1)
                team_info[temp_df2.HT][3].pop(-1)
                team_info[temp_df2.AT][1].insert(0,temp_df2.AS)
                team_info[temp_df2.AT][1].pop(-1)
                team_info[temp_df2.AT][2].insert(0,temp_df2.HS)
                team_info[temp_df2.AT][2].pop(-1)
                team_info[temp_df2.AT][3].insert(0,-1)
                team_info[temp_df2.AT][3].pop(-1)
                team_info[temp_df2.HT][0]+=1
                team_info[temp_df2.AT][0]+=1
                #print(temp_df2.HT,team_info[temp_df2.HT])
     
        if type(i)==str:
            Berrar_Recencyfeature_team_dict[i]=team_info 
        else:
            for m in i:    
                Berrar_Recencyfeature_team_dict[m]=team_info                                                                                             
    #Attacking strength feature,Defensive strength feature,strength of opposition, home advantage
    for i in range(n):
        df1[f"attacking_stength_{i+1}_HT"]=time_dict_attacking_stength_HT[i+1]
        df1[f"defensive_stength_{i+1}_HT"]=time_dict_defensive_stength_HT[i+1]
        df1[f"strength_opposition_{i+1}_HT"]=time_dict_strength_opposition_HT[i+1]
        df1[f"home_advantage_{i+1}_HT"]=time_dict_home_advantage_HT[i+1]
        df1[f"attacking_stength_{i+1}_AT"]=time_dict_attacking_stength_AT[i+1]
        df1[f"defensive_stength_{i+1}_AT"]=time_dict_defensive_stength_AT[i+1]
        df1[f"strength_opposition_{i+1}_AT"]=time_dict_strength_opposition_AT[i+1]
        df1[f"home_advantage_{i+1}_AT"]=time_dict_home_advantage_AT[i+1]
    return df1,Berrar_Recencyfeature_team_dict
  
def Berrar_Recencyfeature_valid_final(dataframe,Berrar_Recencyfeature_team_dict,n=9):
    df1=dataframe
    time_dict_attacking_stength_HT={} #create dict for each features
    time_dict_attacking_stength_AT={}
    time_dict_defensive_stength_HT={}
    time_dict_defensive_stength_AT={}
    time_dict_strength_opposition_HT={}
    time_dict_strength_opposition_AT={}
    time_dict_home_advantage_HT={}
    time_dict_home_advantage_AT={}
    for num in range(n): #sub_dict for each time n
        time_dict_attacking_stength_HT[num+1]=[np.nan]
        time_dict_attacking_stength_AT[num+1]=[np.nan]
        time_dict_defensive_stength_HT[num+1]=[np.nan]
        time_dict_defensive_stength_AT[num+1]=[np.nan]
        time_dict_strength_opposition_HT[num+1]=[np.nan]
        time_dict_strength_opposition_AT[num+1]=[np.nan]
        time_dict_home_advantage_HT[num+1]=[np.nan]
        time_dict_home_advantage_AT[num+1]=[np.nan]
    for j in range(1):   
         for l in range(n): #league, team, metrics, number
             #record the features in the valid set
            # import pdb;pdb.set_trace()
            time_dict_attacking_stength_HT[l+1][j]=Berrar_Recencyfeature_team_dict[df1["Lge"]][df1 ["HT"]][1][l]
            time_dict_defensive_stength_HT[l+1][j]=Berrar_Recencyfeature_team_dict[df1["Lge"]][df1["HT"]][2][l]
            time_dict_strength_opposition_HT[l+1][j]=Berrar_Recencyfeature_team_dict[df1["Lge"]][df1["HT"]][4][l]
            time_dict_home_advantage_HT[l+1][j]=Berrar_Recencyfeature_team_dict[df1["Lge"]][df1["HT"]][3][l]
            time_dict_attacking_stength_AT[l+1][j]=Berrar_Recencyfeature_team_dict[df1["Lge"]][df1["AT"]][1][l]
            time_dict_defensive_stength_AT[l+1][j]=Berrar_Recencyfeature_team_dict[df1["Lge"]][df1["AT"]][2][l]
            time_dict_strength_opposition_AT[l+1][j]=Berrar_Recencyfeature_team_dict[df1["Lge"]][df1["AT"]][4][l]
            time_dict_home_advantage_AT[l+1][j]=Berrar_Recencyfeature_team_dict[df1["Lge"]][df1["AT"]][3][l]
             
          #update based on the validset    
    for i in range(n):
        df1[f"attacking_stength_{i+1}_HT"]=time_dict_attacking_stength_HT[i+1][0]
        df1[f"defensive_stength_{i+1}_HT"]=time_dict_defensive_stength_HT[i+1][0]
        df1[f"strength_opposition_{i+1}_HT"]=time_dict_strength_opposition_HT[i+1][0]
        df1[f"home_advantage_{i+1}_HT"]=time_dict_home_advantage_HT[i+1][0]
        df1[f"attacking_stength_{i+1}_AT"]=time_dict_attacking_stength_AT[i+1][0]
        df1[f"defensive_stength_{i+1}_AT"]=time_dict_defensive_stength_AT[i+1][0]
        df1[f"strength_opposition_{i+1}_AT"]=time_dict_strength_opposition_AT[i+1][0]
        df1[f"home_advantage_{i+1}_AT"]=time_dict_home_advantage_AT[i+1][0]      
    return df1

def Berrar_Recencyfeature_update_final(dataframe,Berrar_Recencyfeature_team_dict,n=9):
    df1=dataframe
    for j in range(1): 
        if df1["Lge"] in ['ENG1', 'ENG2','ENG3', 'ENG4', 'ENG5'] :
            league_list=['ENG1', 'ENG2','ENG3', 'ENG4', 'ENG5']
        elif df1["Lge"] in  ['FRA1', 'FRA2', 'FRA3']:
            league_list=['FRA1', 'FRA2', 'FRA3']
        elif df1["Lge"] in ['GER1', 'GER2','GER3']:
            league_list=['GER1', 'GER2','GER3']
        elif df1["Lge"] in ['ITA1', 'ITA2']:
            league_list=['ITA1', 'ITA2']
        elif df1["Lge"] in ['RUS1', 'RUS2']:
            league_list=['RUS1', 'RUS2']
        elif df1["Lge"] in ['SCO1', 'SCO2', 'SCO3', 'SCO4']:
            league_list=['SCO1', 'SCO2', 'SCO3', 'SCO4']
        elif df1["Lge"] in ['SPA1', 'SPA2']:
            league_list=['SPA1', 'SPA2']
        elif df1["Lge"] in ['JPN1', 'JPN2']:
            league_list=['JPN1', 'JPN2']
        elif df1["Lge"] in ['USA1', 'USA2']:
            league_list=['USA1', 'USA2']
        else:
            league_list=[df1["Lge"]]
        for k in league_list:
            #opp strength
            Berrar_Recencyfeature_team_dict[k][df1["HT"]][4].insert(0,sum([x - y for x, y in zip(Berrar_Recencyfeature_team_dict[df1["Lge"]][df1["AT"]][1], Berrar_Recencyfeature_team_dict[df1["Lge"]][df1["AT"]][2])])/n)
            Berrar_Recencyfeature_team_dict[k][df1["HT"]][4].pop(-1)
            Berrar_Recencyfeature_team_dict[k][df1["AT"]][4].insert(0,sum([x - y for x, y in zip(Berrar_Recencyfeature_team_dict[df1["Lge"]][df1["HT"]][1], Berrar_Recencyfeature_team_dict[df1["Lge"]][df1["HT"]][2])])/n)
            Berrar_Recencyfeature_team_dict[k][df1["AT"]][4].pop(-1)
            #attack
            Berrar_Recencyfeature_team_dict[k][df1["HT"]][1].insert(0,df1.HS)
            Berrar_Recencyfeature_team_dict[k][df1["HT"]][1].pop(-1)
            Berrar_Recencyfeature_team_dict[k][df1["AT"]][1].insert(0,df1.AS)
            Berrar_Recencyfeature_team_dict[k][df1["AT"]][1].pop(-1)
            #home advantage
            Berrar_Recencyfeature_team_dict[k][df1["HT"]][3].insert(0,1)
            Berrar_Recencyfeature_team_dict[k][df1["HT"]][3].pop(-1)
            Berrar_Recencyfeature_team_dict[k][df1["AT"]][3].insert(0,-1)
            Berrar_Recencyfeature_team_dict[k][df1["AT"]][3].pop(-1)
            #defense
            Berrar_Recencyfeature_team_dict[k][df1["HT"]][2].insert(0,df1.AS)
            Berrar_Recencyfeature_team_dict[k][df1["HT"]][2].pop(-1)
            Berrar_Recencyfeature_team_dict[k][df1["AT"]][2].insert(0,df1.HS)
            Berrar_Recencyfeature_team_dict[k][df1["AT"]][2].pop(-1)
            #appearence
            Berrar_Recencyfeature_team_dict[k][df1["HT"]][0]+=1
            Berrar_Recencyfeature_team_dict[k][df1["HT"]][0]+=1 
    
    return Berrar_Recencyfeature_team_dict


#%% GD= goal difference GS= Goals Scored and GC = Goals Conceded
def GD_GS_GC_train(dataframe): #df1= train set
    df1=dataframe
    GS_HT=[np.nan]*len(df1)
    GC_HT=[np.nan]*len(df1)
    
    GS_AT=[np.nan]*len(df1)
    GC_AT=[np.nan]*len(df1)
    
    team_GD_GS_GC_dict={}
    for l in df1.Lge.unique(): #loop for each league
        df2=df1[df1["Lge"]==l]    #get league df
        team_GD_GS_GC_dict_per_league={}
        for i in df2.Sea.unique(): #loop for each season
            temp_df=df2[df2["Sea"]==i] #get season df
            for j in np.unique(temp_df[['HT', 'AT']].values): #loop for team in the league
                temp_df_1=temp_df[(temp_df["HT"]==j)|(temp_df["AT"]==j)] #get team df
                temp_GS=0
                temp_GC=0
                for k in range(len(temp_df_1)): #loop the team df
                    temp_df_2=temp_df_1.iloc[k] #get row of the team df
                    if  (k==0)&(j==temp_df_2["HT"]):     
                        temp_GS=temp_df_2["HS"]
                        temp_GC=temp_df_2["AS"]
                    elif (k==0)&(j==temp_df_2["AT"]):   
                        temp_GS=temp_df_2["AS"]
                        temp_GC=temp_df_2["HS"]
                    elif (k!=0)&(j==temp_df_2["HT"]):
                        GS_HT[temp_df_2["index"]]=temp_GS
                        GC_HT[temp_df_2["index"]]=temp_GC
                        temp_GS+=temp_df_2["HS"]
                        temp_GC+=temp_df_2["AS"]
                    elif (k!=0)&(j==temp_df_2["AT"]):
                        GS_AT[temp_df_2["index"]]=temp_GS
                        GC_AT[temp_df_2["index"]]=temp_GC
                        temp_GS+=temp_df_2["HS"]
                        temp_GC+=temp_df_2["AS"]          
                    else:
                        print("error")
                #save the last value here
                if i == df2.Sea.unique()[-1] :
                    team_GD_GS_GC_dict_per_league[j]=[temp_GS,temp_GC]
        team_GD_GS_GC_dict[l]=team_GD_GS_GC_dict_per_league
        
    GD_HT=[x - y for x, y in zip(GS_HT, GC_HT)]
    GD_AT=[x - y for x, y in zip(GS_AT, GC_AT)]
    
    df1["GD_HT"]=GD_HT
    df1["GS_HT"]=GS_HT
    df1["GC_HT"]=GC_HT
    df1["GD_AT"]=GD_AT
    df1["GS_AT"]=GS_AT
    df1["GC_AT"]=GC_AT
    return df1, team_GD_GS_GC_dict

def GD_GS_GC_valid(dataframe,team_GD_GS_GC_dict): #df1= prediction set
    df=dataframe
    df["GS_HT"]=np.nan
    df["GC_HT"]=np.nan
    df["GS_AT"]=np.nan
    df["GC_AT"]=np.nan
    for i in range(len(df)): #loop the prediction set
        df.iloc[i]["GS_HT"]=team_GD_GS_GC_dict[df.iloc[i]["Lge"]][df.iloc[i]["HT"]][0]
        df.iloc[i]["GC_HT"]=team_GD_GS_GC_dict[df.iloc[i]["Lge"]][df.iloc[i]["HT"]][1]
        df.iloc[i]["GS_AT"]=team_GD_GS_GC_dict[df.iloc[i]["Lge"]][df.iloc[i]["AT"]][0]
        #print(df.iloc[i]["AT"],team_GD_GS_GC_dict[df.iloc[i]["Lge"]][df.iloc[i]["AT"]][1])
        df.iloc[i]["GC_AT"]=team_GD_GS_GC_dict[df.iloc[i]["Lge"]][df.iloc[i]["AT"]][1]
        df.iloc[i]["GD_HT"]=df.iloc[i]["GS_HT"]-df.iloc[i]["GC_HT"]
        df.iloc[i]["GD_AT"]=df.iloc[i]["GS_AT"]-df.iloc[i]["GC_AT"]
    return df

#models perform prediction and update the prediction set

def GD_GS_GC_update(dataframe,team_GD_GS_GC_dict):
    df=dataframe
    for i in range(len(df)):
        team_GD_GS_GC_dict[df.iloc[i]["Lge"]][df.iloc[i]["HT"]][0]+=df.iloc[i]["HS"]
        team_GD_GS_GC_dict[df.iloc[i]["Lge"]][df.iloc[i]["HT"]][1]+=df.iloc[i]["AS"]
        team_GD_GS_GC_dict[df.iloc[i]["Lge"]][df.iloc[i]["AT"]][0]+=df.iloc[i]["AS"]
        team_GD_GS_GC_dict[df.iloc[i]["Lge"]][df.iloc[i]["AT"]][1]+=df.iloc[i]["HS"]
    return team_GD_GS_GC_dict


#%% Streak and Weighted_Streak

def win_draw_loss(a,b):
    if a>b:
        return 3
    elif a==b:
        return 1
    elif a<b:
        return 0
    else:
        print("error_1")

def Streak_and_Weighted_Streak_train(dataframe,hyperparameter_k=6):
    df1=dataframe
    Streak_HT=[np.nan]*len(df1)
    Streak_AT=[np.nan]*len(df1)
    Weighted_Streak_HT=[np.nan]*len(df1)
    Weighted_Streak_AT=[np.nan]*len(df1)
    Streak_and_Weighted_Streak_dict={}
    for l in df1.Lge.unique(): #loop for league
    #for l in ["09-10"]:
        df2=df1[df1["Lge"]==l]     #league df
        #count=0
        Streak_and_Weighted_Streak_dict_per_league={}
        for i in df2.Sea.unique(): #loop for season
        #for i in ["RUS2"]:
            #count+=1
            temp_df=df2[df2["Sea"]==i] #season df
            #print(int(count/len(df2.Lge.unique())*100),"%")

            for j in np.unique(temp_df[['HT', 'AT']].values): #loop for team
            #for j in ["FC Volgar Astrakhan"]:
                if i == df2.Sea.unique()[-1]:
                    Streak_and_Weighted_Streak_dict_per_league[j]=[]
                temp_df_1=temp_df[(temp_df["HT"]==j)|(temp_df["AT"]==j)] 
#team df
                for k in range(hyperparameter_k,len(temp_df_1)): #loop for each row starting from row hyperparameter_k+1
                    temp_Streak=0
                    temp_Weighted_Streak=0
                    for m in reversed(range(1,hyperparameter_k+1)):  #from t-hyperparameter k to t-1
                        temp_df_2=temp_df_1.iloc[k-m]
                        if temp_df_2.HT==j:
                            temp_Streak+=win_draw_loss(temp_df_2["HS"],temp_df_2["AS"])
                            temp_Weighted_Streak+=(hyperparameter_k-m+1)*win_draw_loss(temp_df_2["HS"],temp_df_2["AS"])
                            if k== len(temp_df_1)-1 and i == df2.Sea.unique()[-1]: #if last row of the df and last season
                                Streak_and_Weighted_Streak_dict_per_league[j].append(win_draw_loss(temp_df_2["HS"],temp_df_2["AS"]))
                                #print(j,Streak_and_Weighted_Streak_dict_per_league[j])
                        elif temp_df_2.AT==j:
                            temp_Streak+=win_draw_loss(temp_df_2["AS"],temp_df_2["HS"])
                            temp_Weighted_Streak+=(hyperparameter_k-m+1)*win_draw_loss(temp_df_2["AS"],temp_df_2["HS"])
                            if k== len(temp_df_1)-1 and i == df2.Sea.unique()[-1]:
                                Streak_and_Weighted_Streak_dict_per_league[j].append(win_draw_loss(temp_df_2["HS"],temp_df_2["AS"]))
                                #print(j,Streak_and_Weighted_Streak_dict_per_league[j])
                    temp_Streak=temp_Streak/(3*hyperparameter_k)
                    temp_Weighted_Streak=(2*temp_Weighted_Streak)/(3*hyperparameter_k*(hyperparameter_k+1))
                    if temp_df_1.iloc[k]["HT"]==j: 
                        Streak_HT[temp_df_1.iloc[k]["index"]]=temp_Streak
                        Weighted_Streak_HT[temp_df_1.iloc[k]["index"]]=temp_Weighted_Streak
                    elif temp_df_1.iloc[k]["AT"]==j:
                        Streak_AT[temp_df_1.iloc[k]["index"]]=temp_Streak
                        Weighted_Streak_AT[temp_df_1.iloc[k]["index"]]=temp_Weighted_Streak
        
        Streak_and_Weighted_Streak_dict[l]=Streak_and_Weighted_Streak_dict_per_league
    #streak
    df1["Streak_HT"]=Streak_HT
    df1["Weighted_Streak_HT"]=Weighted_Streak_HT
    df1["Streak_AT"]=Streak_AT
    df1["Weighted_Streak_AT"]=Weighted_Streak_AT
    return df1,Streak_and_Weighted_Streak_dict

def Streak_and_Weighted_Streak_valid(dataframe,Streak_and_Weighted_Streak_dict,hyperparameter_k=6):
    df=dataframe
    df["Streak_HT"]=np.nan
    df["Weighted_Streak_HT"]=np.nan
    df["Streak_AT"]=np.nan
    df["Weighted_Streak_AT"]=np.nan
    for i in range(len(df)): #loop the prediction set
        #print(i,df.iloc[i].Lge,df.iloc[i].HT,df.iloc[i].AT)
        temp_Streak_HT=0
        temp_Weighted_Streak_HT=0
        temp_Streak_AT=0
        temp_Weighted_Streak_AT=0
        for m in reversed(range(1,hyperparameter_k+1)):
            #import pdb; pdb.set_trace()
            temp_Streak_HT+=Streak_and_Weighted_Streak_dict[df.iloc[i]["Lge"]][df.iloc[i]["HT"]][m-1]
            temp_Weighted_Streak_HT+=(hyperparameter_k-m+1)*Streak_and_Weighted_Streak_dict[df.iloc[i]["Lge"]][df.iloc[i]["HT"]][m-1]
            temp_Streak_AT+=Streak_and_Weighted_Streak_dict[df.iloc[i]["Lge"]][df.iloc[i]["AT"]][m-1]
            temp_Weighted_Streak_AT+=(hyperparameter_k-m+1)*Streak_and_Weighted_Streak_dict[df.iloc[i]["Lge"]][df.iloc[i]["AT"]][m-1]
        temp_Streak_HT=temp_Streak_HT/(3*hyperparameter_k)
        temp_Streak_AT=temp_Streak_AT/(3*hyperparameter_k)
        temp_Weighted_Streak_HT=(2*temp_Weighted_Streak_HT)/(3*hyperparameter_k*(hyperparameter_k+1))
        temp_Weighted_Streak_AT=(2*temp_Weighted_Streak_AT)/(3*hyperparameter_k*(hyperparameter_k+1))
        df.iloc[i]["Streak_HT"]=temp_Streak_HT
        df.iloc[i]["Weighted_Streak_HT"]=temp_Weighted_Streak_HT
        df.iloc[i]["Streak_AT"]=temp_Streak_AT
        df.iloc[i]["Weighted_Streak_AT"]=temp_Weighted_Streak_AT
    return df

def Streak_and_Weighted_Streak_update(dataframe,Streak_and_Weighted_Streak_dict,hyperparameter_k=6):
    df=dataframe
    for i in range(len(df)):    
        Streak_and_Weighted_Streak_dict[df.iloc[i]["Lge"]][df.iloc[i]["HT"]].append(win_draw_loss(df.iloc[i]["HS"],df.iloc[i]["AS"]))
        Streak_and_Weighted_Streak_dict[df.iloc[i]["Lge"]][df.iloc[i]["HT"]].pop(0)
        Streak_and_Weighted_Streak_dict[df.iloc[i]["Lge"]][df.iloc[i]["AT"]].append(win_draw_loss(df.iloc[i]["AS"],df.iloc[i]["HS"]))
        Streak_and_Weighted_Streak_dict[df.iloc[i]["Lge"]][df.iloc[i]["AT"]].pop(0)
    return Streak_and_Weighted_Streak_dict

#%% Form 
def Form_train(dataframe,initialize_form=1,gamma=0.33):
    df1=dataframe
    pre_match_Form_HT=[np.nan]*len(df1)
    pre_match_Form_AT=[np.nan]*len(df1)
    pos_match_Form_HT=[np.nan]*len(df1)
    pos_match_Form_AT=[np.nan]*len(df1)
    Form_dict={}
    for m in df1.Lge.unique():
        df2=df1[df1["Lge"]==m]     
        for i in df2.Sea.unique():
            temp_df=df2[df2["Sea"]==i]
            temp_dict = {}
            for j in np.unique(temp_df[['HT', 'AT']].values):
                temp_dict[j] = initialize_form
            for k in range(len(temp_df)):
                temp_df_2=temp_df.iloc[k]
                
                pre_match_Form_HT[temp_df_2["index"]]=temp_dict[temp_df_2.HT]
                pre_match_Form_AT[temp_df_2["index"]]=temp_dict[temp_df_2.AT]
                
                if temp_df_2.WDL=="D":
                    temp_HT_form=temp_dict[temp_df_2.HT]-gamma*(temp_dict[temp_df_2.HT]-temp_dict[temp_df_2.AT])
                    temp_AT_form=temp_dict[temp_df_2.AT]-gamma*(temp_dict[temp_df_2.AT]-temp_dict[temp_df_2.HT])
                    temp_dict[temp_df_2.HT]=temp_HT_form
                    temp_dict[temp_df_2.AT]=temp_AT_form
                elif temp_df_2.WDL=="W": #HT win
                    temp_dict[temp_df_2.HT]=temp_dict[temp_df_2.HT]+gamma*temp_dict[temp_df_2.AT]
                    temp_dict[temp_df_2.AT]=temp_dict[temp_df_2.AT]-gamma*temp_dict[temp_df_2.AT]
                elif temp_df_2.WDL=="L": #HT loss
                    temp_dict[temp_df_2.AT]=temp_dict[temp_df_2.AT]+gamma*temp_dict[temp_df_2.HT]
                    temp_dict[temp_df_2.HT]=temp_dict[temp_df_2.HT]-gamma*temp_dict[temp_df_2.HT]
    
                pos_match_Form_HT[temp_df_2["index"]]=temp_dict[temp_df_2.HT]
                pos_match_Form_AT[temp_df_2["index"]]=temp_dict[temp_df_2.AT]
            
            if i == df2.Sea.unique()[-1]:
                Form_dict[m]=temp_dict
            
    df1["pre_match_Form_HT"] = pre_match_Form_HT
    df1["pre_match_Form_AT"] = pre_match_Form_AT
    #df1["pos_match_Form_HT"] = pos_match_Form_HT
    #df1["pos_match_Form_AT"] = pos_match_Form_AT
    return df1, Form_dict

def Form_valid(dataframe,Form_dict,initialize_form=1,gamma=0.33):
    df1=dataframe
    df1["pre_match_Form_HT"] = np.nan
    df1["pre_match_Form_AT"] = np.nan
    #df1["pos_match_Form_HT"] = np.nan
    #df1["pos_match_Form_AT"] = np.nan
    for i in range(len(df1)):
        df1.iloc[i]["pre_match_Form_HT"]=Form_dict[df1.iloc[i]["Lge"]][df1.iloc[i]["HT"]]
        df1.iloc[i]["pre_match_Form_AT"]=Form_dict[df1.iloc[i]["Lge"]][df1.iloc[i]["AT"]]
    return df1

def Form_update(dataframe,Form_dict,initialize_form=1,gamma=0.33):
    df1=dataframe
    for i in range(len(df1)):
        if df1.iloc[i].WDL=="D":
            temp_HT_form=Form_dict[df1.iloc[i]["Lge"]][df1.iloc[i]["HT"]]-gamma*(Form_dict[df1.iloc[i]["Lge"]][df1.iloc[i]["HT"]]-Form_dict[df1.iloc[i]["Lge"]][df1.iloc[i]["AT"]])
            temp_AT_form=Form_dict[df1.iloc[i]["Lge"]][df1.iloc[i]["AT"]]-gamma*(Form_dict[df1.iloc[i]["Lge"]][df1.iloc[i]["AT"]]-Form_dict[df1.iloc[i]["Lge"]][df1.iloc[i]["HT"]])
            Form_dict[df1.iloc[i]["Lge"]][df1.iloc[i]["HT"]]=temp_HT_form
            Form_dict[df1.iloc[i]["Lge"]][df1.iloc[i]["AT"]]=temp_AT_form
        elif df1.iloc[i].WDL=="W": #HT win
            Form_dict[df1.iloc[i]["Lge"]][df1.iloc[i]["HT"]]=Form_dict[df1.iloc[i]["Lge"]][df1.iloc[i]["HT"]]+gamma*Form_dict[df1.iloc[i]["Lge"]][df1.iloc[i]["AT"]]
            Form_dict[df1.iloc[i]["Lge"]][df1.iloc[i]["AT"]]=Form_dict[df1.iloc[i]["Lge"]][df1.iloc[i]["AT"]]-gamma*Form_dict[df1.iloc[i]["Lge"]][df1.iloc[i]["AT"]]
        elif df1.iloc[i].WDL=="L": #HT loss
            Form_dict[df1.iloc[i]["Lge"]][df1.iloc[i]["HT"]]=Form_dict[df1.iloc[i]["Lge"]][df1.iloc[i]["AT"]]+gamma*Form_dict[df1.iloc[i]["Lge"]][df1.iloc[i]["HT"]]
            Form_dict[df1.iloc[i]["Lge"]][df1.iloc[i]["AT"]]=Form_dict[df1.iloc[i]["Lge"]][df1.iloc[i]["HT"]]-gamma*Form_dict[df1.iloc[i]["Lge"]][df1.iloc[i]["HT"]]
    return Form_dict

#%% Attacking strength feature,Defensive strength feature,strength of opposition, home advantage
#continuous-season and super-league data integration approach
def Berrar_Recencyfeature_train(dataframe,n=9):
    df1=dataframe
    time_dict_attacking_stength_HT={} #create dict for each features
    time_dict_attacking_stength_AT={}
    time_dict_defensive_stength_HT={}
    time_dict_defensive_stength_AT={}
    time_dict_strength_opposition_HT={}
    time_dict_strength_opposition_AT={}
    time_dict_home_advantage_HT={}
    time_dict_home_advantage_AT={}
    Berrar_Recencyfeature_team_dict={}
    
    
    for num in range(n): #sub_dict for each time n
        time_dict_attacking_stength_HT[num+1]=[np.nan]*len(df1)
        time_dict_attacking_stength_AT[num+1]=[np.nan]*len(df1)
        time_dict_defensive_stength_HT[num+1]=[np.nan]*len(df1)
        time_dict_defensive_stength_AT[num+1]=[np.nan]*len(df1)
        time_dict_strength_opposition_HT[num+1]=[np.nan]*len(df1)
        time_dict_strength_opposition_AT[num+1]=[np.nan]*len(df1)
        time_dict_home_advantage_HT[num+1]=[np.nan]*len(df1)
        time_dict_home_advantage_AT[num+1]=[np.nan]*len(df1)
    
    
    df2=df1[:] 
    #loop for (super) leagues
    for i in ['ARG1', 'AUS1', 'BEL1', 'CHE1', 'CHL1', 'DZA1', ['ENG1', 'ENG2','ENG3', 'ENG4', 'ENG5'], ['FRA1', 'FRA2', 'FRA3'], ['GER1', 'GER2','GER3'], 'HOL1', ['ITA1', 'ITA2'], 'MAR1', 'MEX1', 'POR1',['RUS1', 'RUS2'], ['SCO1', 'SCO2', 'SCO3', 'SCO4'], ['SPA1', 'SPA2'],'SWE1', 'USA1', 'ZAF1']:
        #print(i)
        if type(i)==str:
            temp_df=df2[df2["Lge"]==i]
        else:
            temp_df=df2[df2["Lge"].isin(i)]
        team_list=np.unique(temp_df[['HT', 'AT']].values)
        team_info={}
        goal_scored=[np.nan]*n
        goal_conceded=[np.nan]*n
        home_advantage=[np.nan]*n
        strength_opp=[np.nan]*n
        for k in team_list:
            team_info[k] = [0,goal_scored.copy(),goal_conceded.copy(),home_advantage.copy(),strength_opp.copy()] #team appearance, goal scored, goal conceded, home advantage, strength_opposition
        for j in range(len(temp_df)):
            temp_df2=temp_df.iloc[j] #loop the df
            if team_info[temp_df2.HT][0]<=n or team_info[temp_df2.AT][0]<=n or np.nan in team_info[temp_df2.HT][4] or np.nan in team_info[temp_df2.HT][4] or np.nan in team_info[temp_df2.AT][4] or np.nan in team_info[temp_df2.AT][4]:
                if np.nan not in team_info[temp_df2.AT][1] and np.nan not in team_info[temp_df2.AT][2]:
                    team_info[temp_df2.HT][4].insert(0,sum([x - y for x, y in zip(team_info[temp_df2.AT][1], team_info[temp_df2.AT][2])])/n)
                    team_info[temp_df2.HT][4].pop(-1)
                if np.nan not in team_info[temp_df2.HT][1] and np.nan not in team_info[temp_df2.HT][2]:
                    team_info[temp_df2.AT][4].insert(0,sum([x - y for x, y in zip(team_info[temp_df2.HT][1], team_info[temp_df2.HT][2])])/n)
                    team_info[temp_df2.AT][4].pop(-1)
                #import pdb; pdb.set_trace()
                #print(temp_df2.HT,team_info[temp_df2.HT])
                team_info[temp_df2.HT][1].insert(0,temp_df2.HS)
                team_info[temp_df2.HT][1].pop(-1)
                team_info[temp_df2.HT][2].insert(0,temp_df2.AS)
                team_info[temp_df2.HT][2].pop(-1)
                team_info[temp_df2.HT][3].insert(0,1)
                team_info[temp_df2.HT][3].pop(-1)
                team_info[temp_df2.AT][1].insert(0,temp_df2.AS)
                team_info[temp_df2.AT][1].pop(-1)
                team_info[temp_df2.AT][2].insert(0,temp_df2.HS)
                team_info[temp_df2.AT][2].pop(-1)
                team_info[temp_df2.AT][3].insert(0,-1)
                team_info[temp_df2.AT][3].pop(-1)
                team_info[temp_df2.HT][0]+=1
                team_info[temp_df2.AT][0]+=1
            else:
                if np.nan in [team_info[temp_df2.HT][1],team_info[temp_df2.HT][2],team_info[temp_df2.HT][3]]:
                    print("error")
                for l in range(n):
                    time_dict_attacking_stength_HT[l+1][temp_df2["index"]]=team_info[temp_df2.HT][1][l]
                    time_dict_defensive_stength_HT[l+1][temp_df2["index"]]=team_info[temp_df2.HT][2][l]
                    time_dict_home_advantage_HT[l+1][temp_df2["index"]]=team_info[temp_df2.HT][3][l]
                    time_dict_attacking_stength_AT[l+1][temp_df2["index"]]=team_info[temp_df2.AT][1][l]
                    time_dict_defensive_stength_AT[l+1][temp_df2["index"]]=team_info[temp_df2.AT][2][l]
                    time_dict_home_advantage_AT[l+1][temp_df2["index"]]=team_info[temp_df2.AT][3][l]
                    time_dict_strength_opposition_HT[l+1][temp_df2["index"]]=team_info[temp_df2.AT][4][l]
                    time_dict_strength_opposition_AT[l+1][temp_df2["index"]]=team_info[temp_df2.AT][4][l]
  
                team_info[temp_df2.HT][4].insert(0,sum([x - y for x, y in zip(team_info[temp_df2.AT][1], team_info[temp_df2.AT][2])])/n)
                team_info[temp_df2.HT][4].pop(-1)
                team_info[temp_df2.AT][4].insert(0,sum([x - y for x, y in zip(team_info[temp_df2.HT][1], team_info[temp_df2.HT][2])])/n)
                team_info[temp_df2.AT][4].pop(-1)
                team_info[temp_df2.HT][1].insert(0,temp_df2.HS)
                team_info[temp_df2.HT][1].pop(-1)
                team_info[temp_df2.HT][2].insert(0,temp_df2.AS)
                team_info[temp_df2.HT][2].pop(-1)
                team_info[temp_df2.HT][3].insert(0,1)
                team_info[temp_df2.HT][3].pop(-1)
                team_info[temp_df2.AT][1].insert(0,temp_df2.AS)
                team_info[temp_df2.AT][1].pop(-1)
                team_info[temp_df2.AT][2].insert(0,temp_df2.HS)
                team_info[temp_df2.AT][2].pop(-1)
                team_info[temp_df2.AT][3].insert(0,-1)
                team_info[temp_df2.AT][3].pop(-1)
                team_info[temp_df2.HT][0]+=1
                team_info[temp_df2.AT][0]+=1
                #print(temp_df2.HT,team_info[temp_df2.HT])
     
        if type(i)==str:
            Berrar_Recencyfeature_team_dict[i]=team_info 
        else:
            for m in i:    
                Berrar_Recencyfeature_team_dict[m]=team_info                                                                                             
    #Attacking strength feature,Defensive strength feature,strength of opposition, home advantage
    for i in range(n):
        df1[f"attacking_stength_{i+1}_HT"]=time_dict_attacking_stength_HT[i+1]
        df1[f"defensive_stength_{i+1}_HT"]=time_dict_defensive_stength_HT[i+1]
        df1[f"strength_opposition_{i+1}_HT"]=time_dict_strength_opposition_HT[i+1]
        df1[f"home_advantage_{i+1}_HT"]=time_dict_home_advantage_HT[i+1]
        df1[f"attacking_stength_{i+1}_AT"]=time_dict_attacking_stength_AT[i+1]
        df1[f"defensive_stength_{i+1}_AT"]=time_dict_defensive_stength_AT[i+1]
        df1[f"strength_opposition_{i+1}_AT"]=time_dict_strength_opposition_AT[i+1]
        df1[f"home_advantage_{i+1}_AT"]=time_dict_home_advantage_AT[i+1]
    return df1,Berrar_Recencyfeature_team_dict
  
def Berrar_Recencyfeature_valid(dataframe,Berrar_Recencyfeature_team_dict,n=9):
    df1=dataframe
    time_dict_attacking_stength_HT={} #create dict for each features
    time_dict_attacking_stength_AT={}
    time_dict_defensive_stength_HT={}
    time_dict_defensive_stength_AT={}
    time_dict_strength_opposition_HT={}
    time_dict_strength_opposition_AT={}
    time_dict_home_advantage_HT={}
    time_dict_home_advantage_AT={}
    for num in range(n): #sub_dict for each time n
        time_dict_attacking_stength_HT[num+1]=[np.nan]*len(df1)
        time_dict_attacking_stength_AT[num+1]=[np.nan]*len(df1)
        time_dict_defensive_stength_HT[num+1]=[np.nan]*len(df1)
        time_dict_defensive_stength_AT[num+1]=[np.nan]*len(df1)
        time_dict_strength_opposition_HT[num+1]=[np.nan]*len(df1)
        time_dict_strength_opposition_AT[num+1]=[np.nan]*len(df1)
        time_dict_home_advantage_HT[num+1]=[np.nan]*len(df1)
        time_dict_home_advantage_AT[num+1]=[np.nan]*len(df1)
    for j in range(len(df1)):   
         for l in range(n): #league, team, metrics, number
             #record the features in the valid set
             #import pdb;pdb.set_trace()
             time_dict_attacking_stength_HT[l+1][j]=Berrar_Recencyfeature_team_dict[df1.iloc[j]["Lge"]][df1.iloc[j]["HT"]][1][l]
             time_dict_defensive_stength_HT[l+1][j]=Berrar_Recencyfeature_team_dict[df1.iloc[j]["Lge"]][df1.iloc[j]["HT"]][2][l]
             time_dict_strength_opposition_HT[l+1][j]=Berrar_Recencyfeature_team_dict[df1.iloc[j]["Lge"]][df1.iloc[j]["HT"]][4][l]
             time_dict_home_advantage_HT[l+1][j]=Berrar_Recencyfeature_team_dict[df1.iloc[j]["Lge"]][df1.iloc[j]["HT"]][3][l]
             time_dict_attacking_stength_AT[l+1][j]=Berrar_Recencyfeature_team_dict[df1.iloc[j]["Lge"]][df1.iloc[j]["AT"]][1][l]
             time_dict_defensive_stength_AT[l+1][j]=Berrar_Recencyfeature_team_dict[df1.iloc[j]["Lge"]][df1.iloc[j]["AT"]][2][l]
             time_dict_strength_opposition_AT[l+1][j]=Berrar_Recencyfeature_team_dict[df1.iloc[j]["Lge"]][df1.iloc[j]["AT"]][4][l]
             time_dict_home_advantage_AT[l+1][j]=Berrar_Recencyfeature_team_dict[df1.iloc[j]["Lge"]][df1.iloc[j]["AT"]][3][l]
             
          #update based on the validset    
    for i in range(n):
        df1[f"attacking_stength_{i+1}_HT"]=time_dict_attacking_stength_HT[i+1]
        df1[f"defensive_stength_{i+1}_HT"]=time_dict_defensive_stength_HT[i+1]
        df1[f"strength_opposition_{i+1}_HT"]=time_dict_strength_opposition_HT[i+1]
        df1[f"home_advantage_{i+1}_HT"]=time_dict_home_advantage_HT[i+1]
        df1[f"attacking_stength_{i+1}_AT"]=time_dict_attacking_stength_AT[i+1]
        df1[f"defensive_stength_{i+1}_AT"]=time_dict_defensive_stength_AT[i+1]
        df1[f"strength_opposition_{i+1}_AT"]=time_dict_strength_opposition_AT[i+1]
        df1[f"home_advantage_{i+1}_AT"]=time_dict_home_advantage_AT[i+1]      
    return df1

def Berrar_Recencyfeature_update(dataframe,Berrar_Recencyfeature_team_dict,n=9):
    df1=dataframe
    for j in range(len(df1)): 
        if df1.iloc[j]["Lge"] in ['ENG1', 'ENG2','ENG3', 'ENG4', 'ENG5'] :
            league_list=['ENG1', 'ENG2','ENG3', 'ENG4', 'ENG5']
        elif df1.iloc[j]["Lge"] in  ['FRA1', 'FRA2', 'FRA3']:
            league_list=['FRA1', 'FRA2', 'FRA3']
        elif df1.iloc[j]["Lge"] in ['GER1', 'GER2','GER3']:
            league_list=['GER1', 'GER2','GER3']
        elif df1.iloc[j]["Lge"] in ['ITA1', 'ITA2']:
            league_list=['ITA1', 'ITA2']
        elif df1.iloc[j]["Lge"] in ['RUS1', 'RUS2']:
            league_list=['RUS1', 'RUS2']
        elif df1.iloc[j]["Lge"] in ['SCO1', 'SCO2', 'SCO3', 'SCO4']:
            league_list=['SCO1', 'SCO2', 'SCO3', 'SCO4']
        elif df1.iloc[j]["Lge"] in ['SPA1', 'SPA2']:
            league_list=['SPA1', 'SPA2']
        else:
            league_list=[df1.iloc[j]["Lge"]]
        for k in league_list:
            #opp strength
            Berrar_Recencyfeature_team_dict[k][df1.iloc[j]["HT"]][4].insert(0,sum([x - y for x, y in zip(Berrar_Recencyfeature_team_dict[df1.iloc[j]["Lge"]][df1.iloc[j]["AT"]][1], Berrar_Recencyfeature_team_dict[df1.iloc[j]["Lge"]][df1.iloc[j]["AT"]][2])])/n)
            Berrar_Recencyfeature_team_dict[k][df1.iloc[j]["HT"]][4].pop(-1)
            Berrar_Recencyfeature_team_dict[k][df1.iloc[j]["AT"]][4].insert(0,sum([x - y for x, y in zip(Berrar_Recencyfeature_team_dict[df1.iloc[j]["Lge"]][df1.iloc[j]["HT"]][1], Berrar_Recencyfeature_team_dict[df1.iloc[j]["Lge"]][df1.iloc[j]["HT"]][2])])/n)
            Berrar_Recencyfeature_team_dict[k][df1.iloc[j]["AT"]][4].pop(-1)
            #attack
            Berrar_Recencyfeature_team_dict[k][df1.iloc[j]["HT"]][1].insert(0,df1.iloc[j].HS)
            Berrar_Recencyfeature_team_dict[k][df1.iloc[j]["HT"]][1].pop(-1)
            Berrar_Recencyfeature_team_dict[k][df1.iloc[j]["AT"]][1].insert(0,df1.iloc[j].AS)
            Berrar_Recencyfeature_team_dict[k][df1.iloc[j]["AT"]][1].pop(-1)
            #home advantage
            Berrar_Recencyfeature_team_dict[k][df1.iloc[j]["HT"]][3].insert(0,1)
            Berrar_Recencyfeature_team_dict[k][df1.iloc[j]["HT"]][3].pop(-1)
            Berrar_Recencyfeature_team_dict[k][df1.iloc[j]["AT"]][3].insert(0,-1)
            Berrar_Recencyfeature_team_dict[k][df1.iloc[j]["AT"]][3].pop(-1)
            #defense
            Berrar_Recencyfeature_team_dict[k][df1.iloc[j]["HT"]][2].insert(0,df1.iloc[j].AS)
            Berrar_Recencyfeature_team_dict[k][df1.iloc[j]["HT"]][2].pop(-1)
            Berrar_Recencyfeature_team_dict[k][df1.iloc[j]["AT"]][2].insert(0,df1.iloc[j].HS)
            Berrar_Recencyfeature_team_dict[k][df1.iloc[j]["AT"]][2].pop(-1)
            #appearence
            Berrar_Recencyfeature_team_dict[k][df1.iloc[j]["HT"]][0]+=1
            Berrar_Recencyfeature_team_dict[k][df1.iloc[j]["HT"]][0]+=1 
    
    return Berrar_Recencyfeature_team_dict

#%% Round

def Round_train(dataframe):
    df1=dataframe
    Round=[np.nan]*len(df1)
    
    for l in df1.Lge.unique(): #loop for each league
        df2=df1[df1["Lge"]==l]    #get league df
        for i in df2.Sea.unique(): #loop for each season
            temp_df=df2[df2["Sea"]==i] #get season df
            for j in np.unique(temp_df[['HT', 'AT']].values): #loop for team in the league
                temp_df_1=temp_df[(temp_df["HT"]==j)|(temp_df["AT"]==j)] #get team df
                num=0
                for k in temp_df_1.index:
                    num+=1
                    Round[k]=num
    df1["Round"]=Round
    return df1

def Round_valid(dataframe1,dataframe2): #dataframe 1=previous matches,dataframe 2= prediction target
    df1=dataframe1
    df2=dataframe2
    df_merged = df1.append(df2, ignore_index=True)
    df_merged=Round_train(df_merged)
    df_merged=df_merged[-len(df2):]
    return df_merged

#%% Match important


def Match_important_train(dataframe,top=5,bottom=5):

    df1=dataframe
    if 'Round' not in df1.columns:
        print("Please create ronund features first")
        return
    league_table={} #[League][Season][Team]
    temp_value={}
    for i in range(top):
        temp_value[f"L_up_{i}_HT"]=[np.nan]*len(df1)
        temp_value[f"L_up_{i}_AT"]=[np.nan]*len(df1)
        df1[f"L_up_{i}_HT"]=np.nan#point diff between the team and top i team 
        df1[f"L_up_{i}_AT"]=np.nan
    for j in range(bottom): 
        temp_value[f"L_down_{j}_HT"]=[np.nan]*len(df1)
        temp_value[f"L_down_{j}_AT"]=[np.nan]*len(df1)
        df1[f"L_down_{j}_HT"]=np.nan#point diff between the team and bottom i team 
        df1[f"L_down_{j}_AT"]=np.nan
        
    for k in df1.Lge.unique(): #loop for league
        league_table[k]={} #sub_dict for each league
        df2=df1[df1["Lge"]==k] #get leage df
        for m in df2.Sea.unique(): #loop for season
            league_table[k][m]={} #sub_dict for each season
            df3=df2[df2["Sea"]==m] #get season df
            team_list=np.unique(df3[['HT', 'AT']].values) 
            for o in team_list:
                league_table[k][m][o]=0 #initial value for each team
            for n in range(len(df3)): 
                Current_Round=df3.iloc[n]["Round"]
                Preivous_Round= df3.iloc[n-1]["Round"]
                if Current_Round!=Preivous_Round:
                    lock_league_table=league_table #update for next round of match 
                #get the features value
                for p in range(top):
                    temp_value[f"L_up_{p}_HT"][df3.iloc[n]["index"]]= sorted(lock_league_table[k][m].values())[-(p+1)]-lock_league_table[k][m][df3.iloc[n].HT]
                    temp_value[f"L_up_{p}_AT"][df3.iloc[n]["index"]]= sorted(lock_league_table[k][m].values())[-(p+1)]-lock_league_table[k][m][df3.iloc[n].AT]
                for q in range(bottom):
                    temp_value[f"L_down_{q}_HT"][df3.iloc[n]["index"]]= sorted(lock_league_table[k][m].values())[q]-lock_league_table[k][m][df3.iloc[n].HT]
                    temp_value[f"L_down_{q}_AT"][df3.iloc[n]["index"]]= sorted(lock_league_table[k][m].values())[q]-lock_league_table[k][m][df3.iloc[n].AT]
                #update the value after each match 
                
                league_table[k][m][df3.iloc[n].HT]+= win_draw_loss(df3.iloc[n].HS,df3.iloc[n].AS)
                league_table[k][m][df3.iloc[n].AT]+= win_draw_loss(df3.iloc[n].AS,df3.iloc[n].HS)
    
    for r in range(top):
        df1[f"L_up_{r}_HT"]= temp_value[f"L_up_{r}_HT"]
        df1[f"L_up_{r}_AT"]= temp_value[f"L_up_{r}_AT"]
    for s in range(bottom): 
        df1[f"L_down_{s}_HT"]=temp_value[f"L_down_{s}_HT"]
        df1[f"L_down_{s}_AT"]=temp_value[f"L_down_{s}_AT"] 
    return df1,league_table

#please update for each round     
def Match_important_valid(dataframe,league_table,top=5,bottom=5):
    print("please update for each round")
    df1=dataframe
    for i in range(top):
        df1[f"L_up_{i}_HT"]=np.nan#point diff between the team and top i team 
        df1[f"L_up_{i}_AT"]=np.nan
    for j in range(bottom): 
        df1[f"L_down_{j}_HT"]=np.nan#point diff between the team and bottom i team 
        df1[f"L_down_{j}_AT"]=np.nan    
    for n in range(len(df1)): 
        #get the features value
        for p in range(top):
            df1.iloc[n][f"L_up_{p}_HT"]= sorted(league_table[df1.iloc[n]["Lge"]][df1.iloc[n]["Sea"]].values())[-(p+1)]-league_table[df1.iloc[n]["Lge"]][df1.iloc[n]["Sea"]][df1.iloc[n].HT]
            df1.iloc[n][f"L_up_{p}_AT"]= sorted(league_table[df1.iloc[n]["Lge"]][df1.iloc[n]["Sea"]].values())[-(p+1)]-league_table[df1.iloc[n]["Lge"]][df1.iloc[n]["Sea"]][df1.iloc[n].AT]
        for q in range(bottom):
            df1.iloc[n][f"L_down_{q}_HT"]= sorted(league_table[df1.iloc[n]["Lge"]][df1.iloc[n]["Sea"]].values())[q]-league_table[df1.iloc[n]["Lge"]][df1.iloc[n]["Sea"]][df1.iloc[n].HT]
            df1.iloc[n][f"L_down_{q}_AT"]= sorted(league_table[df1.iloc[n]["Lge"]][df1.iloc[n]["Sea"]].values())[q]-league_table[df1.iloc[n]["Lge"]][df1.iloc[n]["Sea"]][df1.iloc[n].AT]

    return df1

#please update for each round     
def Match_important_update(dataframe,league_table,top=5,bottom=5):
    print("please update for each round")
    df1=dataframe
    for n in range(len(df1)): 
        #update the value after the match 
        league_table[df1.iloc[n]["Lge"]][df1.iloc[n]["Sea"]][df1.iloc[n].HT]+= win_draw_loss(df1.iloc[n].HS,df1.iloc[n].AS)
        league_table[df1.iloc[n]["Lge"]][df1.iloc[n]["Sea"]][df1.iloc[n].AT]+= win_draw_loss(df1.iloc[n].AS,df1.iloc[n].HS)
    return league_table
#%% newly promoted and demotedtable 1

#get the list of team in the previous season
#check if the team in current season in the previous season or lower and upper league



def newly_promoted_domoted_train(dataframe,team_list={}):
    season=[' 00-01', ' 01-02', ' 02-03', ' 03-04', ' 04-05', ' 05-06', ' 06-07',
     ' 07-08', ' 08-09', ' 09-10', ' 10-11', ' 11-12', ' 12-13', ' 13-14',
     ' 14-15', ' 15-16', ' 16-17', ' 17-18', ' 18-19', ' 19-20', ' 20-21',
     ' 21-22', ' 22-23', ' 23-24']
    df1=dataframe
    league_starting_season={}
    df1["newly_promoted_HT"]=np.nan
    df1["newly_demoted_HT"]=np.nan
    df1["newly_promoted_AT"]=np.nan
    df1["newly_demoted_AT"]=np.nan
    newly_promoted_HT=[np.nan]*len(df1)
    newly_demoted_HT=[np.nan]*len(df1)
    newly_promoted_AT=[np.nan]*len(df1)
    newly_demoted_AT=[np.nan]*len(df1)
    if team_list=={}:
        for i in df1.Lge.unique():
            team_list[i]={} #sub dict on league
            df2=df1[df1["Lge"]==i]
            league_starting_season[i]=df2.Sea.unique()[0]
            for j in df2.Sea.unique():
                df3=df2[df2["Sea"]==j]
                team_list[i][j]=np.unique(df3[['HT', 'AT']].values).tolist() #sub dict on season
    #import pdb; pdb.set_trace()
    for k in range(len(df1)):
        if df1.iloc[k].Sea==league_starting_season[df1.iloc[k].Lge]: #skip the first season of each league
            continue
        if df1.iloc[k].Sea==" 05-06" and df1.iloc[k].Lge=="AUS1":
            continue
        if df1.iloc[k].Sea==" 12-13" and df1.iloc[k].Lge=="SCO4": #skip for now
            continue
        if df1.iloc[k].Sea==" 18-19" and df1.iloc[k].Lge=="CHL1": #skip for now
            continue
        if df1.iloc[k].Sea==" 23-24" and df1.iloc[k].Lge=="MEX1": #skip for now
            continue
        if df1.iloc[k]["Lge"] in ['ENG2','ENG3', 'ENG4', 'ENG5'] : #check if only one league or the top league in the country
            league_list=['ENG1', 'ENG2','ENG3', 'ENG4', 'ENG5']
        elif df1.iloc[k]["Lge"] in  ['FRA2', 'FRA3']:
            league_list=['FRA1', 'FRA2', 'FRA3']
        elif df1.iloc[k]["Lge"] in ['GER2','GER3']:
            league_list=['GER1', 'GER2','GER3']
        elif df1.iloc[k]["Lge"] in ['ITA2']:
            league_list=['ITA1', 'ITA2']
        elif df1.iloc[k]["Lge"] in ['RUS2']:
            league_list=['RUS1', 'RUS2']
        elif df1.iloc[k]["Lge"] in ['SCO2', 'SCO3', 'SCO4']:
            league_list=['SCO1', 'SCO2', 'SCO3', 'SCO4']
        elif df1.iloc[k]["Lge"] in ['SPA2']:
            league_list=['SPA1', 'SPA2']
        else:
            league_list=[df1.iloc[k]["Lge"]]
        
        if len(league_list)==1: 
            #print(k)
            # print(df1.iloc[k].HT)
            #print(df1.iloc[k].Lge)
            # print(df1.iloc[k].Sea)
            # if k==117209:
            #     import pdb; pdb.set_trace()
            # print(team_list[df1.iloc[k].Lge][season[season.index(df1.iloc[k].Sea)-1]])
            
            if df1.iloc[k].HT in team_list[df1.iloc[k].Lge][season[season.index(df1.iloc[k].Sea)-1]]:
                newly_promoted_HT[df1.iloc[k]["index"]]=0
                newly_demoted_HT[df1.iloc[k]["index"]]=0
            else:
                newly_promoted_HT[df1.iloc[k]["index"]]=1
                newly_demoted_HT[df1.iloc[k]["index"]]=0
            if df1.iloc[k].AT in team_list[df1.iloc[k].Lge][season[season.index(df1.iloc[k].Sea)-1]]:
                newly_promoted_AT[df1.iloc[k]["index"]]=0
                newly_demoted_AT[df1.iloc[k]["index"]]=0
            else:
                newly_promoted_AT[df1.iloc[k]["index"]]=1
                newly_demoted_AT[df1.iloc[k]["index"]]=0

        elif df1.iloc[k].Lge == league_list[-1]:
            if df1.iloc[k].HT in team_list[df1.iloc[k].Lge][season[season.index(df1.iloc[k].Sea)-1]]: #check if stay
                newly_promoted_HT[df1.iloc[k]["index"]]=0
                newly_demoted_HT[df1.iloc[k]["index"]]=0
            elif df1.iloc[k].HT in team_list[league_list[-2]][season[season.index(df1.iloc[k].Sea)-1]]: #check if demoted
                newly_promoted_HT[df1.iloc[k]["index"]]=0
                newly_demoted_HT[df1.iloc[k]["index"]]=1
            else:                                                                           #o/w promoted
                newly_promoted_HT[df1.iloc[k]["index"]]=1
                newly_demoted_HT[df1.iloc[k]["index"]]=0
            if df1.iloc[k].AT in team_list[df1.iloc[k].Lge][season[season.index(df1.iloc[k].Sea)-1]]: #check if stay
                newly_promoted_AT[df1.iloc[k]["index"]]=0
                newly_demoted_AT[df1.iloc[k]["index"]]=0
            elif df1.iloc[k].AT in team_list[league_list[-2]][season[season.index(df1.iloc[k].Sea)-1]]: #check if demoted
                newly_promoted_AT[df1.iloc[k]["index"]]=0
                newly_demoted_AT[df1.iloc[k]["index"]]=1
            else:                                                                           #o/w promoted
                newly_promoted_AT[df1.iloc[k]["index"]]=1
                newly_demoted_AT[df1.iloc[k]["index"]]=0
                
        
        else:
            if df1.iloc[k].HT in team_list[df1.iloc[k].Lge][season[season.index(df1.iloc[k].Sea)-1]]: #check if stay
                newly_promoted_HT[df1.iloc[k]["index"]]=0
                newly_demoted_HT[df1.iloc[k]["index"]]=0
            elif df1.iloc[k].HT in team_list[league_list[league_list.index(df1.iloc[k].Lge)-1]][season[season.index(df1.iloc[k].Sea)-1]]: #check if demoted
                newly_promoted_HT[df1.iloc[k]["index"]]=0
                newly_demoted_HT[df1.iloc[k]["index"]]=1
            elif df1.iloc[k].HT in team_list[league_list[league_list.index(df1.iloc[k].Lge)+1]][season[season.index(df1.iloc[k].Sea)-1]]: #o/w promoted
                newly_promoted_HT[df1.iloc[k]["index"]]=1
                newly_demoted_HT[df1.iloc[k]["index"]]=0
            if df1.iloc[k].AT in team_list[df1.iloc[k].Lge][season[season.index(df1.iloc[k].Sea)-1]]: #check if stay
                newly_promoted_AT[df1.iloc[k]["index"]]=0
                newly_demoted_AT[df1.iloc[k]["index"]]=0
            elif df1.iloc[k].AT in team_list[league_list[league_list.index(df1.iloc[k].Lge)-1]][season[season.index(df1.iloc[k].Sea)-1]]: #check if demoted
                newly_promoted_AT[df1.iloc[k]["index"]]=0
                newly_demoted_AT[df1.iloc[k]["index"]]=1
            elif team_list[league_list[league_list.index(df1.iloc[k].Lge)+1]][season[season.index(df1.iloc[k].Sea)-1]]:                                                                           #o/w promoted
                newly_promoted_AT[df1.iloc[k]["index"]]=1
                newly_demoted_AT[df1.iloc[k]["index"]]=0
            
    df1["newly_promoted_HT"]=newly_promoted_HT
    df1["newly_demoted_HT"]=newly_demoted_HT
    df1["newly_promoted_AT"]=newly_promoted_AT
    df1["newly_demoted_AT"]=newly_demoted_AT   
    
    return df1,team_list

def newly_promoted_domoted_valid(dataframe,team_list):
    df1,_=newly_promoted_domoted_train(dataframe,team_list=team_list)
    return df1

#%% Days since previous match table 1

def days_since_previous_train(dataframe):
    from datetime import datetime 
    df1=dataframe
    df1["days_since_previous_HT"]=np.nan
    df1["days_since_previous_AT"]=np.nan
    temp_HT=[np.nan]*len(df1)
    temp_AT=[np.nan]*len(df1)
    for i in df1.Lge.unique():
        df2=df1[df1["Lge"]==i]
        for j in np.unique(df2[['HT', 'AT']].values): #loop for team in the league
            df3=df2[(df2["HT"]==j)|(df2["AT"]==j)] #get team df
            for k in range(1,len(df3)):
                Date=datetime.strptime(df3.iloc[k]["Date"],"%d/%m/%Y")-datetime.strptime(df3.iloc[k-1]["Date"],"%d/%m/%Y")
                if df3.iloc[k]["HT"]==j:
                    temp_HT[df3.iloc[k]["index"]]=Date.days
                elif df3.iloc[k]["AT"]==j:
                    temp_AT[df3.iloc[k]["index"]]=Date.days
    df1["days_since_previous_HT"]=temp_HT
    df1["days_since_previous_AT"]=temp_AT
    return df1

def days_since_previous_valid(dataframe1,dataframe2): #dataframe 1=previous matches,dataframe 2= prediction target
    df1=dataframe1
    df2=dataframe2
    df_merged = df1.append(df2, ignore_index=True)
    df_merged=days_since_previous_train(df_merged)
    df_merged=df_merged[-len(df2):]
    return df_merged
    
    

#%% Form table 1

def Form_tb1_train(dataframe,hyperparameter=3):
    df1=dataframe
    Form_tb1_HT=[np.nan]*len(df1)
    Form_tb1_AT=[np.nan]*len(df1)
    Form_tb1_dict={}
    for l in df1.Lge.unique(): #loop for league
        df2=df1[df1["Lge"]==l]     #league df
        Form_tb1_dict[l]={}
        for i in df2.Sea.unique(): #loop for season
            temp_df=df2[df2["Sea"]==i] #season df
            Form_tb1_dict[l][i]={}
            for j in np.unique(temp_df[['HT', 'AT']].values): #loop for team
                Form_tb1_dict[l][i][j]=[np.nan]*hyperparameter
                temp_df_1=temp_df[(temp_df["HT"]==j)|(temp_df["AT"]==j)] 
                for k in range(len(temp_df_1)):
                    row=temp_df_1.iloc[k]
                    if np.nan not in Form_tb1_dict[l][i][j]: #get value
                        if row.HT==j:
                            Form_tb1_HT[row["index"]]=sum(Form_tb1_dict[l][i][j])/(3*hyperparameter)
                        elif row.AT==j:
                            Form_tb1_AT[row["index"]]=sum(Form_tb1_dict[l][i][j])/(3*hyperparameter)
                    if row.HT == j: #update
                        Form_tb1_dict[l][i][j].insert(0,win_draw_loss(row.HS, row.AS))
                        Form_tb1_dict[l][i][j].pop(-1)
                    elif row.AT == j:
                        Form_tb1_dict[l][i][j].insert(0,win_draw_loss(row.AS, row.HS))
                        Form_tb1_dict[l][i][j].pop(-1)


    df1["Form_tb1_HT"]=Form_tb1_HT
    df1["Form_tb1_AT"]=Form_tb1_AT
    return df1,Form_tb1_dict

def Form_tb1_valid(dataframe,Form_tb1_dict,hyperparameter=3):
    df1=dataframe
    Form_tb1_HT=[np.nan]*len(df1)
    Form_tb1_AT=[np.nan]*len(df1)
    for i in range(len(df1)):
        row=df1.iloc[i]
        Form_tb1_HT[i]=sum(Form_tb1_dict[row.Lge][row.Sea][row.HT])/(3*hyperparameter)
        Form_tb1_AT[i]=sum(Form_tb1_dict[row.Lge][row.Sea][row.AT])/(3*hyperparameter)
    df1["Form_tb1_HT"]=Form_tb1_HT
    df1["Form_tb1_AT"]=Form_tb1_AT
    return df1
def Form_tb1_update(dataframe,Form_tb1_dict):
    df1=dataframe
    for i in range(len(df1)):
        row=df1.iloc[i]
        Form_tb1_dict[row.Lge][row.Sea][row.HT].insert(0,win_draw_loss(row.HS, row.AS))
        Form_tb1_dict[row.Lge][row.Sea][row.HT].pop(-1)
        Form_tb1_dict[row.Lge][row.Sea][row.AT].insert(0,win_draw_loss(row.AS, row.HS))
        Form_tb1_dict[row.Lge][row.Sea][row.AT].pop(-1)
    return Form_tb1_dict

#%% point tally table 1
def point_tally_train(dataframe):
    df1=dataframe
    point_tally_HT=[np.nan]*len(df1)
    point_tally_AT=[np.nan]*len(df1)
    point_tally_dict={}
    for l in df1.Lge.unique(): #loop for league
        df2=df1[df1["Lge"]==l]     #league df
        point_tally_dict[l]={}
        for i in df2.Sea.unique(): #loop for season
            temp_df=df2[df2["Sea"]==i] #season df
            point_tally_dict[l][i]={}
            for j in np.unique(temp_df[['HT', 'AT']].values): #loop for team
                point_tally_dict[l][i][j]=0
                temp_df_1=temp_df[(temp_df["HT"]==j)|(temp_df["AT"]==j)] 
                for k in range(len(temp_df_1)):
                    row=temp_df_1.iloc[k]
                    if row.HT==j: #get value
                        point_tally_HT[row["index"]]=point_tally_dict[l][i][j]
                    elif row.AT==j:
                        point_tally_AT[row["index"]]=point_tally_dict[l][i][j]
                    if row.HT == j: #update
                        point_tally_dict[l][i][j]+=win_draw_loss(row.HS, row.AS)
                    elif row.AT == j:
                        point_tally_dict[l][i][j]+=win_draw_loss(row.AS, row.HS)

    df1["point_tally_HT"]=point_tally_HT
    df1["point_tally_AT"]=point_tally_AT
    return df1,point_tally_dict

def point_tally_valid(dataframe,point_tally_dict):
    df1=dataframe
    point_tally_HT=[np.nan]*len(df1)
    point_tally_AT=[np.nan]*len(df1)
    for i in range(len(df1)):
        row=df1.iloc[i]
        point_tally_HT[i]=point_tally_dict[row.Lge][row.Sea][row.HT]
        point_tally_AT[i]=point_tally_dict[row.Lge][row.Sea][row.AT]
    df1["point_tally_dict_HT"]=point_tally_HT
    df1["point_tally_dict_AT"]=point_tally_AT
    return df1

def point_tally_update(dataframe,point_tally_dict):
    df1=dataframe
    for i in range(len(df1)):
        row=df1.iloc[i]
        point_tally_dict[row.Lge][row.Sea][row.HT]+=win_draw_loss(row.HS, row.AS)
        point_tally_dict[row.Lge][row.Sea][row.AT]+=win_draw_loss(row.AS, row.HS)
    return point_tally_dict


#%% point per match table 1
def point_per_match_train(dataframe):
    df1=dataframe
    point_per_match_HT=[np.nan]*len(df1)
    point_per_match_AT=[np.nan]*len(df1)
    point_per_match_dict={}
    for l in df1.Lge.unique(): #loop for league
        df2=df1[df1["Lge"]==l]     #league df
        point_per_match_dict[l]={}
        for i in df2.Sea.unique(): #loop for season
            temp_df=df2[df2["Sea"]==i] #season df
            point_per_match_dict[l][i]={}
            for j in np.unique(temp_df[['HT', 'AT']].values): #loop for team
                point_per_match_dict[l][i][j]=[0,0] #app,socre
                temp_df_1=temp_df[(temp_df["HT"]==j)|(temp_df["AT"]==j)] 
                for k in range(len(temp_df_1)):
                    row=temp_df_1.iloc[k]
                    #import pdb; pdb.set_trace()
                    if row.HT==j: #get value
                        if point_per_match_dict[l][i][j][0]!=0: 
                            point_per_match_HT[row["index"]]=point_per_match_dict[l][i][j][1]/point_per_match_dict[l][i][j][0]
                    elif row.AT==j:
                        if point_per_match_dict[l][i][j][0]!=0: 
                            point_per_match_AT[row["index"]]=point_per_match_dict[l][i][j][1]/point_per_match_dict[l][i][j][0]
                    if row.HT == j: #update
                        point_per_match_dict[l][i][j][1]+=win_draw_loss(row.HS, row.AS)
                        point_per_match_dict[l][i][j][0]+=1
                    elif row.AT == j:
                        point_per_match_dict[l][i][j][1]+=win_draw_loss(row.AS, row.HS)
                        point_per_match_dict[l][i][j][0]+=1

    df1["point_per_match_HT"]=point_per_match_HT
    df1["point_per_match_AT"]=point_per_match_AT
    
    return df1,point_per_match_dict

def point_per_match_valid(dataframe,point_per_match_dict):
    df1=dataframe
    point_per_match_HT=[np.nan]*len(df1)
    point_per_match_AT=[np.nan]*len(df1)
    for i in range(len(df1)):
        row=df1.iloc[i]
        if point_per_match_dict[row.Lge][row.Sea][row.HT][0]!=0:
            point_per_match_HT[i]=point_per_match_dict[row.Lge][row.Sea][row.HT][1]/point_per_match_dict[row.Lge][row.Sea][row.HT][0]
        if point_per_match_dict[row.Lge][row.Sea][row.AT][0]!=0:
            point_per_match_AT[i]=point_per_match_dict[row.Lge][row.Sea][row.AT][1]/point_per_match_dict[row.Lge][row.Sea][row.AT][0]
    df1["point_per_match_dict_HT"]=point_per_match_HT
    df1["point_per_match_dict_AT"]=point_per_match_AT
    return df1

def point_per_match_update(dataframe,point_per_match_dict):
    df1=dataframe
    for i in range(len(df1)):
        row=df1.iloc[i]
        point_per_match_dict[row.Lge][row.Sea][row.HT][1]+=win_draw_loss(row.HS, row.AS)
        point_per_match_dict[row.Lge][row.Sea][row.HT][0]+=1
        point_per_match_dict[row.Lge][row.Sea][row.AT][1]+=win_draw_loss(row.AS, row.HS)
        point_per_match_dict[row.Lge][row.Sea][row.AT][0]+=1
    return point_per_match_dict

#%% previous season points tally goal score goal conceded goal diff table 1 (in the same league, if promote or demote, np.nan)

def previous_stats_train(dataframe):
    season=[' 00-01', ' 01-02', ' 02-03', ' 03-04', ' 04-05', ' 05-06', ' 06-07',
     ' 07-08', ' 08-09', ' 09-10', ' 10-11', ' 11-12', ' 12-13', ' 13-14',
     ' 14-15', ' 15-16', ' 16-17', ' 17-18', ' 18-19', ' 19-20', ' 20-21',
     ' 21-22', ' 22-23', ' 23-24']
    df1=dataframe
    previous_point_tally_HT=[np.nan]*len(df1)
    previous_GS_HT=[np.nan]*len(df1)
    previous_GC_HT=[np.nan]*len(df1)
    previous_GD_HT=[np.nan]*len(df1)
    previous_point_tally_AT=[np.nan]*len(df1)
    previous_GS_AT=[np.nan]*len(df1)
    previous_GC_AT=[np.nan]*len(df1)
    previous_GD_AT=[np.nan]*len(df1)
    team_previous_stats_dict={}
    league_starting_season={}
    for i in df1.Lge.unique(): #create dict for the stats
        team_previous_stats_dict[i]={}
        league_df=df1[df1["Lge"]==i]
        league_starting_season[i]=league_df.Sea.unique()[0]
        for j in league_df.Sea.unique():
            season_df=league_df[league_df["Sea"]==j]
            team_previous_stats_dict[i][j]={}
            for k in np.unique(season_df[['HT', 'AT']].values):
                team_previous_stats_dict[i][j][k]=[0]*5 #point tally,goal score,goal conceded, goal diff
                team_df=season_df[(season_df["HT"]==k)|(season_df["AT"]==k)]
                #import pdb; pdb.set_trace() 
                for m in range(len(team_df)):
                    row=team_df.iloc[m]
                    if row.HT == k:
                        team_previous_stats_dict[i][j][k][0]+=win_draw_loss(row.HS, row.AS)
                        team_previous_stats_dict[i][j][k][1]+=row.HS
                        team_previous_stats_dict[i][j][k][2]+=row.AS
                        team_previous_stats_dict[i][j][k][3]+=row.HS-row.AS
                        team_previous_stats_dict[i][j][k][4]+=1
                    elif row.AT == k:
                        team_previous_stats_dict[i][j][k][0]+=win_draw_loss(row.AS, row.HS)
                        team_previous_stats_dict[i][j][k][1]+=row.AS
                        team_previous_stats_dict[i][j][k][2]+=row.HS
                        team_previous_stats_dict[i][j][k][3]+=row.AS-row.HS
                        team_previous_stats_dict[i][j][k][4]+=1
                        
      
                
    #get value
    for i in range(len(df1)):
        if df1.iloc[i].Sea==league_starting_season[df1.iloc[i].Lge]: #skip the first season of each league
            continue
        if df1.iloc[i].Sea==" 05-06" and df1.iloc[i].Lge=="AUS1":
            continue
        if df1.iloc[i].Sea==" 12-13" and df1.iloc[i].Lge=="SCO4": #skip for now
            continue
        if df1.iloc[i].Sea==" 18-19" and df1.iloc[i].Lge=="CHL1": #skip for now
            continue
        if df1.iloc[i].Sea==" 23-24" and df1.iloc[i].Lge=="MEX1": #skip for now
            continue
        row=df1.iloc[i]
        required_season=season[season.index(row.Sea)-1]
        if row.HT in team_previous_stats_dict[row.Lge][required_season]:
            previous_point_tally_HT[row["index"]]=team_previous_stats_dict[row.Lge][required_season][row.HT][0]
            previous_GS_HT[row["index"]]=team_previous_stats_dict[row.Lge][required_season][row.HT][1]
            previous_GC_HT[row["index"]]=team_previous_stats_dict[row.Lge][required_season][row.HT][2]
            previous_GD_HT[row["index"]]=team_previous_stats_dict[row.Lge][required_season][row.HT][3]
        if row.AT in team_previous_stats_dict[row.Lge][required_season]:
            previous_point_tally_AT[row["index"]]=team_previous_stats_dict[row.Lge][required_season][row.AT][0]
            previous_GS_AT[row["index"]]=team_previous_stats_dict[row.Lge][required_season][row.AT][1]
            previous_GC_AT[row["index"]]=team_previous_stats_dict[row.Lge][required_season][row.AT][2]
            previous_GD_AT[row["index"]]=team_previous_stats_dict[row.Lge][required_season][row.AT][3]
              
    df1["previous_point_tally_HT"]=previous_point_tally_HT 
    df1["previous_GS_HT"]=previous_GS_HT
    df1["previous_GC_HT"]=previous_GC_HT
    df1["previous_GD_HT"]=previous_GD_HT
    df1["previous_point_tally_AT"]=previous_point_tally_AT 
    df1["previous_GS_AT"]=previous_GS_AT
    df1["previous_GC_AT"]=previous_GC_AT
    df1["previous_GD_AT"]=previous_GD_AT
    return df1, team_previous_stats_dict


def previous_stats_valid(dataframe,input_dict):
    team_previous_stats_dict=input_dict
    season=[' 00-01', ' 01-02', ' 02-03', ' 03-04', ' 04-05', ' 05-06', ' 06-07',
     ' 07-08', ' 08-09', ' 09-10', ' 10-11', ' 11-12', ' 12-13', ' 13-14',
     ' 14-15', ' 15-16', ' 16-17', ' 17-18', ' 18-19', ' 19-20', ' 20-21',
     ' 21-22', ' 22-23', ' 23-24']
    df1=dataframe
    previous_point_tally_HT=[np.nan]*len(df1)
    previous_GS_HT=[np.nan]*len(df1)
    previous_GC_HT=[np.nan]*len(df1)
    previous_GD_HT=[np.nan]*len(df1)
    previous_point_tally_AT=[np.nan]*len(df1)
    previous_GS_AT=[np.nan]*len(df1)
    previous_GC_AT=[np.nan]*len(df1)
    previous_GD_AT=[np.nan]*len(df1)
    for i in range(len(df1)):
        if df1.iloc[i].Sea==" 05-06" and df1.iloc[i].Lge=="AUS1":
            continue
        if df1.iloc[i].Sea==" 12-13" and df1.iloc[i].Lge=="SCO4": #skip for now
            continue
        if df1.iloc[i].Sea==" 18-19" and df1.iloc[i].Lge=="CHL1": #skip for now
            continue
        if df1.iloc[i].Sea==" 23-24" and df1.iloc[i].Lge=="MEX1": #skip for now
            continue
        row=df1.iloc[i]
        required_season=season[season.index(row.Sea)-1]
        if row.HT in team_previous_stats_dict[row.Lge][required_season]:
            previous_point_tally_HT[row["index"]]=team_previous_stats_dict[row.Lge][required_season][row.HT][0]
            previous_GS_HT[row["index"]]=team_previous_stats_dict[row.Lge][required_season][row.HT][1]
            previous_GC_HT[row["index"]]=team_previous_stats_dict[row.Lge][required_season][row.HT][2]
            previous_GD_HT[row["index"]]=team_previous_stats_dict[row.Lge][required_season][row.HT][3]
        if row.AT in team_previous_stats_dict[row.Lge][required_season]:
            previous_point_tally_AT[row["index"]]=team_previous_stats_dict[row.Lge][required_season][row.AT][0]
            previous_GS_AT[row["index"]]=team_previous_stats_dict[row.Lge][required_season][row.AT][1]
            previous_GC_AT[row["index"]]=team_previous_stats_dict[row.Lge][required_season][row.AT][2]
            previous_GD_AT[row["index"]]=team_previous_stats_dict[row.Lge][required_season][row.AT][3]
    return df1


#%% Days since first match in the league

def days_since_first_match(dataframe):
    from datetime import datetime
    df1=dataframe
    day_list=[np.nan]*len(df1)
    lge_start_dict={}
    for i in df1.Lge.unique():
        lge_df=df1[df1["Lge"]==i]
        lge_start_dict[i]={}
        for j in lge_df.Sea.unique():
            lge_start_dict[i][j]=lge_df[lge_df["Sea"]==j].iloc[0].Date
            
    for i in range(len(df1)):
        row=df1.iloc[i]
        Day=datetime.strptime(row.Date,"%d/%m/%Y")-datetime.strptime(lge_start_dict[row.Lge][row.Sea],"%d/%m/%Y")
        day_list[row["index"]]=Day.days
        
    df1["days_since_first_match"]=day_list
    return df1,lge_start_dict



#%% quarter table 1

def quater_value(x):
    if x in [1,2,3]:
        return 1
    elif x in [4,5,6]:
        return 2
    elif x in [7,8,9]:
        return 3
    elif x in [10,11,12]:
        return 4

def quater(dataframe):
    from datetime import datetime 
    df1=dataframe
    df1["quater"]=np.nan
    temp=[np.nan]*len(df1)
    for k in range(len(df1)):
        Date=datetime.strptime(df1.iloc[k]["Date"],"%d/%m/%Y")
        temp[df1.iloc[k]["index"]]=quater_value(int(Date.month))
    df1["quater"]=temp
    return df1

#%% current form
def current_form(dataframe,n=5):
    df1=dataframe
    
    win_pct_ht=[np.nan]*len(df1)
    draw_pct_ht=[np.nan]*len(df1)  
    GS_AVG_ht=[np.nan]*len(df1)  
    GC_AVG_ht=[np.nan]*len(df1)  
    GS_STD_ht=[np.nan]*len(df1)  
    GC_STD_ht=[np.nan]*len(df1)  
    win_pct_at=[np.nan]*len(df1)  
    draw_pct_at=[np.nan]*len(df1)  
    GS_AVG_at=[np.nan]*len(df1)  
    GC_AVG_at=[np.nan]*len(df1)  
    GS_STD_at=[np.nan]*len(df1)  
    GC_STD_at=[np.nan]*len(df1)
    
    current_form_dict={} #[Lge][Sea][team]
    for i in df1.Lge.unique():
        df_lge=df1[df1["Lge"]==i]
        current_form_dict[i]={}
        for j in df_lge.Sea.unique():
            df_sea=df_lge[df_lge["Sea"]==j]
            current_form_dict[i][j]={}
            for k in np.unique(df_sea[['HT', 'AT']].values): #loop for team
                df_team=df_sea[(df_sea["HT"]==k)|(df_sea["AT"]==k)]  
                current_form_dict[i][j][k]=[0,0,0,[np.nan]*n,[np.nan]*n] #game played, game win, game draw, goal scored, goal conceded
                for m in range(len(df_team)):
                    row=df_team.iloc[m]
                    
                    #get value
                    if m>=n:
                        if k==row.HT:
                            win_pct_ht[row["index"]]=current_form_dict[i][j][k][1]/current_form_dict[i][j][k][0]
                            draw_pct_ht[row["index"]]=current_form_dict[i][j][k][2]/current_form_dict[i][j][k][0]
                            GS_AVG_ht[row["index"]]= np.mean(current_form_dict[i][j][k][3])
                            GC_AVG_ht[row["index"]]= np.mean(current_form_dict[i][j][k][4])
                            GS_STD_ht[row["index"]]= np.std(current_form_dict[i][j][k][3])
                            GC_STD_ht[row["index"]]= np.std(current_form_dict[i][j][k][4])
                        elif k==row.AT:
                            win_pct_at[row["index"]]=current_form_dict[i][j][k][1]/current_form_dict[i][j][k][0]
                            draw_pct_at[row["index"]]= current_form_dict[i][j][k][2]/current_form_dict[i][j][k][0]
                            GS_AVG_at[row["index"]]= np.mean(current_form_dict[i][j][k][3])
                            GC_AVG_at[row["index"]]= np.mean(current_form_dict[i][j][k][4])
                            GS_STD_at[row["index"]]= np.std(current_form_dict[i][j][k][3])
                            GC_STD_at[row["index"]]= np.std(current_form_dict[i][j][k][4])

                    #update
                    current_form_dict[i][j][k][0]+=1
                    if row.WDL=="W":
                        current_form_dict[i][j][k][1]+=1
                    elif row.WDL=="D":
                        current_form_dict[i][j][k][2]+=1
                    if k==row.HT:
                        #import pdb; pdb.set_trace() 
                        current_form_dict[i][j][k][3].append(row.HS)
                        current_form_dict[i][j][k][3].pop(0)
                        current_form_dict[i][j][k][4].append(row.AS)
                        current_form_dict[i][j][k][4].pop(0)
                    elif k==row.AT:
                        #import pdb; pdb.set_trace() 
                        current_form_dict[i][j][k][4].append(row.HS)
                        current_form_dict[i][j][k][4].pop(0)
                        current_form_dict[i][j][k][3].append(row.AS)
                        current_form_dict[i][j][k][3].pop(0)
    df1["win_pct_ht"]=win_pct_ht
    df1["draw_pct_ht"]=draw_pct_ht  
    df1["GS_AVG_ht"]=GS_AVG_ht  
    df1["GC_AVG_ht"]=GC_AVG_ht  
    df1["GS_STD_ht"]=GS_STD_ht  
    df1["GC_STD_ht"]=GC_STD_ht  
    df1["win_pct_at"]=win_pct_at  
    df1["draw_pct_at"]=draw_pct_at 
    df1["GS_AVG_at"]=GS_AVG_at  
    df1["GC_AVG_at"]=GC_AVG_at  
    df1["GS_STD_at"]=GS_STD_at  
    df1["GC_STD_at"]=GC_STD_at  

        
    return df1,current_form_dict

#%%


if __name__ == '__main__':
    df=pd.read_excel("C:/Users/calvi/Desktop/2023 soccer prediction challenge/TrainingSet_2023_02_08.xlsx")
    prediction_set=pd.read_excel("C:/Users/calvi/Desktop/2023 soccer prediction challenge/PredictionSet_2023_01_31.xlsx")
    #% check number of match and leagure in each season

    # for i in df.Sea.unique():
    #     check_df=df[df["Sea"]==i]
    #     print(f"# of Lge in {i}: ",len(check_df.Lge.unique()))
    #     #print(check_df.Lge.value_counts())
        
    # print("# of Lge in prediction set",len(prediction_set.Lge.unique()))   
    #% 
    '''
    ['00-01', '01-02', '02-03', '03-04', '04-05', '05-06', '06-07',
           '07-08', '08-09', '09-10', '10-11', '11-12', '12-13', '13-14',
           '14-15', '15-16', '16-17', '17-18', '18-19', '19-20', '20-21',
           '21-22', '22-23']
    # of Lge in 00-01:  27
    # of Lge in 01-02:  28
    # of Lge in 02-03:  29
    # of Lge in 03-04:  31
    # of Lge in 04-05:  35
    # of Lge in 05-06:  37
    # of Lge in 06-07:  41
    # of Lge in 07-08:  43
    # of Lge in 08-09:  46
    # of Lge in 09-10:  48
    # of Lge in 10-11:  48
    # of Lge in 11-12:  50
    # of Lge in 12-13:  51
    # of Lge in 13-14:  51
    # of Lge in 14-15:  51
    # of Lge in 15-16:  51
    # of Lge in 16-17:  51
    # of Lge in 17-18:  48
    # of Lge in 18-19:  48
    # of Lge in 19-20:  48
    # of Lge in 20-21:  48
    # of Lge in 21-22:  48
    # of Lge in 22-23:  9
    '''
    '''
    # of Lge in prediction set:  34
    ['ARG1', 'AUS1', 'BEL1', 'CHE1', 'CHL1', 'DZA1', 'ENG1', 'ENG2',
           'ENG3', 'ENG4', 'ENG5', 'FRA1', 'FRA2', 'FRA3', 'GER1', 'GER2',
           'GER3', 'HOL1', 'ITA1', 'ITA2', 'MAR1', 'MEX1', 'POR1',
           'RUS1', 'RUS2', 'SCO1', 'SCO2', 'SCO3', 'SCO4', 'SPA1', 'SPA2',
           'SWE1', 'USA1', 'ZAF1']
    #exclude NOR1
    '''
    #% filter out leagues that are not required and create the index features
    required_league=['ARG1', 'AUS1', 'BEL1', 'CHE1', 'CHL1', 'DZA1', 'ENG1', 'ENG2',
           'ENG3', 'ENG4', 'ENG5', 'FRA1', 'FRA2', 'FRA3', 'GER1', 'GER2',
           'GER3', 'HOL1', 'ITA1', 'ITA2', 'MAR1', 'MEX1', 'POR1',
           'RUS1', 'RUS2', 'SCO1', 'SCO2', 'SCO3', 'SCO4', 'SPA1', 'SPA2',
           'SWE1', 'USA1', 'ZAF1']
    df1=df[df["Lge"].isin(required_league)]
    df1=df1.reset_index(drop=True)
    df1=df1.reset_index()
    
    prediction_set=prediction_set[prediction_set["Lge"]!="NOR1"]
    
    
    #% train set featrues create
    testing,team_GD_GS_GC_dict=GD_GS_GC_train(df1)
    testing,Streak_and_Weighted_Streak_dict=Streak_and_Weighted_Streak_train(df1,hyperparameter_k=6)
    testing,Form_dict=Form_train(df1,initialize_form=1,gamma=0.33)
    testing,Berrar_Recencyfeature_team_dict=Berrar_Recencyfeature_train(df1,n=9)
    testing=Round_train(df1)
    testing,league_table=Match_important_train(df1,top=5,bottom=5)
    testing=days_since_previous_train(df1)
    testing=quater(df1)
    testing,_=newly_promoted_domoted_train(df1)
    testing,_=Form_tb1_train(df1)
    testing,_=point_tally_train(df1)
    testing,_=point_per_match_train(df1)
    testing,temp_dict=previous_stats_train(df1)
    testing,_=days_since_first_match(df1)
    testing,testing1=current_form(df1,n=5)
    #% validation set features create
    
    
    
    
    #% check 
    # i= 'RUS2' 
    # j='Zenit St Petersburg 2'
    # j="FC Volgar Astrakhan"
    # j="Spartak Moscow 2"
    
    # temp_df=df1[df1["Lge"]==i]
    # read_1=temp_df[(temp_df["HT"]==j)|(temp_df["AT"]==j)]








