
# -*- coding: utf-8 -*-

#%%
import numpy as np
import pandas as pd
#from sklearn.preprocessing import LabelEncoder
#import argparse
from pyswarms.single.global_best import GlobalBestPSO
import warnings
import pickle
import time
#%%

def calculate_expected_goals(alpha, beta_H, gamma_H, beta_A, gamma_A, o_H, d_A, o_A, d_H):
    G_H_hat = alpha / (1 + np.exp(- beta_H * (o_H + d_A) - gamma_H))
    G_A_hat = alpha / (1 + np.exp(- beta_A * (o_A + d_H) - gamma_A))
    return G_H_hat, G_A_hat

def update_offensive_defensive_strengths(G_H, G_H_hat, G_A, G_A_hat, o_H, d_H, o_A, d_A, omega_o_H, omega_d_H, omega_o_A, omega_d_A):
    o_H += omega_o_H * (G_H - G_H_hat)
    d_H += omega_d_H * (G_A - G_A_hat)
    o_A += omega_o_A * (G_A - G_A_hat)
    d_A += omega_d_A * (G_H - G_H_hat)
    return o_H, d_H, o_A, d_A

def individual_goal_pred_error(G_H, G_H_hat, G_A, G_A_hat):
    f = 0.5*((G_H - G_H_hat)**2 + (G_A - G_A_hat)**2)
    return f

#ratings parameter
alpha = 5 # max_goals = max(df['HS'].max(), df['AS'].max())

beta_H = 1 # slope of the logistic function. bounds [0,5]
beta_A = 1 
gamma_H = 0.1 # bias, e.g., home advantage. bounds [-5,5]
gamma_A = 0.1
omega_o_H = 0.2 # used to control the learning rate of the update rule for the offensive and defensive strengths of each team. bounds [0, 1.5]
omega_d_H = 0.2
omega_o_A = 0.2
omega_d_A = 0.2

def total_goal_prediction_error(x=(beta_H, beta_A, gamma_H, gamma_A, omega_o_H, omega_d_H, omega_o_A, omega_d_A)):
    loss=[]

    beta_H = x[:,0] # slope of the logistic function. bounds [0,5]
    beta_A = x[:,1]
    gamma_H = x[:,2] # bias, e.g., home advantage. bounds [-5,5]
    gamma_A = x[:,3] 
    omega_o_H = x[:,4]  # used to control the learning rate of the update rule for the offensive and defensive strengths of each team. bounds [0, 1.5]
    omega_d_H = x[:,5] 
    omega_o_A = x[:,6] 
    omega_d_A = x[:,7]
    team_ratings={}
    #print(round(beta_H,2), round(gamma_H,2), round(beta_A,2), round(gamma_A,2))
    for i in np.unique(df1[['HT', 'AT']].values):
        team_ratings[i]=[0,0,0,0] # Home attack, Home defense, Away Attack, Away defense
    total_goal_pred_error = 0
    total_match=0
    for index, row in df1.iterrows():
        HT = row['HT']
        HS = row['HS']
        AT = row['AT']
        AS = row['AS']

        G_H_hat, G_A_hat = calculate_expected_goals(alpha, beta_H, gamma_H, beta_A, gamma_A,
                                                    team_ratings[HT][0], team_ratings[AT][3],
                                                    team_ratings[AT][2], team_ratings[HT][1])
        G_H=HS
        G_A=AS
        team_ratings[HT][0] += omega_o_H * (G_H - G_H_hat)
        team_ratings[HT][1] += omega_d_H * (G_A - G_A_hat)
        team_ratings[AT][2] += omega_o_A * (G_A - G_A_hat)
        team_ratings[AT][3] += omega_d_A * (G_H - G_H_hat)

        total_goal_pred_error += individual_goal_pred_error(HS, G_H_hat, AS, G_A_hat)
        total_match+=1
    loss=total_goal_pred_error/total_match
    #print("\n",len(loss),round(min(loss),2))
    #import pdb; pdb.set_trace()
    return loss

def berrar_rating(dataframe,x=(beta_H, beta_A, gamma_H, gamma_A, omega_o_H, omega_d_H, omega_o_A, omega_d_A)):
    df1=dataframe
    beta_H = x[0] # slope of the logistic function. bounds [0,5]
    beta_A = x[1] 
    gamma_H = x[2]  # bias, e.g., home advantage. bounds [-5,5]
    gamma_A = x[3] 
    omega_o_H = x[4]  # used to control the learning rate of the update rule for the offensive and defensive strengths of each team. bounds [0, 1.5]
    omega_d_H = x[5] 
    omega_o_A = x[6] 
    omega_d_A = x[7] 
    team_ratings={}
    for i in np.unique(df1[['HT', 'AT']].values):
        team_ratings[i]=[0,0,0,0] # Home attack, Home defense, Away Attack, Away defense
    for index, row in df1.iterrows():
        HT = row['HT']
        HS = row['HS']
        AT = row['AT']
        AS = row['AS']

        #import pdb; pdb.set_trace()
        df1.loc[index, 'HT_H_Off_Rating'] = team_ratings[HT][0]
        #import pdb; pdb.set_trace()
        df1.loc[index, 'HT_H_Def_Rating'] = team_ratings[HT][1]
        df1.loc[index, 'HT_A_Off_Rating'] = team_ratings[HT][2]
        df1.loc[index, 'HT_A_Def_Rating'] = team_ratings[HT][3]
        df1.loc[index, 'AT_H_Off_Rating'] = team_ratings[AT][0]
        df1.loc[index, 'AT_H_Def_Rating'] = team_ratings[AT][1]
        df1.loc[index, 'AT_A_Off_Rating'] = team_ratings[AT][2]
        df1.loc[index, 'AT_A_Def_Rating'] = team_ratings[AT][3]

        G_H_hat, G_A_hat = calculate_expected_goals(alpha, beta_H, gamma_H, beta_A, gamma_A,
                                                    team_ratings[HT][0], team_ratings[AT][3],
                                                    team_ratings[AT][2], team_ratings[HT][1])
        
        df1.loc[index, 'HT_EG'] = G_H_hat
        df1.loc[index, 'AT_EG'] = G_A_hat

        team_ratings[HT][0],team_ratings[HT][1],team_ratings[AT][2],team_ratings[AT][3] = \
            update_offensive_defensive_strengths(HS, G_H_hat, AS, G_A_hat,team_ratings[HT][0], team_ratings[HT][1], team_ratings[AT][2], team_ratings[AT][3],omega_o_H, omega_d_H, omega_o_A, omega_d_A)
    return df1,team_ratings

def berrar_rating_valid(dataframe,team_ratings_dict,x=(beta_H, beta_A, gamma_H, gamma_A, omega_o_H, omega_d_H, omega_o_A, omega_d_A)):
    df1=dataframe
    beta_H = x[0] # slope of the logistic function. bounds [0,5]
    beta_A = x[1] 
    gamma_H = x[2]  # bias, e.g., home advantage. bounds [-5,5]
    gamma_A = x[3] 
    omega_o_H = x[4]  # used to control the learning rate of the update rule for the offensive and defensive strengths of each team. bounds [0, 1.5]
    omega_d_H = x[5] 
    omega_o_A = x[6] 
    omega_d_A = x[7] 
    team_ratings=team_ratings_dict
    for index, row in df1.iterrows():
        HT = row['HT']
        HS = row['HS']
        AT = row['AT']
        AS = row['AS']

        #import pdb; pdb.set_trace()
        df1.loc[index, 'HT_H_Off_Rating'] = team_ratings[HT][0]
        #import pdb; pdb.set_trace()
        df1.loc[index, 'HT_H_Def_Rating'] = team_ratings[HT][1]
        df1.loc[index, 'HT_A_Off_Rating'] = team_ratings[HT][2]
        df1.loc[index, 'HT_A_Def_Rating'] = team_ratings[HT][3]
        df1.loc[index, 'AT_H_Off_Rating'] = team_ratings[AT][0]
        df1.loc[index, 'AT_H_Def_Rating'] = team_ratings[AT][1]
        df1.loc[index, 'AT_A_Off_Rating'] = team_ratings[AT][2]
        df1.loc[index, 'AT_A_Def_Rating'] = team_ratings[AT][3]

        G_H_hat, G_A_hat = calculate_expected_goals(alpha, beta_H, gamma_H, beta_A, gamma_A,
                                                    team_ratings[HT][0], team_ratings[AT][3],
                                                    team_ratings[AT][2], team_ratings[HT][1])
        
        df1.loc[index, 'HT_EG'] = G_H_hat
        df1.loc[index, 'AT_EG'] = G_A_hat

        team_ratings[HT][0],team_ratings[HT][1],team_ratings[AT][2],team_ratings[AT][3] = \
            update_offensive_defensive_strengths(HS, G_H_hat, AS, G_A_hat,team_ratings[HT][0], team_ratings[HT][1], team_ratings[AT][2], team_ratings[AT][3],omega_o_H, omega_d_H, omega_o_A, omega_d_A)
    return df1,team_ratings

def berrar_rating_valid_final(dataframe,team_ratings_dict,x=(beta_H, beta_A, gamma_H, gamma_A, omega_o_H, omega_d_H, omega_o_A, omega_d_A)):
    df1=dataframe
    beta_H = x[0] # slope of the logistic function. bounds [0,5]
    beta_A = x[1] 
    gamma_H = x[2]  # bias, e.g., home advantage. bounds [-5,5]
    gamma_A = x[3] 
    omega_o_H = x[4]  # used to control the learning rate of the update rule for the offensive and defensive strengths of each team. bounds [0, 1.5]
    omega_d_H = x[5] 
    omega_o_A = x[6] 
    omega_d_A = x[7] 
    team_ratings=team_ratings_dict

    HT = df1['HT']
    AT = df1['AT']


    G_H_hat, G_A_hat = calculate_expected_goals(alpha, beta_H, gamma_H, beta_A, gamma_A,
                                                team_ratings[HT][0], team_ratings[AT][3],
                                                team_ratings[AT][2], team_ratings[HT][1])
    
    return G_H_hat,G_A_hat

if __name__ == '__main__':
    start_time = time.time()
    for file in ["trainset_16-17_before","trainset_17-18_before","trainset_18-19_before","trainset_19-20_before","trainset_20-21_before",
                 "trainset_16-17_exact","trainset_17-18_exact","trainset_18-19_exact","trainset_19-20_exact","trainset_20-21_exact"]:
        df = pd.read_csv(f"C:/Users/calvi\Desktop/2023 soccer prediction challenge/split dataset 5yr/{file}.csv")
        df=df[df["Sea"].isin(df["Sea"].unique()[-5:])] #take 5 season of data only
        
        #hyperparameter for PSO
        x_max = np.array([5, 5, 5, 5, 1.5, 1.5, 1.5, 1.5])
        x_min = np.array([0, 0, -5, -5, 0, 0, 0, 0])
        bounds = (x_min, x_max)
        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
        
        warnings.simplefilter('ignore')
        for league in df.Lge.unique():
            print(league)
            df1=df[df["Lge"]==league]
            #team_to_index = {team_name: i for i, team_name in enumerate(np.unique(np.concatenate((df1['HT'].tolist(), df1['AT'].tolist()), axis=0)))}
            optimizer = GlobalBestPSO(n_particles=50, dimensions=8, options=options, bounds=bounds)
            cost, pos = optimizer.optimize(total_goal_prediction_error, 200)
            # pos=([ 4.01929603,  3.99244614,  1.10115083, -2.06364501,  0.80629929,
            #         0.57379537,  0.95790575,  1.00337636])
            df1,team_ratings=berrar_rating(df1,x=pos)
            df1.to_csv(f"C:/Users/calvi/Desktop/2023 soccer prediction challenge/berrar rating/{file}_{league}.csv", index=False)
            with open(f'C:/Users/calvi/Desktop/2023 soccer prediction challenge/berrar rating/{file}_{league}_berrarratings_hyperparameters.pickle', 'wb') as handle:
                pickle.dump(pos, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(f'C:/Users/calvi/Desktop/2023 soccer prediction challenge/berrar rating/{file}_{league}_team_ratings_dict.pickle', 'wb') as handle:
                pickle.dump(team_ratings, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("--- %s seconds ---" % (time.time() - start_time))
