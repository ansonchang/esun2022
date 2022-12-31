import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

df_=pd.read_csv('../output/final_output.csv')

file_list=[]
for i, row in df_.iterrows():
    if row['stacker2']>=0.015:
        file_list.append( f"{row['seed']}_stacker2_{row['csv']}")
    if row['blender']>=0.015:
        file_list.append(f"{row['seed']}_blender_{row['csv']}")    

folder='../output/'

file_cnt=len(file_list)

col_list=['prob','rank']
res_list=['alert_key']
for i in range(1):
    for col in col_list:
        res_list.append(f'{col}_{i}')

df_ = pd.DataFrame(columns=res_list)

for i,file in enumerate(file_list):
    if i==0:
        tmp_df = pd.read_csv(folder+file)
        tmp_df.columns=['alert_key','prob_0']
        tmp_df=tmp_df.sort_values('prob_0',ascending = False).reset_index(drop=True)
        tmp_df['rank_0']=tmp_df.index+1
        df_=df_.append(tmp_df)
    else:
        tmp_df = pd.read_csv(folder+file)
        tmp_df.columns=['alert_key','prob_'+str(i)]
        tmp_rk_df=tmp_df.sort_values(f'prob_{i}',ascending = False).reset_index(drop=True)
        tmp_rk_df['rank_'+str(i)]=tmp_rk_df.index+1
        df_ = pd.merge(df_,tmp_rk_df,on=['alert_key'],how='left')

rank_list=df_['rank_0'].values.tolist()
for i in range(1,file_cnt):
    rank_list = (np.array(rank_list) + np.array(df_['rank_'+str(i)].values.tolist())).tolist()

df_['rank_sum']=rank_list


# Rank sum

df_=df_.sort_values('rank_sum',ascending = True).reset_index(drop=True)

df_['rank_idx']=df_.index

df_['rank_score']= df_['rank_idx'].apply(lambda x: (3850-x)/3851)

submit=df_[['alert_key','rank_score']]
submit.columns=['alert_key','probability']

submit.to_csv('../output/final_rank_sum.csv', index=False)
