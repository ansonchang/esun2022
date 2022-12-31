import pandas as pd
import numpy as np
#from sklearn.model_selection import train_test_split

def recall_n(pred_p, true_l):
    '''metric Recall@N'''
    n, cur_sum = sum(true_l), 0
    rank_pair = sorted(zip(pred_p,true_l),key=lambda x: x[0],reverse=True)
    for i, (_p, _l) in enumerate(rank_pair):
        cur_sum += _l
        if cur_sum==n:
            return n/(i+1)
        
def recall_n_1(pred_p, true_l):
    '''metric Recall@N-1'''
    n, cur_sum = sum(true_l)-1, 0
    rank_pair = sorted(zip(pred_p,true_l),key=lambda x: x[0],reverse=True)
    for i, (_p, _l) in enumerate(rank_pair):
        cur_sum += _l
        if cur_sum==n:
            return n/(i+1)

cust=pd.read_csv('../../data/public_train_x_custinfo_full_hashed.csv')
ccba=pd.read_csv('../../data/public_train_x_ccba_full_hashed.csv')
cdtx=pd.read_csv('../../data/public_train_x_cdtx0001_full_hashed.csv')
dp=pd.read_csv('../../data/public_train_x_dp_full_hashed.csv')
remit=pd.read_csv('../../data/public_train_x_remit1_full_hashed.csv')
alert_test=pd.read_csv('../../data/public_x_alert_date.csv')

alert_train=pd.read_csv('../../data/train_x_alert_date.csv')
y_train=pd.read_csv('../../data/train_y_answer.csv')
y_test=pd.read_csv('../../data/Sample.csv')

cust_private=pd.read_csv('../../data/private_x_custinfo_full_hashed.csv')
ccba_private=pd.read_csv('../../data/private_x_ccba_full_hashed.csv')
cdtx_private=pd.read_csv('../../data/private_x_cdtx0001_full_hashed.csv')
dp_private=pd.read_csv('../../data/private_x_dp_full_hashed.csv')
remit_private=pd.read_csv('../../data/private_x_remit1_full_hashed.csv')
alert_test_private=pd.read_csv('../../data/private_x_alert_date.csv')

cust=pd.concat([cust,cust_private],axis=0).reset_index(drop=True)
ccba=pd.concat([ccba,ccba_private],axis=0).reset_index(drop=True)
cdtx=pd.concat([cdtx,cdtx_private],axis=0).reset_index(drop=True)
dp=pd.concat([dp,dp_private],axis=0).reset_index(drop=True)
remit=pd.concat([remit,remit_private],axis=0).reset_index(drop=True)
alert_test=pd.concat([alert_test,alert_test_private],axis=0).reset_index(drop=True)

# Calculate dataset (date<=290)
 
y_test['sar_flag']=np.nan

alert_all=pd.concat([alert_train,alert_test],axis=0)

y_all=pd.concat([y_train,y_test[['alert_key','sar_flag']]],axis=0)

all_data = pd.merge(y_all,cust,how='left')
all_data = pd.merge(all_data,alert_all,how='left')

all_date1=all_data[(all_data.date<=290)]

all_date2=all_data[all_data.sar_flag.isnull()]

all_data=pd.concat([all_date1, all_date2], axis=0)

dp1_agg = dp.groupby(['cust_id','tx_date']).agg({
    'ATM':['sum','count'],
}).reset_index()
dp1_agg.columns = ['_'.join([f'{y}' for y in x if y]) for x in dp1_agg.columns]
dp1_agg['ATM_rate']=dp1_agg.ATM_sum/dp1_agg.ATM_count

dp1_agg.columns=['cust_id', 'date', 'ATM_sum','daily_count','ATM_rate']
all_data = pd.merge(all_data,dp1_agg,on=['cust_id','date'],how='left')

dp_agg = dp.groupby(['cust_id','tx_date','tx_type']).agg({
    'tx_amt': ['max','mean','std']
}).unstack().reset_index()
dp_agg.columns = ['_'.join([f'{y}' for y in x if y]) for x in dp_agg.columns]

dp_agg.columns=['cust_id', 'date', 'tx_amt_max_1', 'tx_amt_max_2', 'tx_amt_max_3',
       'tx_amt_mean_1', 'tx_amt_mean_2', 'tx_amt_mean_3', 'tx_amt_std_1',
       'tx_amt_std_2', 'tx_amt_std_3']

all_data = pd.merge(all_data,dp_agg,on=['cust_id','date'],how='left')

all_data['AGE']=all_data['AGE'].astype('category') 
AGE_ohe = pd.get_dummies(all_data[['AGE']])
all_data=pd.concat([all_data, AGE_ohe], axis=1)

all_data['occupation_code']=all_data['occupation_code'].astype('category') 
occupation_code_ohe = pd.get_dummies(all_data[['occupation_code']])
all_data=pd.concat([all_data, occupation_code_ohe], axis=1)

all_data=all_data.drop(['AGE','occupation_code'], axis=1)

date_agg=all_data.groupby(['cust_id']).agg({'date': ['count']}).reset_index()
date_agg.columns = ['_'.join([str(y) for y in x if y]) for x in date_agg.columns]
all_data = pd.merge(all_data,date_agg,on=['cust_id'],how='left')


df_train_all=all_data[all_data.sar_flag.notnull()]
df_test=all_data[all_data.sar_flag.isnull()]

custMaxDate_train={}
for i, _id in enumerate(df_train_all['cust_id'].unique()):
    date_max=df_train_all[df_train_all['cust_id']==_id].date.max()
    custMaxDate_train[_id]=date_max
df_train_maxdate =pd.DataFrame(custMaxDate_train.items(), columns=['cust_id', 'maxdate'])
df_train_all=pd.merge(df_train_all,df_train_maxdate, how='left')

custMaxDate_test={}
for i, _id in enumerate(df_test['cust_id'].unique()):
    date_max=df_test[df_test['cust_id']==_id].date.max()
    custMaxDate_test[_id]=date_max
df_test_maxdate =pd.DataFrame(custMaxDate_test.items(), columns=['cust_id', 'maxdate'])
df_test=pd.merge(df_test,df_test_maxdate, how='left')

df_train_all['maxdate_dif']=df_train_all['maxdate']-df_train_all['date']
df_test['maxdate_dif']=df_test['maxdate']-df_test['date']

df_train=df_train_all.copy()

df_test.to_csv('../csv/df_test.csv', index=False)

df_train.to_csv('../csv/df_train_290.csv', index=False)


# Calculate dataset (date<=280)

y_test['sar_flag']=np.nan

alert_all=pd.concat([alert_train,alert_test],axis=0)

y_all=pd.concat([y_train,y_test[['alert_key','sar_flag']]],axis=0)

all_data = pd.merge(y_all,cust,how='left')
all_data = pd.merge(all_data,alert_all,how='left')

all_date1=all_data[(all_data.date<=280)]

all_date2=all_data[all_data.sar_flag.isnull()]

all_data=pd.concat([all_date1, all_date2], axis=0)

dp1_agg = dp.groupby(['cust_id','tx_date']).agg({
    'ATM':['sum','count'],
}).reset_index()
dp1_agg.columns = ['_'.join([f'{y}' for y in x if y]) for x in dp1_agg.columns]
dp1_agg['ATM_rate']=dp1_agg.ATM_sum/dp1_agg.ATM_count

dp1_agg.columns=['cust_id', 'date', 'ATM_sum','daily_count','ATM_rate']
all_data = pd.merge(all_data,dp1_agg,on=['cust_id','date'],how='left')

dp_agg = dp.groupby(['cust_id','tx_date','tx_type']).agg({
    'tx_amt': ['max','mean','std']
}).unstack().reset_index()
dp_agg.columns = ['_'.join([f'{y}' for y in x if y]) for x in dp_agg.columns]

dp_agg.columns=['cust_id', 'date', 'tx_amt_max_1', 'tx_amt_max_2', 'tx_amt_max_3',
       'tx_amt_mean_1', 'tx_amt_mean_2', 'tx_amt_mean_3', 'tx_amt_std_1',
       'tx_amt_std_2', 'tx_amt_std_3']

all_data = pd.merge(all_data,dp_agg,on=['cust_id','date'],how='left')

all_data['AGE']=all_data['AGE'].astype('category') 
AGE_ohe = pd.get_dummies(all_data[['AGE']])
all_data=pd.concat([all_data, AGE_ohe], axis=1)

all_data['occupation_code']=all_data['occupation_code'].astype('category') 
occupation_code_ohe = pd.get_dummies(all_data[['occupation_code']])
all_data=pd.concat([all_data, occupation_code_ohe], axis=1)

all_data=all_data.drop(['AGE','occupation_code'], axis=1)

date_agg=all_data.groupby(['cust_id']).agg({'date': ['count']}).reset_index()
date_agg.columns = ['_'.join([str(y) for y in x if y]) for x in date_agg.columns]
all_data = pd.merge(all_data,date_agg,on=['cust_id'],how='left')


df_train_all=all_data[all_data.sar_flag.notnull()]
df_test=all_data[all_data.sar_flag.isnull()]

custMaxDate_train={}
for i, _id in enumerate(df_train_all['cust_id'].unique()):
    date_max=df_train_all[df_train_all['cust_id']==_id].date.max()
    custMaxDate_train[_id]=date_max
df_train_maxdate =pd.DataFrame(custMaxDate_train.items(), columns=['cust_id', 'maxdate'])
df_train_all=pd.merge(df_train_all,df_train_maxdate, how='left')

custMaxDate_test={}
for i, _id in enumerate(df_test['cust_id'].unique()):
    date_max=df_test[df_test['cust_id']==_id].date.max()
    custMaxDate_test[_id]=date_max
df_test_maxdate =pd.DataFrame(custMaxDate_test.items(), columns=['cust_id', 'maxdate'])
df_test=pd.merge(df_test,df_test_maxdate, how='left')

df_train_all['maxdate_dif']=df_train_all['maxdate']-df_train_all['date']
df_test['maxdate_dif']=df_test['maxdate']-df_test['date']

df_train=df_train_all.copy()


df_train.to_csv('../csv/df_train_280.csv', index=False)