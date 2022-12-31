import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pycaret.classification import *
import time

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

'''
Decide model 計算組合: train_csv_list X seed_list X model_list 
本次繳交總共 100 個組合

train_csv_list : data period, 總共使用前 290 days 與前 280 days 訓練集
seed_list : 採用 50 個 random seed for train_tes_split
model_list : 使用 stacker model 
'''        

seed_list=[42,142,242,342,442,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,
           13,113,213,313,413,513,613,713,813,913,1013,1113,1213,1313,1413,
           79,179,279,379,479,579,679,779,879,979,1079,1179,1279,1379,1479,
           721,1721,2721,3721,4721]
train_csv_list = ['df_train_290.csv','df_train_280.csv']
model_list = ['stacker2']


df_ = pd.DataFrame(columns=['csv','seed','cols','lgb','xgb','xgb2','blender','stacker2'])

df_test = pd.read_csv('../csv/df_test.csv')
y=[355091,355152,355724,363033,358453,359668,363320,363896,356602,354939,361617]
df_test['sar_flag']=np.where(df_test.alert_key.isin(y),1,0)

start = time.time()
for seed in seed_list:
    print(str(seed) +" seed 執行時間：%f 秒" % ( time.time() - start))
    for csv in train_csv_list:
        print(csv+" 執行時間：%f 秒" % ( time.time() - start))
        df_train = pd.read_csv('../csv/'+csv)
        cols=0
        lgb_score=0
        xgb_score=0
        xgb2_score=0
        blender_score=0
        stacker2_score=0
        s = setup(data = df_train, target = 'sar_flag', session_id=seed, train_size=0.8,
                   ignore_features=['alert_key','cust_id','date','maxdate','ATM_sum',
                          'date_count','tx_amt_max_3','tx_amt_mean_3','tx_amt_std_3'])
        X_train = get_config('X_train')
        cols = len(X_train.columns)
        
        lgb=create_model('lightgbm')
        xgb=create_model('xgboost')
        xgb2=create_model('xgboost', max_depth=6,  learning_rate=0.2, fold=10, tree_method='gpu_hist', gpu_id=0, sort='Recall')
        blender=blend_models(estimator_list = [lgb,xgb], method = 'soft',fold=5)
        
        for model_name in model_list:
            print(model_name+ " 執行時間：%f 秒" % ( time.time() - start))
            if model_name=='lgb':
                new_model=lgb
            elif model_name=='xgb':                
                new_model= xgb
            elif model_name=='xgb2':                
                new_model=xgb2 
            elif model_name=='blender':                
                new_model = blender
            elif model_name=='stacker2':
                new_model = stack_models([xgb2,lgb,blender],meta_model=xgb2,fold=5)                
        
            print(model_name)
            pred_holdout3 = predict_model(new_model, data=df_test) 
            pred_holdout3['proba']=np.where(pred_holdout3.prediction_label==1,pred_holdout3.prediction_score, 1-pred_holdout3.prediction_score)
            pred_holdout3=pd.concat([df_test[['cust_id','alert_key','date','maxdate']],pred_holdout3],axis=1)
            pred_holdout3[['alert_key','proba']].to_csv('../output/'+str(seed)+'_'+model_name+'_'+csv, index=False)
            
            score=recall_n_1(pred_holdout3.proba.to_list(), pred_holdout3.sar_flag.to_list())
            score_n=recall_n(pred_holdout3.proba.to_list(), pred_holdout3.sar_flag.to_list())
            
            if model_name=='lgb':
                lgb_score=score 
            elif model_name=='xgb':                
                xgb_score=score 
            elif model_name=='xgb2':                
                xgb2_score=score                 
            elif model_name=='blender':                
                blender_score = score
            elif model_name=='stacker2':                  
                stacker2_score = score
                
        df_=df_.append({'csv' : csv , 'seed' : seed, 'cols' : cols,
                       'lgb':lgb_score,'xgb':xgb_score,'xgb2':xgb2_score,
                        'blender':blender_score,'stacker2':stacker2_score} , ignore_index=True)
print("執行時間：%f 秒" % ( time.time() - start))

df_.to_csv('../output/final_output.csv', index=False)


