## 環境
* 系統平台：Ubuntu
* 程式語言：python 3.7.13

## 每個資料夾/檔案的用途
```
├ Preprocess
│ └ esun_preprocess.py             (前處理與特徵工程, 在 csv 目錄產生 train dataset 與 private test tdataset)
├ Model
│ ├ esun_model_predict.py          (model訓練, 在 output 目錄產生 model預測結果)
│ └ esun_rank_ensemble.py          (ensenble model預測結果, 產生最終的結果)
├ requirements.txt
└ README.md
```

## 可復現步驟
0. dataset 先放到上一層 data 目錄

1. 執行 preprocess :
   cd Proprocess
   python esun_preprocess.py
   
2. 執行 model training & prediction : (總共 500 次 model training 需要執行 1.5 hours)
   cd Model
   python esun_preprocess.py

3. 執行 rank ensemble => python ./Model/esun_rank_ensemble.py

4. 得到最後結果 


## 可復現模型結果之超參數設定
