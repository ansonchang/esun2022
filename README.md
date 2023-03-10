## 環境
- 系統平台：Ubuntu 18.04.6
- 程式語言：python 3.8.13

## 每個資料夾/檔案的用途
```
├ Preprocess
│ └ esun_preprocess.py             (前處理與特徵工程, 在 csv 目錄產生 train/test dataset)
├ Model
│ ├ esun_model_predict.py          (model 訓練與預測, 在 output 目錄產生 model 預測結果)
│ └ esun_rank_ensemble.py          (ensemble 各 model 的預測結果, 在 output 目錄產生最終的結果)
├ csv                              (放置 data prepocess 後的資料集目錄)
├ output                           (放置 model prediction 後的資料集目錄)
├ requirements.txt
└ README.md
```

## 可復現步驟

0. 安裝套件
```
$ pip install -r requirements.txt
```

1. 執行資料預處理步驟 : (原始 dataset 需放置在上一層 data 目錄)
```
$ cd Proprocess
$ python esun_preprocess.py
```

2. 執行模型訓練與預測 : (總共 500 次 model training 產生 100個預測檔案, 需要 1.5 hours)
```
$ cd Model
$ python esun_model_predict.py
```

3. 執行  Rank ensemble (ensemble 100 預測檔案, 得到最終檔案)
 ```  
$ cd Model
$ python esun_rank_ensemble.py
 ```
