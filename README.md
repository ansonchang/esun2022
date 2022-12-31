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
0. 安裝套件

   $ pip install requirements.txt

1. 原始 dataset 需放置在上一層 data 目錄

2. 執行資料預處理步驟 :
  $ cd Proprocess
  $ python esun_preprocess.py

3. 執行模型訓練與預測 : (總共 500 次 model training 產生 100個預測檔案, 需要 1.5 hours)
  $ cd Model
  $ python esun_preprocess.py

4. 執行  Rank ensemble (100 預測檔案, 預測最終檔案)
  $ cd Model
  $ python ./Model/esun_rank_ensemble.py

5. 得到最後結果 

