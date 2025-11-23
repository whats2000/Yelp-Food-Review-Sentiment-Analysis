# Yelp 評論情緒分類 (CNN 與 LSTM 比較)

## Goal（專案目標）

利用 `yelp.csv` 中的評論文字 `text` 與評分 `stars`：
- 將評分轉為二元情緒標籤（positive / negative）
- 將資料切成 80% train、20% test
- 以 **Keras / TensorFlow** 分別建立：
  - CNN 文字分類模型
  - LSTM 文字分類模型
- 在 test 集上計算並比較各模型的 **Accuracy**
- 比較有無 Dropout（建議 dropout rate = 0.7）對模型表現與過擬合的影響

---

## Data Structure（資料結構）

### 原始資料結構（`yelp.csv`）

主要欄位（已由資料檔檢查）：

- `business_id`：店家 ID，字串
- `date`：評論日期，字串或日期格式
- `review_id`：評論 ID，字串
- `stars`：評分（整數，通常 1–5）
- `text`：評論內容（英文文字為主）
- `type`：記錄類型（例如 "review"）
- `user_id`：使用者 ID
- `cool`：該評論被標記為 cool 的次數（整數）
- `useful`：被標記為 useful 的次數（整數）
- `funny`：被標記為 funny 的次數（整數）

本作業 **只會使用**：

- `text`：作為輸入特徵（X）
- `stars`：轉為二元標籤（y）

### 標籤轉換規則

- 若 `stars >= 4` → 標記為 `1`（positive）
- 若 `stars < 4` → 標記為 `0`（negative）

### 資料切分規則

- 讀取完整 `yelp.csv`
- 僅保留 `text`、`stars`
- 依照標籤轉換產生新欄位（例如 `label`）
- 以 **80% / 20%** 的比例將資料切為：
  - `train_df`（訓練集）
  - `test_df`（測試集）
- 切分時應考慮 **stratify**（依標籤比例分層抽樣），讓 train / test 中正負比率相近

---

## Overview of Steps（整體流程概觀）

1. **資料讀取與基本檢查**  
   - 讀取 `yelp.csv`  
   - 檢查欄位、資料筆數、基本統計資訊  

2. **資料前處理與標籤建立**  
   - 僅保留 `text`、`stars`  
   - 依規則將 `stars` 轉為二元標籤  
   - 切分為 80% train / 20% test  

3. **文字清理與停用詞（stop words）處理**  
   - 統一大小寫、移除標點符號  
   - 移除英文停用詞  

4. **文字轉向量（向量化）**  
   - 使用 Tokenizer 建立字典  
   - 將文字轉為序列並補齊長度（pad sequences）  
   - 使用 Embedding 層（可視為 Word2Vec 類似概念）  

5. **建立 CNN 模型**  
   - 建立 baseline CNN  
   - 加入 Dropout（建議 rate=0.7）建立第二個 CNN 模型  
   - 訓練並記錄 Accuracy / Loss 變化  

6. **建立 LSTM 模型**  
   - 建立 baseline LSTM  
   - 加入 Dropout（建議 rate=0.7）建立第二個 LSTM 模型  
   - 訓練並記錄 Accuracy / Loss 變化  

7. **模型評估與比較**  
   - 在 test 集上計算 CNN / LSTM（含/不含 Dropout）之 Accuracy  
   - 彙整並比較結果  

8. **視覺化與報告**  
   - 繪製訓練過程的 Accuracy / Loss 曲線  
   - 比較 Dropout 對 overfitting 的影響  
   - 撰寫結論與觀察  

---

# Step 1. 資料讀取與基本檢查

### Objective

確保 `yelp.csv` 讀取正確，理解資料大小、欄位以及基本內容，確認後續前處理方向正確。

### Instructions

1. 在 notebook 開頭匯入必要函式庫（pandas、numpy、Keras / TensorFlow 等）。  
2. 讀取 `data/yelp.csv` 為 DataFrame。  
3. 顯示：
   - 前幾筆資料（head）
   - 資料筆數（shape）
   - 欄位名稱與型別（dtypes / info）
4. 檢查：
   - `text`、`stars` 欄位是否存在、型態是否合理
   - 有無缺失值（null 計數）  
5. 若 `text` 或 `stars` 有空值：
   - 規劃刪除或填補策略（本作業建議：刪除缺失列）

### Requirements / Tools

- pandas：讀取與檢查 CSV 資料  
- 基本 EDA：`head()`、`info()`、`isna().sum()` 等

### Expected Output

✅ `df_raw` — 含所有欄位的完整 Yelp 資料表（已大致確認結構與品質）

---

# Step 2. 資料前處理與標籤建立（train / test 切分）

### Objective

將資料轉為二元分類問題，只保留必要欄位，並建立 80/20 的 train / test 分割。

### Instructions

1. 從 `df_raw` 中 **只保留** `text`、`stars` 兩欄形成新 DataFrame（例如 `df`）。  
2. 建立新的標籤欄位（例如 `label`）：
   - 條件：若 `stars >= 4` → `label = 1`（positive）  
   - 否則 → `label = 0`（negative）  
3. 檢查各標籤比例（value_counts / normalize），了解正負樣本是否平衡。  
4. 將 `text` 和 `label` 分開成：
   - 特徵：`X = df["text"]`
   - 標籤：`y = df["label"]`
5. 使用 train/test split：
   - `test_size = 0.2`（20%）
   - `train_size = 0.8`（80%）
   - `stratify = y`（確保標籤比例一致）
   - 固定 `random_state` 以利重現結果  
6. 得到：
   - `X_train`, `X_test`, `y_train`, `y_test`

### Requirements / Tools

- `sklearn.model_selection.train_test_split`
- 二元標籤轉換規則（stars → 0/1）

### Expected Output

✅ `X_train`, `X_test`, `y_train`, `y_test` — 已完成二元標籤與 80/20 分割的文字與標籤資料

---

# Step 3. 文字清理與停用詞處理

### Objective

移除噪音（標點、大小寫、停用詞），讓後續向量化與模型訓練更穩定。

### Instructions

1. 對 `X_train`、`X_test` 的每一則評論進行基本清理：
   - 全部轉為小寫（lowercase）  
   - 去除多餘空白  
   - 移除標點符號、數字（可使用正則表示式）  
2. 準備英文停用詞表：
   - 可使用 `nltk.corpus.stopwords` 或 `sklearn.feature_extraction.text.ENGLISH_STOP_WORDS` 清單  
3. 在清理步驟中加入 **移除停用詞**：
   - 將句子分詞（以空白切割或使用更精細 tokenizer）  
   - 過濾出不在 stop words 清單中的詞  
   - 再重新組合為句子字串或保留詞列表，視後續設計而定  
4. 確認處理後的文字仍然保留主要語意，並檢查幾筆範例。

### Requirements / Tools

- Python 文字操作（字串、正則表示式）
- 停用詞清單（NLTK 或 sklearn）

### Expected Output

✅ `X_train_clean`, `X_test_clean` — 已清洗與去除停用詞的文字資料

---

# Step 4. 文字轉向量（Tokenizer + Embedding）

> CNN / LSTM 在 Keras 中通常接收 **序列整數 ID** 並搭配 Embedding 層。  
> 這裡可視為一種學習式 Word2Vec / 文字向量表示。

### Objective

將清理後文字轉為固定長度的整數序列，並準備對應的 Embedding 輸入。

### Instructions

1. 設定超參數：
   - `max_words`：字典中保留的最多單字數（例如 20,000）
   - `max_len`：每則評論最大長度（字數）。可先看分佈（字數 histogram），選擇如 100、200 等。  
2. 使用 Keras 的 Tokenizer：
   - 在 **train 資料上** `fit_on_texts(X_train_clean)` 建立字典  
   - 不要在 test 上 fit，以避免資訊洩漏  
3. 將文字轉為整數序列：
   - `X_train_seq = tokenizer.texts_to_sequences(X_train_clean)`  
   - `X_test_seq = tokenizer.texts_to_sequences(X_test_clean)`  
4. 對序列進行補齊（padding）：
   - 使用 `pad_sequences` 將所有序列補到 `max_len`  
   - 建立 `X_train_pad`, `X_test_pad` 作為模型的最終輸入  
5. 準備 Embedding 層設定：
   - `input_dim` = 字典大小 + 1（`len(tokenizer.word_index) + 1`）  
   - `output_dim` = 嵌入維度（例如 100 或 128）  
   - `input_length` = `max_len`  

> 若想進階：可以選擇載入預訓練 Word2Vec / GloVe 向量並建立 Embedding matrix，但非必要。

### Requirements / Tools

- `keras.preprocessing.text.Tokenizer`
- `keras.preprocessing.sequence.pad_sequences`
- Keras Embedding layer 設定

### Expected Output

✅ `X_train_pad`, `X_test_pad`, `y_train`, `y_test`, `tokenizer` — 可直接餵給 CNN / LSTM 的序列資料與字典

---

# Step 5. 建立與訓練 CNN 模型（含 Dropout 比較）

### Objective

使用卷積神經網路（CNN）進行情緒分類，並比較加入高 Dropout（約 0.7）前後的訓練與測試表現，以觀察 overfitting 狀況。

### Instructions

1. 準備標籤格式：
   - 若輸出層使用單一神經元 + sigmoid：維持 `y_train`, `y_test` 為 0/1 整數即可  
2. 設計 **CNN baseline 模型**（無或少量 Dropout）：
   - 輸入層：對接 padded sequences  
   - Embedding 層：依前一節設定  
   - 一層或多層 1D Convolution（Conv1D）+ MaxPooling1D  
   - Flatten 或 GlobalMaxPooling1D  
   - Dense 隱藏層（例如 64 / 128 units, activation='relu'）  
   - 輸出層：1 unit, activation='sigmoid'（二元分類）  
3. 編譯模型：
   - loss：`binary_crossentropy`
   - optimizer：例如 `adam`
   - metrics：至少包含 `accuracy`  
4. 設定訓練參數：
   - `batch_size`：例如 64 或 128  
   - `epochs`：例如 5–10（可視跑的時間調整）  
   - `validation_split`：在 train 內切一小部分作為 validation（例如 0.1）  
5. 訓練 baseline CNN 並保留訓練紀錄（history）：
   - 用於後續繪製 Accuracy / Loss 曲線  
6. 設計 **CNN + Dropout 模型**：
   - 與 baseline 結構相似，但在：
     - Embedding 後或 Conv/Pooling 後  
     - Dense 隱藏層前/後  
     加入 Dropout 層，`rate ≈ 0.7`  
   - 其他設定（loss、optimizer、epochs 等）保持相同以利比較  
7. 同樣訓練 CNN+Dropout 模型並記錄 history。  

### Requirements / Tools

- Keras Sequential 或 Functional API  
- Conv1D、MaxPooling1D、GlobalMaxPooling1D / Flatten、Dense、Dropout、Embedding  
- 損失函數：`binary_crossentropy`  
- 優化器：`adam`  

### Expected Output

✅ `cnn_model_baseline`, `cnn_model_dropout`, `history_cnn_base`, `history_cnn_dropout` — 兩個訓練完成的 CNN 模型與訓練過程紀錄

---

# Step 6. 建立與訓練 LSTM 模型（含 Dropout 比較）

### Objective

使用 LSTM（長短期記憶）網路進行情緒分類，並比較加入高 Dropout（約 0.7）前後的表現。

### Instructions

1. 設計 **LSTM baseline 模型**：
   - 輸入層：padded sequences  
   - Embedding 層  
   - LSTM 層（例如 64 或 128 units），可決定是否回傳完整序列或最後狀態（通常最後狀態即可）  
   - Dense 隱藏層（可選）  
   - 輸出層：1 unit, activation='sigmoid'  
2. 編譯與訓練設定：
   - loss：`binary_crossentropy`
   - optimizer：例如 `adam`
   - metrics：`accuracy`  
   - batch_size、epochs、validation_split 建議與 CNN 模型保持一致  
3. 訓練 baseline LSTM 並保存 history。  
4. 設計 **LSTM + Dropout 模型**：
   - 在 Embedding 或 LSTM 之後加入 Dropout 層，`rate ≈ 0.7`  
   - 也可設定 LSTM 自帶的 `dropout` / `recurrent_dropout` 參數  
   - 其他設定與 baseline 相同  
5. 訓練 LSTM+Dropout 模型並保存 history。  

### Requirements / Tools

- Keras LSTM 層  
- Dropout 層或 LSTM 內建 dropout 參數  
- 與 Step 5 一致的訓練設定，方便比較  

### Expected Output

✅ `lstm_model_baseline`, `lstm_model_dropout`, `history_lstm_base`, `history_lstm_dropout` — 兩個訓練完成的 LSTM 模型與其訓練紀錄

---

# Step 7. 訓練過程的 Accuracy / Loss 視覺化

### Objective

根據作業要求，plot 出訓練過程中的 Accuracy 與 Loss 變化，並比較不同模型／Dropout 的學習行為與過擬合狀況。

### Instructions

1. 從各 `history` 物件中讀取：
   - `history.history["loss"]`
   - `history.history["val_loss"]`
   - `history.history["accuracy"]`
   - `history.history["val_accuracy"]`  
2. 針對下列模型分別繪圖（可一圖一模型，或一圖多曲線對比）：
   - CNN baseline
   - CNN + Dropout
   - LSTM baseline
   - LSTM + Dropout  
3. 對每個模型至少繪製兩張圖：
   - Loss vs. Epoch（訓練 / 驗證）
   - Accuracy vs. Epoch（訓練 / 驗證）  
4. 在圖上加上標題與圖例，說明是哪個模型與哪種資料（train / val）。  
5. 觀察：
   - 是否出現訓練 Accuracy 一路上升但驗證 Accuracy 不升反降的情況（overfitting）  
   - Dropout 是否讓驗證曲線較穩定  

### Requirements / Tools

- matplotlib（或 Keras 內建繪圖方式）  
- Accuracy / Loss 曲線繪製  

### Expected Output

✅ 一組圖像 — 顯示四種模型（CNN/LSTM × baseline/Dropout）的訓練與驗證 Loss / Accuracy 變化

---

# Step 8. 模型評估與 Accuracy 計算

### Objective

使用 test 資料評估 CNN 與 LSTM 模型的最終表現，並比較 Dropout 前後的 Accuracy。

### Instructions

1. 使用 `X_test_pad` 作為輸入資料，對每個模型進行預測：
   - CNN baseline
   - CNN + Dropout
   - LSTM baseline
   - LSTM + Dropout  
2. 預測輸出為機率（0–1 之間），需以閾值（threshold，例如 0.5）轉為 0/1 預測標籤。  
3. 與 `y_test` 比較，計算 Accuracy：
   - 使用 sklearn `accuracy_score` 或 Keras `evaluate`。  
4. 整理結果成表格或文字列出，例如：

   | Model              | Dropout | Test Accuracy |
   |--------------------|---------|---------------|
   | CNN baseline       | No      | 0.xxx         |
   | CNN dropout(0.7)   | Yes     | 0.xxx         |
   | LSTM baseline      | No      | 0.xxx         |
   | LSTM dropout(0.7)  | Yes     | 0.xxx         |

5. 簡短分析：
   - 哪種架構（CNN vs LSTM）在此資料上的表現較好？  
   - Dropout 是否提升了 test Accuracy 或減少 overfitting？  

### Requirements / Tools

- `model.evaluate` 或 `accuracy_score`  
- 0.5 閾值轉換為二元標籤  

### Expected Output

✅ 一個清楚列出四個模型在 test 集上 Accuracy 的比較結果表，以及對結果的文字說明

---

# Reporting / Visualization（報告與說明）

### What to include

- **資料前處理摘要**
  - 如何將 `stars` 轉為 0/1  
  - 是否有刪除缺失值、文字清理的規則  
- **模型架構概要**
  - CNN 與 LSTM 的層級結構與關鍵超參數  
  - Dropout 放置位置與 rate（0.7）  
- **訓練曲線觀察**
  - 各模型的 train / val Loss、Accuracy 曲線比較  
  - 哪些模型出現明顯 overfitting，Dropout 是否改善  
- **測試結果比較**
  - 四種模型（CNN/LSTM × baseline/Dropout）的 test Accuracy 表  
  - 對結果原因的推測（例如 LSTM 擅長長序列、CNN 擅長 n-gram 模式等）  
- **可能改進方向（選填）**
  - 調超參數（embedding 維度、max_len、batch_size、epochs）  
  - 使用預訓練詞向量（GloVe, Word2Vec）  
  - 調整 Dropout rate 或增加正則化手段  

✅ Output: 一段完整 markdown 報告，包含圖表與文字說明（顯示於 notebook）

---

# Technical Notes（技術備註）

- **Language:** Python  
- **Environment:** Jupyter Notebook  
- **Libraries:**
  - `pandas`, `numpy` — 資料處理  
  - `scikit-learn` — train/test split、metrics（accuracy）  
  - `tensorflow` / `keras` — 深度學習模型（Embedding, Conv1D, LSTM, Dropout 等）  
  - `matplotlib` — 訓練過程視覺化  
- **Reproducibility:**
  - 在程式一開始設定隨機種子（例如 numpy、tensorflow）  
  - 固定 `random_state` 於 train/test split  
- **Coding style:**
  - 可將重複步驟（例如建立模型）封裝為函式  
  - 在各區段使用 markdown 標題與註解，說明目的與結果  

---

# Deliverables（最終成果）

- **一個 Jupyter Notebook**，內容包含：
  - 資料載入與前處理（含標籤建立與 80/20 切分）  
  - 文字清理與停用詞處理  
  - 文字向量化（Tokenizer + Embedding）  
  - CNN 與 LSTM 模型（baseline 與 Dropout 版本）之建立與訓練  
  - 訓練過程 Accuracy / Loss 圖  
  - test 集 Accuracy 計算與比較表  
  - 簡短文字報告與結論  
