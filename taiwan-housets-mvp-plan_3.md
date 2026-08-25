# TaiwanTS MVP：零成本假設驗證計畫

## 一句話目標

用五本 Colab notebook，在 8 週內回答：「衛星影像特徵能否提升台灣房價預測？」基礎設施成本為零，VLM token 支出 US$0.3 — 16（視 NB0 的粒度決策而定，見第 9 節）。

---

## 0. 設計原則

這份計畫只追求一件事——走到 go/no-go 決策點。不做產品化、不做 UI 打磨、不買任何付費服務。所有基礎設施都用免費方案，唯一的現金支出是 VLM API token：US$0.3（鄉鎮市區粒度）到 US$16（村里全樣本），詳見第 9 節。

如果一個人全職做，8 週。兩個人，4-5 週。

---

## 1. 核心假設與成功標準

**假設：** Sentinel-2 遙測指數（NDVI、NDBI）加入 XGBoost 後，房價預測 MAPE 可降低。

**量化成功標準（所有 % 為絕對百分點）：**

- 六都測試集整體 MAPE 改善 ≥ 2 個絕對百分點（例：baseline MAPE 15% → 影像增強 ≤ 13%）
- 至少 3 個都市區達到統計顯著改善（paired t-test, p < 0.05；pairing 單位為「分析單元 × 季」，非 × 月——見第 5.3 節說明）
- SHAP 分析中，遙測特徵至少有 1 項進入前 10 重要特徵

**質性成功標準：**

- VLM 能從 Sentinel-2 影像對中辨識出台灣都市變遷（新建案、捷運、綠地消失）
- 產出 3-5 個「影像變遷 × 房價跳升」的 compelling story，可用於 demo

---

## 2. 範圍設定

| 維度 | MVP 規格 | 理由 |
|---|---|---|
| 空間 | 六都（台北、新北、桃園、台中、台南、高雄） | 交易量佔全台 ~75% |
| 粒度 | **待定：村里 或 鄉鎮市區**（見第 5.0 節） | 取決於交易筆數密度分析結果 |
| 時間 | 2017Q1 — 2024Q4（8 年） | Sentinel-2 L2A 全球始於 2017 |
| 房價頻率 | **待定：月度 或 季度** | 取決於交易筆數密度分析結果 |
| 影像頻率 | 季度合成 | 降低台灣高雲覆蓋影響 |
| 建物類型 | 僅電梯大樓 | 交易量最大、同質性最高，先驗證再擴展 |

粒度與頻率不在開工前鎖死——第一件事是分析 Y 的密度分布，再決定。

---

## 3. 全免費工具棧

| 功能 | 工具 | 費用 | 備註 |
|---|---|---|---|
| 影像處理 | Google Earth Engine | $0 | 非商業專案，每月免費運算配額 |
| 運算環境 | Google Colab Free | $0 | T4 GPU（本案不需 GPU），12hr/session |
| 備援運算 | Kaggle Notebooks | $0 | 30 GPU hrs/week |
| 資料庫 | DuckDB（嵌入式） | $0 | 直接在 Colab 裡跑 SQL + spatial join |
| 空間處理 | geopandas + shapely | $0 | pip install |
| 建模 | XGBoost + scikit-learn | $0 | CPU 即可 |
| VLM 影像解讀 | Claude Sonnet Batch API | **US$0.3 — 16** | 全樣本；區間取決於 NB0 的粒度決策（見第 9 節） |
| 程式碼管理 | GitHub | $0 | Public 或 Free private repo |
| 資料儲存 | Google Drive | $0 | 15 GB 免費 |
| Demo 部署 | Streamlit Community Cloud | $0 | 從 GitHub 直接部署 |
| 地圖視覺化 | Folium + Kepler.gl | $0 | 嵌入 Streamlit |

**Total infrastructure cost: ≈ US$0**（基礎設施全部免費；唯一現金支出是上表的 VLM token，最壞情況 US$16）

---

## 4. 資料來源一覽

### 4.1 目標變數：實價登錄

- 來源：內政部不動產交易實價查詢網 plvr.land.moi.gov.tw
- 格式：CSV 批次下載，每月 1/11/21 日更新
- 費用：$0

清整規則：

- 僅保留「建物型態 = 住宅大樓(11層含以上有電梯)」
- 排除：總價 = 0、單價 < 5 萬/坪或 > 250 萬/坪、備註含「親友」「法拍」「特殊」
- 拆解車位交易：總價減車位價、坪數減車位坪數
- 聚合單位和統計量待 Notebook 0 決定（見第 5.0 節）

### 4.2 衛星影像

| 衛星 | 波段 | 原生解析度 | 重訪 | 取得方式 | 費用 |
|---|---|---|---|---|---|
| Sentinel-2 L2A | B2, B3, B4, B8 | 10m | 5 天 | GEE `COPERNICUS/S2_SR_HARMONIZED` | $0 |
| Sentinel-2 L2A | B11, B12（SWIR） | **20m** | 5 天 | 同上，GEE 自動重採樣至 10m | $0 |
| Sentinel-1 SAR | VV, VH | 10m | 12 天 | GEE `COPERNICUS/S1_GRD`（雲備援） | $0 |

注意：NDBI 和 MNDWI 公式用到 B11（SWIR），原生解析度為 20m。GEE 在 reduceRegion 時會自動用雙線性內插重採樣至請求解析度，不會報錯，但實際有效空間解析度為 20m，非 10m。在報告中應如實說明：「NDVI 有效解析度 10m；NDBI/MNDWI 有效解析度 20m，經重採樣對齊」。

影像不需要下載。在 GEE 上直接運算，只輸出分析單元級統計值 CSV。

### 4.3 輔助特徵

| 資料 | 來源 | 粒度 | 費用 |
|---|---|---|---|
| 交通站點 | data.gov.tw | 點位 + 通車日期 | $0 |
| 學校、醫院、公園 | data.gov.tw | 點位 | $0 |
| 商業 POI | OpenStreetMap Overpass API | 點位 | $0 |
| 人口、戶數 | 內政部戶政司 | 村里 / 月 | $0 |
| 所得中位數 | 財政部 | 鄉鎮市區 / 年 | $0 |
| 土地使用分區 | 全國土地使用分區查詢系統 | Shapefile | $0 |
| 村里邊界 | 政府開放資料 | Shapefile | $0 |

---

## 5. Notebook 架構

整個 MVP 是五本 Colab notebook（含 Notebook 0），存在一個 GitHub repo 裡。

```
taiwants-mvp/
├── notebooks/
│   ├── 00_density_check.ipynb    # Y 變數密度分析 → 決定粒度
│   ├── 01_gee_imagery.ipynb      # Sentinel-2 → 遙測指數 CSV
│   ├── 02_data_merge.ipynb       # 實價登錄 + 遙測 + POI + 社經 → 大表
│   ├── 03_modeling.ipynb         # XGBoost A vs B → MAPE 比較
│   └── 04_vlm_narratives.ipynb   # Claude API 解讀全樣本影像 → 變遷故事
├── data/
│   ├── raw/                     # 實價登錄 CSV、村里 shapefile
│   ├── processed/               # GEE 輸出、合併後大表
│   └── vlm_outputs/             # VLM 結構化結果 JSON
├── app/
│   └── streamlit_app.py         # Demo dashboard
├── requirements.txt
└── README.md
```

### 5.0 Notebook 0：Y 變數密度分析（最重要的前置步驟）

這步決定整個實驗能不能做。在寫任何模型之前，必須先回答：選定的空間粒度 × 時間粒度下，每個 cell 有多少筆交易？

輸入：實價登錄全資料（清整後）+ 村里邊界
輸出：密度分析報告 → 決定最終粒度

分析內容：

1. 對每個「村里 × 月」cell，統計電梯大樓交易筆數
2. 計算分布：中位數、P25、P75、零筆 cell 佔比
3. 對每個「村里 × 季」cell，重複上述統計
4. 對每個「鄉鎮市區 × 月」cell，重複上述統計
5. 六都分別報告（台北市村里交易密度會遠高於台南）

決策規則：

```
若「村里 × 月」中 n ≥ 5 筆的 cell 佔比 > 50%
  → 採用「村里 × 月」
  → 中位單價有統計意義

若「村里 × 月」稀疏但「村里 × 季」中 n ≥ 5 佔比 > 50%
  → 退到「村里 × 季」
  → 房價頻率對齊影像頻率，消除結構性重複問題（見 5.3 節）

若「村里 × 季」仍稀疏
  → 退到「鄉鎮市區 × 月」（六都約 150-170 個區）
  → 犧牲空間解析度換取 Y 的穩定性

無論哪種：n = 1 的 cell 必須標記或排除，因為「中位數」在 n=1 時就是那一筆本身，
沒有統計平滑效果，純粹是單筆交易的噪音。
```

預期結果：根據台灣不動產市場實務，「村里 × 季」或「鄉鎮市區 × 月」較可能是 sweet spot。以下各節用「分析單元」指代最終選定的空間粒度。

### 5.1 Notebook 1：GEE 影像處理

輸入：分析單元邊界（村里或鄉鎮市區 Shapefile，上傳為 GEE Asset）
輸出：`unit_remote_sensing.csv`（分析單元 × 季 × 遙測指數）

處理步驟：

1. 上傳六都分析單元邊界為 GEE FeatureCollection
2. 對每個「分析單元 × 季度」組合：
   - 取該季所有 Sentinel-2 L2A 影像
   - SCL band 雲遮罩（保留值 4=植被、5=裸地、6=水體；排除 1=飽和/缺陷、3=雲影、8=中機率雲、9=高機率雲、10=卷雲、11=冰雪；值 2=暗區域 和 7=未分類 可選擇性保留或排除，建議先排除再視有效像素覆蓋率決定是否放寬）
   - 取各波段中位數合成
3. 計算遙測指數：
   - NDVI = (B8 - B4) / (B8 + B4)　→ 植被覆蓋（有效解析度 10m）
   - NDBI = (B11 - B8) / (B11 + B8)　→ 建築密度（有效解析度 20m，B11 經重採樣）
   - MNDWI = (B3 - B11) / (B3 + B11)　→ 水體（有效解析度 20m，同上）
4. 以分析單元為單位取空間平均值
5. 計算年際差分：ΔNDVI = NDVI(Q) - NDVI(Q-4)，ΔNDBI 同理
6. 記錄每個 cell 的有效像素佔比（valid_pixel_ratio），供後續品質篩選
7. 輸出為 CSV 到 Google Drive

GEE 耗時估算：若粒度為村里（~7,800 個 × 32 季 ≈ 25 萬次 reduceRegion），分批提交約 2-4 小時。若粒度為鄉鎮市區（~170 個 × 32 季 ≈ 5,400 次），幾分鐘內完成。

### 5.2 Notebook 2：資料整合

輸入：實價登錄 CSV + 遙測 CSV + POI + 社經 + 分析單元邊界
輸出：`modeling_table.parquet`（分析單元 × 時間粒度 × 所有特徵）

處理步驟：

1. 清整實價登錄（依 4.1 規則），按 NB0 決定的粒度聚合中位單價
2. DuckDB spatial join：交易地址 → 分析單元歸屬
3. POI 特徵工程：
   - 分析單元重心 500m/1km 半徑內各類 POI 數量
   - 距最近捷運站/學校/公園距離
   - 捷運站「通車前後」二元特徵
4. 社經資料 join（鄉鎮市區級，按分析單元所屬鄉鎮 mapping）
5. 遙測指數 join（季度頻率——若房價粒度也是季度，則 1:1 對齊；若房價粒度為月度，則每季 3 個月共用同一組遙測值，但須在模型評估時處理此結構性重複，見 5.3 節）
6. 土地使用分區比例（住宅/商業/工業/農業面積佔比）
7. 時間特徵：季、年、季 sin/cos encoding；**若 NB0 選定月度粒度，另加月與月 sin/cos**
8. 排除 valid_pixel_ratio < 0.3 的 cell（遙測值不可靠）
9. 排除交易筆數 n < 3 的 cell（Y 過於不穩定）
10. 輸出合併大表為 parquet

### 5.3 Notebook 3：模型實驗

輸入：`modeling_table.parquet`
輸出：實驗結果報告（metrics + SHAP plots + 比較表）

實驗設計：

```
特徵集 A（baseline，無影像）：
  建物屬性（坪數中位、屋齡中位、樓層中位）
  + POI（各類數量、距離）
  + 社經（人口密度、所得、老化指數）
  + 土地使用分區比例
  + 時間特徵

特徵集 B（加影像）：
  特徵集 A
  + NDVI, NDBI, MNDWI（當季值）
  + ΔNDVI, ΔNDBI（年際差分）
```

資料分割（時間切分，避免洩漏）：

- Train：2017Q1 — 2022Q4
- Validation：2023Q1 — 2023Q2
- Test：2023Q3 — 2024Q4

模型流程：

1. XGBoost + 特徵集 A → 記錄 MAPE_A
2. XGBoost + 特徵集 B → 記錄 MAPE_B
3. LightGBM 重複上述（交叉驗證）
4. Paired t-test：MAPE_A vs MAPE_B
5. 按都市分群報告（台北 / 新北 / 桃園 / 台中 / 台南 / 高雄）
6. SHAP 分析：影像特徵的重要性排名和方向性
7. 錯誤分析：哪些分析單元影像特徵幫助最大 → 輸入 Notebook 4

關於季度影像 × 月度房價的結構性重複：

若 NB0 最終採用「分析單元 × 月」，則每季 3 個月共用同一組遙測值。這意味著遙測特徵的獨立觀測只有 32 個（季），卻對應 96 個月的 Y。影響有二：

- SHAP 重要性：**偏誤方向不確定**。樣本重複會加重該特徵的分裂權重（偏高），但季內無變異又使它無法解釋月內波動（偏低），淨效果視資料而定。因此只解讀影像特徵的**符號與排名**，不解讀量級——這也是為什麼成功標準寫的是「進入前 10」而非某個 SHAP 數值。
- 統計檢定：paired t-test 的 pairing 單位應為「分析單元 × 季」而非「× 月」——先將同季 3 個月的預測誤差平均，再做 pair。否則同季 3 個 pair 不獨立，p-value 會被人為壓低。

若 NB0 結果為「分析單元 × 季」，此問題自動消失——遙測特徵和房價 1:1 對齊。

### 5.4 Notebook 4：VLM 都市敘事（全樣本）

輸入：全部分析單元 + Sentinel-2 影像對
輸出：結構化變遷描述 JSON（全樣本）+ demo 素材

Token 成本真實計算（成本隨 NB0 的粒度決策而變，故列兩個情境）：

```
Anthropic 影像 token 公式：(寬 × 高) / 750

256×256 影像 = 87 tokens / 張
一對影像（2017 + 2024）= 174 input tokens
加 system prompt + user prompt ≈ 300 input tokens / 對
output（JSON）≈ 200 tokens / 對

Claude Sonnet Batch API（50% off）：
  input  $1.50 / MTok
  output $7.50 / MTok

情境 A — 粒度＝村里（六都 7,851 個，最壞情況）：
  input  = 7,851 × 300 = 2.36M tokens × $1.50/M = $ 3.53
  output = 7,851 × 200 = 1.57M tokens × $7.50/M = $11.78
  Total ≈ US$15.3

情境 B — 粒度＝鄉鎮市區（六都 ~170 個，較可能）：
  input  = 170 × 300 = 51K tokens × $1.50/M = $0.08
  output = 170 × 200 = 34K tokens × $7.50/M = $0.26
  Total ≈ US$0.34

prompt 驗證（100 對）：≈ US$0.20

結論：兩個情境都便宜到不構成決策因素。沒有理由只挑子集。
注意成本有 77% 在 output——真要壓成本是縮短 narrative_zh，不是壓縮影像。
```

步驟：

1. **產圖：導出大 mosaic，本機裁切**（不要一個單元一個 GEE task）

   全樣本若是村里粒度，需要 7,851 × 2 ≈ 15,700 張小圖。用 `getThumbURL` 會撞 rate limit，用 `Export.image.toDrive` 則是 15,700 個 task，會塞爆 GEE 的任務佇列。正確做法：

   - 在 GEE 導出 **12 張大 mosaic GeoTIFF**（六都 × 2017/2024 兩期 RGB 中位數合成），共 12 個 export task
   - 下載後在 Colab 用 `rasterio` + 分析單元邊界，按 window 在本機裁出 15,700 張 256×256 圖
   - 這樣 GEE 端只有 12 個任務，裁圖是本機幾分鐘的事

2. **全樣本呼叫 Claude Sonnet Batch API（注意單批上限）**

   Batch API 上限為 **100,000 requests 或 256 MB / 批**，request 數量不是問題，**payload 大小是**：

   - 256×256 衛星 PNG ≈ 100 KB，base64 後 ≈ 133 KB，每個 request 帶兩張圖 ≈ 266 KB
   - → 256 MB 只裝得下約 **960 個 request**，7,851 對要拆成 **9 批**
   - 改存 **JPEG（q85，每張 ~25 KB）可壓到 2-3 批**，對 10m 影像的判讀品質沒有可見影響
   - 實作上直接寫成「累加 payload 大小，逼近 200 MB 就切一批」的迴圈，別手動算

```
Prompt 範本：
你是都市規劃分析師。請比較這兩張同一地點不同時期的衛星影像，
以 JSON 格式回答（不要輸出任何其他內容）：
{
  "building_density_change": "increased/decreased/stable",
  "green_space_change": "increased/decreased/stable",
  "new_infrastructure": ["捷運站", "高架道路", ...] 或 [],
  "land_use_shift": "農地轉住宅 / 工業轉商業 / 無明顯變化",
  "confidence": 0.0-1.0,
  "narrative_zh": "一段 50 字以內的中文描述"
}
```

3. 收集全樣本 JSON，可額外做：
   - 統計各類變遷的空間分布（哪些區域都市化最快）
   - 將 VLM 結果反寫為分類特徵，測試是否對模型有額外增量
   - 從高 confidence + 高房價漲幅的交集中挑出 3-5 個最佳 demo 案例

---

## 6. Demo：Streamlit App

一頁式互動 dashboard，部署在 Streamlit Community Cloud（免費）。

功能：

1. 六都地圖（Folium），分析單元著色 = 房價漲幅
2. 點擊分析單元 → 顯示：
   - 房價時間序列折線圖
   - NDVI/NDBI 時間序列疊加
   - XGBoost 預測 vs 實際
   - SHAP waterfall（該單元的特徵貢獻）
3. VLM 變遷總覽：全樣本變遷類型的熱力圖
4. Highlight 區：3-5 個最佳 demo 案例
   - 2017 vs 2024 衛星影像並排
   - VLM 產出的變遷描述
   - 房價漲幅對比

部署方式：GitHub repo 連結 Streamlit Cloud，push 即部署。

---

## 7. 時程（8 週）

| 週 | 任務 | Notebook | 交付物 |
|---|---|---|---|
| W1 | 下載清整實價登錄 + **Y 密度分析** | **NB0** | density_report + **粒度決策** |
| W2 | 收集分析單元邊界、POI、社經資料 | NB2 前置 | raw data in /data/raw/ |
| W3 | GEE pipeline：分析單元遙測指數 **+ 導出 12 張 RGB mosaic** | NB1 | unit_remote_sensing.csv **+ mosaic GeoTIFF** |
| W4 | 資料整合：merge 全部資料源 | NB2 | modeling_table.parquet |
| W5 | 基準模型訓練 + 影像增量實驗 | NB3 | MAPE 比較表 |
| W6 | SHAP 分析 + 統計檢定（季度 pairing） | NB3 | 實驗報告（假設驗證結果） |
| W7 | 本機裁圖 + VLM 全樣本解讀（分批提交） | NB4 | vlm_outputs/*.json（全量） |
| W8 | Streamlit demo + 文件整理 | app/ | 可展示的 live demo |

W1 是關鍵節點：若密度分析顯示即使退到「鄉鎮市區 × 季」仍然稀疏，需要重新評估「僅電梯大樓」的限制，考慮納入公寓和透天。

---

## 8. 執行細節備忘

### GEE 注意事項

- 分析單元數量大時（如村里 ~7,800），一次 reduceRegion 會 timeout → 按縣市分批，每批一個 ee.batch.Export.table.toDrive
- **NB4 的裁圖絕對不要一個單元一個 task。** GEE 對排隊中的任務數有上限，15,700 個 export 會直接卡死。導出 12 張大 mosaic 再用 rasterio 本機裁切（見 5.4 步驟 1）
- Sentinel-2 L2A 的 SCL band 雲遮罩參考值：
  - 保留：4（植被）、5（裸地）、6（水體）
  - 排除：1（飽和/缺陷像素）、3（雲影）、8（中機率雲）、9（高機率雲）、10（卷雲）、11（冰/雪）
  - 自行決定：2（暗區域像素）、7（未分類像素）——建議先排除，若有效像素覆蓋率不足再放寬
- NDBI 和 MNDWI 用到 B11（SWIR，原生 20m），GEE 會自動重採樣但不會提高實際解析度。在村里面積小於 ~0.04 km²（200m × 200m）的情況下，NDBI/MNDWI 的空間平均可能只基於少數幾個獨立 20m 像素，需注意品質
- 台灣平均雲覆蓋率高（~60%），某些季度某些分析單元可能無有效像素 → 用前後季內插或標記為 NaN
- 若某分析單元全年雲覆蓋嚴重，引入 Sentinel-1 SAR 的 VV/VH backscatter 作為備援（SAR 不受雲影響）

### 實價登錄清整陷阱

- 「交易筆棟數」欄位格式如「土地1建物1車位1」，需正則解析
- 車位價格有時包在總價裡，有時拆開 → 統一處理為「總價 - 車位價」
- 同一筆交易的土地、建物可能分成多列 → 以「編號」欄位 groupby
- 2021 年 7 月實價登錄 2.0 上線，前後欄位名稱和格式有差異
- n=1 的 cell 看起來有中位數，其實就是那一筆交易本身，沒有平滑效果——標記或排除

### Colab 使用技巧

- 長時間 GEE 任務用 ee.batch.Export（背景執行），不佔 Colab session
- 中間結果存 Google Drive，避免 session 斷線遺失
- DuckDB 在 Colab 中用 `pip install duckdb --break-system-packages` 安裝
- geopandas spatial join 用 DuckDB spatial extension 會更快

### VLM 使用備註

- 影像裁切為 256×256 即可，token 消耗極低（每張 ≈ 87 tokens）
- **存成 JPEG（q85）而非 PNG。** 不是為了省 token（token 只看像素尺寸），是為了縮小 base64 payload，讓單批塞得下更多 request
- **Batch API 單批上限：100,000 requests 或 256 MB。** 全樣本會先撞到 256 MB，需分批提交（PNG ~9 批、JPEG ~3 批）
- 用 Batch API（非即時），成本再打五折
- system prompt 簡短，強制 JSON only 輸出
- 先用 10 對影像測試 prompt 品質，確認 JSON 格式和中文描述品質後再全量提交
- 全樣本跑完後，VLM 結果本身可作為額外分類特徵回饋給 NB3 做 ablation

---

## 9. 預算明細

| 項目 | 費用（USD） | 說明 |
|---|---|---|
| Google Earth Engine | $0 | 非商業用途免費配額 |
| Google Colab | $0 | Free tier，XGBoost 僅需 CPU |
| Kaggle Notebooks | $0 | 備援運算環境 |
| 衛星影像 | $0 | Sentinel-2 完全免費 |
| 資料來源 | $0 | 全部政府開放資料 |
| DuckDB / geopandas | $0 | 開源工具 |
| GitHub | $0 | Free private repo |
| Google Drive | $0 | 15 GB 免費額度 |
| Streamlit Community Cloud | $0 | 免費部署 |
| Claude Sonnet Batch API（情境 B：鄉鎮市區 ~170 個） | **≈ US$0.34** | 全樣本 VLM 解讀 |
| Claude Sonnet Batch API（情境 A：村里 7,851 個） | **≈ US$15.3** | 全樣本 VLM 解讀（最壞情況） |
| **Total** | **US$0.3 — 16** | **約台幣 10 — 500 元，取決於 NB0 的粒度決策** |

Token 計算明細：256×256 影像 = (256×256)/750 ≈ 87 tokens/張。一對影像 + prompt ≈ 300 input tokens，JSON 回應 ≈ 200 output tokens。Sonnet Batch API 定價 $1.50 / $7.50 per MTok。

- 村里全樣本 7,851 對：2.36M input × $1.50 = $3.53；1.57M output × $7.50 = $11.78 → **$15.31**
- 鄉鎮市區全樣本 170 對：51K input × $1.50 = $0.08；34K output × $7.50 = $0.26 → **$0.34**

無論哪個情境，成本都不是這個專案的約束條件——真正的約束是影像萃取（見 5.4 步驟 1）和 Batch API 的 256 MB 單批上限（見 5.4 步驟 2）。

---

## 10. 風險與緩解

| # | 風險 | 機率 | 影響 | 緩解 |
|---|---|---|---|---|
| **1** | **Y 變數樣本稀疏：村里 × 月 × 僅電梯大樓 → 大量 cell 只有 0-2 筆交易** | **高** | **實驗無法進行** | **NB0 先做密度分析，決定是否退到村里 × 季、鄉鎮市區 × 月、或放寬建物類型** |
| 2 | 10m/20m 解析度抓不到建物級變化 | 中 | 核心假設失敗 | NDBI 差分抓「區域級」建設擴張；B11 實際 20m 的限制如實記載 |
| 3 | 台灣雲覆蓋導致遙測缺值 | 高 | 特徵不完整 | Sentinel-1 SAR 備援 + 前後季內插 + NaN + valid_pixel_ratio 篩選 |
| 4 | 季度遙測 × 月度房價的結構性重複 | 中 | SHAP 偏差、p-value 虛低 | 統計檢定以「分析單元 × 季」為 pairing 單位；或直接採用季度房價 |
| 5 | GEE 配額或**任務數**不夠（全樣本 15,700 張裁圖尤其危險） | 中 | pipeline 跑不完 | reduceRegion 按縣市分批；裁圖改用「導出 12 張大 mosaic + 本機 rasterio 裁切」避開任務數上限（見 5.4）；配額按月重置 |
| 6 | Colab session 中斷 | 中 | 工作遺失 | 中間結果存 Drive；GEE 用 batch export |
| 7 | 實價登錄資料品質差 | 高 | Y 有噪音 | 嚴格過濾 + 中位數 + 排除 n < 3 的 cell |
| 8 | VLM 看不懂 10m Sentinel-2 | 中 | demo 故事不好 | 全樣本跑完從高 confidence 子集選；可換用 Google Earth 歷史截圖補強 |
| 9 | 村里邊界歷年調整 | 低 | 時序不一致 | 以 2020 年邊界為基準，歷史資料 spatial join 重新歸屬 |

第 1 項是最大風險。若六都電梯大樓在「村里 × 季」都撐不起 n ≥ 5 的門檻，這個題目在台灣可能根本做不到村里粒度——但鄉鎮市區粒度仍然可行，只是差異化優勢變小。NB0 的結果直接決定後續所有設計。

---

## 11. 決策樹

```
NB0 密度分析
  │
  ├─ 村里 × 月 n≥5 佔比 > 50% → 採用村里 × 月（最理想，但機率低）
  ├─ 不夠 → 村里 × 季 n≥5 > 50% → 採用村里 × 季
  ├─ 還不夠 → 鄉鎮市區 × 月 n≥5 > 50% → 採用鄉鎮市區 × 月
  └─ 再不夠 → 放寬建物類型（納入公寓）→ 重跑 NB0
         │
         └─ 仍不夠 → 停止。台灣交易量不支撐此粒度。
                     ↓
          轉向：用 VLM 做「都市變遷監測報告」
          賣政府規劃單位，不做房價預測。

             ↓ 粒度確定後

         MAPE 改善 ≥ 2 個絕對百分點？
          ╱                    ╲
        是                      否
        │                       │
  SHAP 前 10 有影像特徵？    改善 0-2pp？
     ╱      ╲                ╱       ╲
   是        否             是        否
   │         │              │         │
  Phase 2    重新設計       轉向       停止
  產品化     影像特徵       都市變遷
  API+融資                  監測 SaaS
```

Phase 2 方向（若成功進入）：

- 導入 Planet Labs 3m 日更影像 → 精度 + 頻率提升
- 包裝為估價 API（RESTful），按查詢次數計費
- 目標客戶：銀行鑑價部門（38 家本國銀行）、估價師事務所、大型仲介
- 種子輪融資規模：TWD 5-10M

---

## 12. 第一天就做的事

1. 註冊 Google Earth Engine 非商業帳號（需審核 1-3 天）
2. 從 plvr.land.moi.gov.tw 下載六都 2017-2024 實價登錄全資料
3. 從 data.gov.tw 下載村里邊界 Shapefile
4. 開 GitHub repo，建好目錄結構
5. 開一本 Colab notebook，`pip install duckdb geopandas`，開始做 NB0 密度分析

不需要等任何審核、購買、或團隊到位。一個人、一台筆電、今天就能開始。第一天結束時你就知道粒度能做到多細。
