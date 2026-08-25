# TaiwanTS MVP

用五本 Colab notebook 驗證一個假設：**Sentinel-2 遙測指數（NDVI、NDBI）加入 XGBoost 後，台灣房價預測 MAPE 能否降低。**

完整規格見 [`taiwan-housets-mvp-plan_3.md`](taiwan-housets-mvp-plan_3.md)。基礎設施成本 $0，唯一現金支出是 NB4 的 VLM token（US$0.3 – 16）。

## 目前進度

| Notebook | 內容 | 狀態 |
|---|---|---|
| [`00_density_check`](notebooks/00_density_check.ipynb) | Y 變數密度分析 → **決定粒度** | ✅ 可執行 |
| [`01_gee_imagery`](notebooks/01_gee_imagery.ipynb) | Sentinel-2 → 遙測指數 CSV | ⬜ 骨架 |
| [`02_data_merge`](notebooks/02_data_merge.ipynb) | 實價登錄 + 遙測 + POI + 社經 → 大表 | ⬜ 骨架 |
| [`03_modeling`](notebooks/03_modeling.ipynb) | XGBoost A vs B → MAPE 比較 | ⬜ 骨架 |
| [`04_vlm_narratives`](notebooks/04_vlm_narratives.ipynb) | Claude 解讀影像對 → 變遷故事 | ⬜ 骨架 |
| [`app/streamlit_app.py`](app/streamlit_app.py) | Demo dashboard | ⬜ 骨架 |

## 先跑 NB0

NB0 決定整個 MVP 的空間與時間粒度，**其他 notebook 都從它的輸出讀粒度，不要硬編**。

```bash
pip install -r requirements.txt
jupyter lab notebooks/00_density_check.ipynb
```

或直接上傳到 Colab 執行（notebook 會自動掛載 Drive）。

NB0 全自動，**不需要手動下載任何資料**：

- 實價登錄：從 `plvr.land.moi.gov.tw` 抓 36 個發布季（106S1–114S4），只解出六都買賣主檔
- 村里／鄉鎮市區母體與戶數：內政部戶政司開放資料 API（`ODRP019`）

執行時間約 15–35 分鐘，主要花在下載。中斷可直接重跑，已完成的季會跳過。

### 產出

| 檔案 | 用途 |
|---|---|
| `data/processed/density_decision.json` | 機器可讀的粒度決策，NB1–NB4 讀這個 |
| `data/processed/density_report.md` | 人看的密度報告 |
| `data/processed/lvr_clean.parquet` | 清整後的實價登錄，NB2 直接重用 |
| `data/processed/figs/*.png` | 三張圖表 |

## 粒度決策規則（計畫 §5.0）

依序檢查，第一個讓「n ≥ 5 的 cell 佔比 > 50%」的方案勝出：
村里×月 → 村里×季 → 鄉鎮市區×月 →（全不過）放寬建物型態重跑。

**佔比的分母含零交易 cell**（單元母體 × 期數）。只算有交易的 cell 會嚴重高估密度。

### 村里粒度沒有精確解

實價登錄批次資料**沒有村里欄位、也沒有座標**，只有鄉鎮市區與門牌。精確的村里歸屬需要
地址 geocoding，那是 NB2 的工作。NB0 用三種方式在不做 geocoding 的前提下逼近：

- **A 理論上限** — 嚴格數學上界（任何空間分布都不可能超過）。上限若 ≤ 50%，村里粒度確定出局
- **B 戶數比例分配** — 「完全無空間聚集」虛無情境
- **C 街道聚集情境** — 用門牌路段的實際聚集程度模擬

恆有 A ≥ C。**決策以 A 為準**，因為只有它不會因估計方法失準而翻案。鄉鎮市區粒度則是精確計算。

## 資料來源的兩個坑

NB0 已處理，但改動清整規則時要留意：

1. **不要用「預售」比對備註來排除交易。** 實價登錄 2.0 的制式備註是「預售屋、或土地及建物分件登記案件；」——兩件事包在同一句，而絕大多數是後者（一般成屋的登記方式）。用「預售」比對會誤殺約 41% 的成屋交易。真正的預售屋另存於 `*_lvr_land_b.csv`，NB0 根本沒讀。
2. **檔名的「季」是發布季不是交易季。** `106S1` 內含的是民國 105 年 10–12 月登記的案件。下載範圍必須比分析期間寬，再用 `交易年月日` 過濾。

另外，實價登錄 2.0（民國110年7月）前後有欄位改名（`車位移轉總面積(平方公尺)` → 去掉括號），
由 `COL_ALIASES` 統一；若未來又改名，`load_one()` 會印出明確警告而不是丟 KeyError。

## 目錄

```
├── notebooks/     # 五本 notebook
├── data/
│   ├── raw/       # 下載的原始資料（不進版控）
│   ├── processed/ # 清整結果、密度報告、決策 JSON
│   └── vlm_outputs/
├── app/           # Streamlit demo
├── docs/          # NB0 的產生腳本
└── taiwan-housets-mvp-plan_3.md
```

`docs/gen_00_density_check.py` 是 NB0 的產生腳本。改 NB0 時可以直接改 `.ipynb`，
但若要大幅重構，改腳本再重新產生比較好管理（notebook 的 JSON diff 很難讀）。
