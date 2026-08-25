# -*- coding: utf-8 -*-
"""產生 notebooks/00_density_check.ipynb"""
import hashlib, json, pathlib

CELLS = []
def md(s):   CELLS.append(("markdown", s.strip("\n")))
def code(s): CELLS.append(("code", s.strip("\n")))

# ============================== 標題 ==============================
md(r"""
# NB0 — Y 變數密度分析

**這本 notebook 決定整個 MVP 的空間與時間粒度。** 在寫任何模型之前，必須先回答：

> 在選定的「空間單元 × 時間單元」下，每個 cell 有多少筆電梯大樓交易？

規格來源：`taiwan-housets-mvp-plan_3.md` 第 5.0 節。決策規則（門檻 n ≥ 5）：

| 順位 | 方案 | 通過條件 |
|---|---|---|
| 1 | 村里 × 月 | n≥5 的 cell 佔比 > 50% |
| 2 | 村里 × 季 | 同上 |
| 3 | 鄉鎮市區 × 月 | 同上 |
| 4 | 放寬建物型態（納入華廈、公寓）後重跑 | 同上 |
| — | 全部不過 | 停止，轉向都市變遷監測 |

**佔比的分母是所有 cell，包含零交易的 cell**（單元母體 × 期數），不是只算有交易的 cell。這點很重要：只看有交易的 cell 會嚴重高估密度。

### 分母的雙口徑

「單元母體」還有第二個問題：六都有相當比例的區（山地、偏鄉）八年來一筆電梯大樓交易都沒有，它們在 NB2（§5.2 步驟 9 排除 n<3）本來就會被丟掉，**不管選哪個粒度都不會進模型**。

所以每個指標都算兩次：**全母體**（全部鄉鎮市區，計畫字面讀法，**主口徑**）與 **子母體**（`_active`，全期至少一筆交易的區）。這不改變四個方案的排名，只改變它們有沒有越過 50% 那條線——而 50% 是計畫裡拍出來的門檻，本身沒有統計學依據。

**兩個口徑給出相反結論時，決策階梯判 `CONFLICT` 就地停住、不自動選邊**，交人工拍板。細節見 §5.0。

---

### 村里粒度的資料限制（重要）

實價登錄批次資料**沒有村里欄位、也沒有座標**，只有鄉鎮市區與門牌地址。要精確算村里密度必須先做地址 geocoding（TGOS 或國土測繪中心圖資），那是 NB2 的工作。

本 notebook 用三種方式在**不做 geocoding** 的前提下逼近村里密度：

| 方法 | 性質 | 用途 |
|---|---|---|
| **A. 理論上限** | 嚴格數學上界 | 若上限就 ≤ 50%，村里粒度**確定出局**，不必等 geocoding |
| **B. 戶數比例分配（multinomial）** | 「完全無空間聚集」虛無情境 | 交易若像戶數一樣攤平會是什麼樣 |
| **C. 街道聚集情境** | 用實際門牌路段的聚集程度 | 帶有真實聚集特性的樂觀估計 |

三者關係恆為 **A ≥ C**，B 為參照。決策以 A 為準——A 是唯一不會因估計方法失準而翻案的數字。

鄉鎮市區粒度則是**精確計算**，無此限制。
""")

# ============================== §0 環境 ==============================
md(r"""
---
## 0. 環境設定
""")

code(r"""
# Colab 執行時取消註解
# !pip install -q pandas numpy pyarrow requests matplotlib
""")

code(r'''
import gc
import io
import json
import re
import sys
import time
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 60)

# ---- 專案路徑 ----
try:
    import google.colab  # noqa: F401
    from google.colab import drive

    drive.mount("/content/drive", force_remount=False)
    PROJECT = Path("/content/drive/MyDrive/TW_Housets")
    IN_COLAB = True
except ImportError:
    _cwd = Path.cwd()
    PROJECT = _cwd.parent if _cwd.name == "notebooks" else _cwd
    IN_COLAB = False

RAW, PROC = PROJECT / "data" / "raw", PROJECT / "data" / "processed"
LVR_DIR, FIG_DIR = RAW / "lvr", PROC / "figs"
for _d in (RAW, PROC, LVR_DIR, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

print(f"IN_COLAB = {IN_COLAB}")
print(f"PROJECT  = {PROJECT}")
''')

code(r'''
# ============================ 分析設定 ============================
# 六都在實價登錄批次檔中的縣市代碼（檔名前綴）
CITY_CODES = {
    "a": "臺北市",
    "f": "新北市",
    "h": "桃園市",
    "b": "臺中市",
    "d": "臺南市",
    "e": "高雄市",
}
CITY_ORDER = ["臺北市", "新北市", "桃園市", "臺中市", "臺南市", "高雄市"]

# 分析期間（交易日期，非發布日期）
DATE_START, DATE_END = pd.Timestamp("2017-01-01"), pd.Timestamp("2024-12-31")

# 要下載的發布季。注意：檔名的「季」是**發布季**不是交易季——
# 例如 106S1 內含的是民國 105 年 10-12 月登記的案件。
# 因此下載範圍必須比分析期間寬，之後再用 交易年月日 過濾。
SEASONS = [f"{y}S{s}" for y in range(106, 115) for s in (1, 2, 3, 4)]  # 106S1 ~ 114S4

# 建物型態：**以「組」為單位**，一組 = 同一種建物型態的不同世代用語。
# 為什麼要分組：來源用語改版時（實價登錄 2.0 就改過），舊用語只在前半段序列出現、
# 新用語只在後半段出現。若 §3.2 的 Layer 2a″ 要求「每個前綴逐季都命中」，那麼
# 照它的錯誤訊息把新用語加進來之後**兩個前綴各有一半的季是 0**，檢查永遠修不好，
# 只能靠關掉 COVERAGE_STRICT——那會一次關掉七層的中止能力。
# 所以 2a″ 檢查的是「每一組**逐季的聯集**非零」：把新用語加進同一組就是可通過的修法。
BUILDING_GROUPS = {
    "住宅大樓": ("住宅大樓",),      # 住宅大樓(11層含以上有電梯)
    "華廈": ("華廈",),              # 華廈(10層含以下有電梯)
    "公寓": ("公寓",),              # 公寓(5樓含以下無電梯)
}
MAIN_GROUPS = ("住宅大樓",)          # 嚴格版只用這一組；其餘是決策樹第 4 分支的放寬版
BUILDING_MAIN = tuple(dict.fromkeys(
    pp for gg in MAIN_GROUPS for pp in BUILDING_GROUPS[gg]))
BUILDING_RELAXED = tuple(dict.fromkeys(
    pp for ps in BUILDING_GROUPS.values() for pp in ps))
assert set(BUILDING_MAIN) <= set(BUILDING_RELAXED)

# 寫給 NB2 的欄位。**這是唯一的真相來源**——§2 的 USE_COLS（read_csv 只讀這些欄）
# 由它推導，所以不可能出現「KEEP 有、根本沒讀進來」的靜默缺欄。
KEEP = [
    "縣市", "鄉鎮市區", "district", "交易日", "month", "quarter",
    "土地位置建物門牌", "street", "street_key", "編號",
    "建物型態", "交易標的", "總價元", "車位總價元", "房屋總價",
    "建物移轉總面積平方公尺", "車位移轉總面積平方公尺", "房屋面積坪", "單價元每坪",
    "n_土地", "n_建物", "n_車位", "車位價未拆",
    "移轉層次", "總樓層數", "建築完成年月", "都市土地使用分區",
    "建物現況格局-房", "建物現況格局-廳", "建物現況格局-衛", "備註", "發布季",
]
# KEEP 裡由清整過程產生（不是來源欄位）的，推導 USE_COLS 時要排除
DERIVED_COLS = {"縣市", "district", "交易日", "month", "quarter", "street", "street_key",
                "房屋總價", "房屋面積坪", "單價元每坪",
                "n_土地", "n_建物", "n_車位", "車位價未拆", "發布季"}

PING_M2 = 3.305785                                 # 1 坪 = 3.305785 平方公尺
PRICE_MIN, PRICE_MAX = 50_000, 2_500_000           # 元/坪 合理區間（計畫 4.1）

# 備註關鍵字：命中即排除（計畫 4.1「親友、法拍、特殊」）
#
# ⚠️ 不要加入「預售」。實價登錄 2.0 的制式備註是
#    「預售屋、或土地及建物分件登記案件；」
#    ——把兩件事包在同一句，而實際上絕大多數是後者（一般成屋的土地建物分件登記）。
#    以「預售」比對會命中約 41% 的成屋交易，足以扭曲密度結論。
#    真正的預售屋交易另存於 *_lvr_land_b.csv，本 notebook 根本沒讀。
#
# 「法拍／拍賣／非自願」在 2.0 之後的實際用語是「受債權債務關係影響或債務抵償」，
# 由「債權」「債務抵償」命中；舊版用語保留在清單裡當安全網。
NOTE_EXCLUDE = ("親友", "員工", "特殊關係", "法拍", "拍賣", "債權", "債務抵償", "非自願")

MIN_N = 5           # 決策門檻：cell 內交易筆數
PASS_SHARE = 0.50   # 決策門檻：n≥MIN_N 的 cell 佔比須 > 50%

# 佔比的分母口徑（見 §5.0「分母的雙口徑」）。兩個口徑一律都算、都寫進報告；
# 這裡只決定哪一個當「主決策」用：
#   "full"   — 六都全部鄉鎮市區（計畫 §5.0 字面讀法，預設）
#   "active" — 全期至少一筆交易的鄉鎮市區（只算可能進 NB2 模型的地區）
# 兩者結論相反時，決策階梯會判 CONFLICT 就地停住、不自動選邊。
DENOMINATOR_PRIMARY = "full"
assert DENOMINATOR_PRIMARY in ("full", "active")

# ---- 涵蓋率檢查（見 §1 與 §3.5）----
# 下載範圍是「發布季」、分析窗口是「交易季」，兩者錯開。少了幾個發布季，
# 對應的交易月會被系統性少算，而密度數字照樣算得出來、**看起來完全正常**。
# 這類錯誤沒有自然的失敗訊號，所以預設中止而不是只印警告。
COVERAGE_STRICT = True      # False = 只警告不中止（確定要在資料不全下繼續才改）
COVERAGE_MIN_RATIO = 0.40   # 月交易量 / 置中 13 個月滾動中位數 的下限

# 季序列階梯的門檻（Layer 2b′ 與 2a‴ 共用）。
# 0.55：市場循環（2017 低 → 2021 高 → 2023 回落）造成的前後段中位數比，實測落在
# 0.94 以上（中位數把循環大幅平滑掉了），餘裕充足；掉到 0.55 以下是資料問題不是市場。
# 上升方向**不**取倒數。實測「階梯式的真實成長」只要 2.0× 就會撞到 1.82
# （前段可以只有 4 季，容易被序列開頭的真實低谷帶低），而真缺陷造成的上升是
# 3.2–3.5×，餘裕很足。所以上升門檻另外設在 2.5，換掉那段誤報空間。
STEP_RATIO = 0.55
STEP_UP_RATIO = 2.5
# Layer 2a‴ 用**另一組**門檻。2b′ 跑的是 raw 的全型態總量，沒有型態組成的趨勢曝險；
# 2a‴ 跑的是逐型態的筆數，而真實資料上「公寓」正在長期衰退——實測下降側六都全部
# 由公寓把持最低值（臺中 0.681、桃園/臺南 0.707、高雄 0.708），對 0.55 只有 1.24 倍餘裕。
# 而且 level_shift 的最後一個切點只用最後 4 季當後段，SEASONS 往後延長時餘裕會繼續縮
# （公寓全國逐季命中數已從峰值 8,642 掉到 3,972）。這跟舊版 2a′ 被臺南趨勢誤報是同一種病。
GROUP_STEP_DOWN = 0.45
GROUP_STEP_UP = 2.5

# Layer 2a″（建物型態前綴的逐季普查）
# 某前綴連續這麼多季零命中就判定為「來源用語從那一季起改掉了」。
# 六都合計每季只要有量的型態都是數千列，連兩季全國零命中在健康資料上不可能發生。
PREFIX_ZERO_RUN = 2
# 但太小的前綴某季自然為 0 是正常的，那種只做全域普查、不查階梯。
PREFIX_MIN_MEDIAN = 50

# ---- Layer 4：外部絕對錨 ----
# 前面每一層的基準幾乎都由資料自身算出，對「基準被污染」結構性無免疫（見 §3.2 已知限制）。
# 這一層的分母來自體制外：內政統計月報 4.5「辦理建物所有權登記」的**買賣移轉棟數**，
# 按縣市 × 登記月公布。我們每個批次檔的列數 ÷ 官方同期棟數 應該長期穩定，
# 比值掉下去就是我們掉了資料——這是唯一能擋「全窗口等比例少算」的檢查。
ANCHOR_ENABLE = True
ANCHOR_REQUIRED = False     # True = 抓不到官方資料就中止。預設 False：抓不到多半是網路問題，
                            # 不該讓整份 notebook 停擺；但會明確標記「這一層沒跑」而不是綠燈。
# 發布季 → 登記季 的落後季數。實測（逐檔看 交易年月日 的分布）：106S1 的交易季
# 眾數是 2016Q4（佔 79.7%）、108S1 是 2018Q4（73.5%）——**發布季的標籤比交易季早
# 一季**。登記在交易後約 30 天，所以登記季橫跨兩個季（該季晚期的交易落到下一季登記），
# 有效落後其實在 0 與 1 之間。因此相鄰 lag 的診斷數字本來就分不開（實測 CV
# lag=0 0.175 vs lag=1 0.215），只有差 2 季以上才有意義。
# 形狀檢查用四季滾動和，這個模糊性被抵銷掉；絕對水準用全窗口總和，同樣不受影響。
ANCHOR_LAG_Q = 1
# 絕對區間（全窗口總和的 我方筆數 / 官方棟數）。
# **即使下限已用真實資料校準，實測解析度也只有「等比例少算 30%」**
# ——少算 25% 仍會通過全部各層。這是這一層的天花板，不是設定問題。
# 真實資料實測（2026-08-25、36 個發布季）：臺北 0.876、新北 0.970、桃園 1.122、
# 臺中 1.109、臺南 1.404、高雄 1.114。**比值大於 1 是正常的**——我方分子是
# 「可解析列數」，含純土地／純車位標的，而官方買賣移轉棟數不算它們；臺南特別高是
# 因為它的透天／土地交易佔比高。舊的上限 1.15 會誤報四個縣市。
# 下限 0.61 ≈ 實測最小值 0.876 × 0.7。上限 2.50 ≈ 實測最大值 1.404 × 1.8，
# 目的降級為「攔住粗大的結構性錯誤」——「同一批資料載入兩次」其實已經被
# clean_lvr 第 4 步的 (縣市, 編號) 去重吃掉了，不需要靠這個上限。
ANCHOR_RATIO_MIN, ANCHOR_RATIO_MAX = 0.61, 2.50
# 落後季數診斷用的候選（設錯 lag 不會讓檢查中標，只會吃掉階梯餘裕，所以要診斷）
ANCHOR_LAG_CANDIDATES = (0, 1, 2, 3)
# 可比對格子的最低比例。沒有這道閘門的話，只要 3 個發布季可比（8.3% 的格子）
# 就會印 ✅，而形狀那兩支根本沒跑。
ANCHOR_MIN_COVER = 0.80
# 形狀檢查用四季滾動和（發布季與登記季的邊界本來就對不齊，逐季比值會有季節性抖動，
# 滾動四季幾乎把錯位抵銷掉）。
ANCHOR_ROLL = 4
ANCHOR_REL_MIN = 0.70       # 滾動比值 / 該縣市中位比值 的下限
# 階梯門檻。**不要用互為倒數的一對**：比值分布右偏，而 level_shift 的上升方向是
# 「29 個切點 × 6 縣市 = 174 次機會的最大值」，所以上升側才是綁死的那一邊。
# 實測（合成資料注入雜訊）：0.80/1.25 的安全區只到 iid ±25%，而 ±25% 時上升側
# 已經到 1.238——離門檻只剩 0.012，跟 Layer 2a′ 曾經只有 0.02 餘裕是同一種形狀。
# 放寬到 0.75/1.50 之後：文件宣告的職責（後段掉到 55–70%、前段掉到 45–55%）全部
# 仍然接得到，而誤報餘裕從 2% 拉到 23%。Layer 2b′ 早就學過這一課（0.55 / 2.5）。
ANCHOR_STEP_DOWN, ANCHOR_STEP_UP = 0.75, 1.50
# 內政統計月報 4.5：現行檔（近 10 年）與歷史檔。分析窗口最早需要民國 105 年第 4 季的
# 登記數，那一年只在歷史檔裡，所以兩個都抓；只抓到現行檔則自動降為「部分涵蓋」。
ANCHOR_URLS = (
    "https://statis.moi.gov.tw/micst/report/324050.ods",
    "https://ws.moi.gov.tw/001/Upload/400/relfile/0/4413/"
    "65a0a7cc-cb81-4555-99fb-8f6c876dd0d3/m4-05.ods",
)

# ---- Layer 5：跨次執行快照 ----
# 外部錨管的是「我們跟官方比對不對」，快照管的是「這次跟上次比有沒有掉」。
# 兩者互補：快照不需要任何外部來源，而且只要上一次是好的，連等比例少算都擋得住。
SNAPSHOT_ENABLE = True
SNAPSHOT_DROP_FAIL = 0.90   # 某 (發布季 × 縣市) 掉到上次的九成以下 → 中止
SNAPSHOT_DROP_WARN = 0.98   # 掉到 98% 以下 → 列出來但不中止（來源確實會撤銷／修正個別案件）
# 這次比對到的格數 / 現在的格數。基準若比現在窄（例如先用窄 SEASONS 試跑一次就寫了
# 基準），沒有這道閘門會印出「比對 N 個 ✅」而實際一格都沒比。
SNAPSHOT_MIN_COVER = 0.90
SNAPSHOT_SCHEMA = 2         # 改了快照內容的定義就要加一，舊快照才會失效
                            # v2：加上 group_counts 與 external_anchor_status


class CoverageError(RuntimeError):
    """涵蓋率檢查失敗。數字會被系統性低估但看起來正常，故預設中止。"""


def coverage_fail(title: str, detail: str) -> None:
    """涵蓋率檢查不通過時的統一出口：預設 raise，COVERAGE_STRICT=False 時降為警告。"""
    msg = f"{title}\n{detail}"
    if COVERAGE_STRICT:
        raise CoverageError(
            msg + "\n\n若已確認可以接受不完整的涵蓋，把 COVERAGE_STRICT 設為 False 再重跑。")
    print(f"{msg}\n\n（COVERAGE_STRICT = False，僅警告不中止——數字可能被系統性低估）\n")


def level_shift(v: np.ndarray, min_side: int = 4) -> tuple:
    """在季序列上找 level shift：以每個切點把序列切成前後兩段比中位數，
    回傳最極端的兩個切點 (下降最深, 上升最高)，各為 (i, 前段中位, 後段中位, 比值)。

    **兩個方向都要看。** 只找下降的話，「序列前段被少算」呈現為上升，會結構性地
    永遠看不到——而「從第一季就少算、到某季才修好」跟「某季之後開始少算」一樣真實，
    且前者會讓 Layer 2a′/2b/3 的中位數基準全部被污染（見本節末的已知限制）。

    用中位數而非平均，才不會被單一異常季帶偏。
    """
    down = up = None
    for i in range(min_side, len(v) - min_side + 1):
        before, after = float(np.median(v[:i])), float(np.median(v[i:]))
        if before <= 0 or after <= 0:
            continue
        cand = (i, before, after, after / before)
        if down is None or cand[3] < down[3]:
            down = cand
        if up is None or cand[3] > up[3]:
            up = cand
    return down, up

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"}

# 分析窗口的期別索引（§3.2 涵蓋率檢查與 §5 密度分析共用）
MONTHS = pd.period_range(DATE_START, DATE_END, freq="M")
QUARTERS = pd.period_range(DATE_START, DATE_END, freq="Q")
N_PERIODS = {"月": len(MONTHS), "季": len(QUARTERS)}

print(f"分析期間 {DATE_START:%Y-%m} ~ {DATE_END:%Y-%m}"
      f"（{N_PERIODS['月']} 個月 / {N_PERIODS['季']} 季）")
print(f"下載發布季 {SEASONS[0]} ~ {SEASONS[-1]}（{len(SEASONS)} 季）")
''')

# ============================== §1 下載 ==============================
md(r"""
---
## 1. 下載實價登錄批次資料

端點：`https://plvr.land.moi.gov.tw/DownloadHistory?type=season&fileName={季別}`

每個 zip 約 60–130 MB，內含全國所有縣市 × 買賣/租賃/預售 × 多種格式共約 600 個檔。
我們只解出六都的**買賣主檔** `{代碼}_lvr_land_a.csv`（每個約 2–5 MB），解完就把 zip 丟掉，
硬碟只留約 100 MB。

36 季全部下載約需 10–30 分鐘（視網速）。已完成的季會跳過，中斷可直接重跑。
""")

code(r'''
_WANT = re.compile(r"^([afhbde])_lvr_land_a\.csv$")


def fetch_season(season: str, retries: int = 3) -> str:
    """下載單一發布季，只解出六都買賣主檔。回傳狀態字串。"""
    marker = LVR_DIR / f".{season}.done"
    if marker.exists():
        return "skip"

    url = f"https://plvr.land.moi.gov.tw/DownloadHistory?type=season&fileName={season}"
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=UA, timeout=900, stream=True)
            resp.raise_for_status()

            buf = io.BytesIO()
            for chunk in resp.iter_content(1 << 20):
                buf.write(chunk)

            # 端點在查無資料時會回傳一頁很小的 HTML 而非 zip，用大小擋掉
            if buf.tell() < 5_000_000:
                raise RuntimeError(f"回應僅 {buf.tell():,} bytes，疑似錯誤頁而非 zip")

            with zipfile.ZipFile(buf) as z:
                members = [n for n in z.namelist() if _WANT.match(n)]
                if not members:
                    raise RuntimeError("zip 內找不到六都買賣主檔")
                for name in members:
                    code = _WANT.match(name).group(1)
                    (LVR_DIR / f"{season}_{code}.csv").write_bytes(z.read(name))

            marker.touch()
            return f"{len(members)} 檔 / {buf.tell() / 1e6:.0f} MB"

        except Exception as exc:  # noqa: BLE001
            if attempt == retries - 1:
                return f"FAILED: {exc}"
            time.sleep(5 * (attempt + 1))
    return "FAILED"


t0 = time.time()
for i, season in enumerate(SEASONS, 1):
    status = fetch_season(season)
    print(f"[{i:2d}/{len(SEASONS)}] {season}  {status}")
    if status.startswith("FAILED"):
        print("    ^^^ 這一季失敗了。可稍後單獨重跑 fetch_season() 補上。")
print(f"\n完成，耗時 {time.time() - t0:.0f} 秒")
''')

code(r'''
# glob 的樣式比 fetch_season 的命名寬鬆——手動解壓官方 zip 會產生
# a_lvr_land_a.csv / f_lvr_land_b.csv 這類同樣以 _{代碼}.csv 結尾的檔名。
# 只認 {發布季}_{縣市代碼} 這一種，其餘忽略並列出，避免解析時炸開。
_LVR_NAME = re.compile(r"^(\d{3}S[1-4])_([afhbde])$")
_found = sorted(LVR_DIR.glob("*_[afhbde].csv"))
files = [f for f in _found if _LVR_NAME.match(f.stem)]
stray = [f.name for f in _found if not _LVR_NAME.match(f.stem)]

print(f"已取得 {len(files)} 個檔案（預期 {len(SEASONS) * 6}），"
      f"共 {sum(f.stat().st_size for f in files) / 1e6:.0f} MB")
if stray:
    print(f"⚠️  忽略 {len(stray)} 個檔名不合格式的檔案"
          f"（需為 {{發布季}}_{{縣市代碼}}.csv）：{stray[:5]}"
          + (" …" if len(stray) > 5 else ""))
    print("    手動解壓 zip 會產生這類檔名；請改用 fetch_season() 下載，否則不會被採計。")

# glob 不看 SEASONS，所以縮短 SEASONS **不會**少載入任何檔案（省不到記憶體），
# 卻會把 Layer 2b / 2a″ / 4 的檢查範圍一起縮小——那些資料照樣進 raw、lvr 與 parquet，
# 只是沒人檢查。所以先按 SEASONS 過濾，再要求兩邊對得上。
_off_range = [f.name for f in files if _LVR_NAME.match(f.stem).group(1) not in set(SEASONS)]
files = [f for f in files if _LVR_NAME.match(f.stem).group(1) in set(SEASONS)]
if _off_range:
    print(f"⚠️  忽略 {len(_off_range)} 個不在 SEASONS 範圍內的檔案"
          f"（{_off_range[:4]}{' …' if len(_off_range) > 4 else ''}）")

# ---- Layer 1：發布季完整性 ----
missing = [s for s in SEASONS if not (LVR_DIR / f".{s}.done").exists()]

# 下載成功但只解出部分縣市：fetch_season 只要求「至少一個」六都主檔存在，
# 因此某季少了一兩個縣市仍會被標記為完成——這裡逐季補查。
have: dict = {}
for f in files:
    season, code = _LVR_NAME.match(f.stem).groups()
    have.setdefault(season, set()).add(code)
# 縣市依 CITY_CODES 的順序（北→南）列出，不用字串排序
partial = {s: [CITY_CODES[c] for c in CITY_CODES if c not in have.get(s, set())]
           for s in SEASONS if s not in missing and have.get(s, set()) != set(CITY_CODES)}

if len(files) != len(SEASONS) * len(CITY_CODES) and not missing and not partial:
    coverage_fail(
        f"❌ 檔案數 {len(files)} ≠ 發布季 {len(SEASONS)} × 縣市 {len(CITY_CODES)}"
        f" = {len(SEASONS) * len(CITY_CODES)}",
        "    每季每個縣市應該恰好一個主檔。多出來的多半是手動解壓產生的重複檔名，\n"
        "    少掉的會讓對應的交易月被系統性少算而統計量看起來正常。")

if missing or partial:
    lines = []
    if missing:
        lines.append(f"缺少 {len(missing)} 個發布季：{missing}")
    if partial:
        lines.append(f"{len(partial)} 個發布季缺部分縣市主檔：")
        lines += [f"    {s}：缺 {'、'.join(cities)}" for s, cities in partial.items()]
    lines += [
        "",
        "下載範圍是發布季、分析窗口是交易季，兩者錯開——缺季會讓對應的交易月被系統性",
        "少算，但密度數字照樣算得出來、看起來完全正常（沒有自然的失敗訊號）。",
        f"補救：對缺的季單獨補跑，例如 fetch_season(\"{(missing or list(partial))[0]}\")，再重跑本 cell。",
    ]
    coverage_fail("❌ 發布季不完整", "\n".join(lines))
else:
    print(f"✅ {len(SEASONS)} 個發布季齊備，每季六都主檔皆到齊")

# §3.2 的 coverage dict 要帶上這一層的結果，否則報告只能寫死「皆齊」而說謊
LAYER1 = {"n_seasons": len(SEASONS), "n_files": len(files),
          "n_expected": len(SEASONS) * len(CITY_CODES),
          "missing_seasons": missing,
          "partial_seasons": {k: list(v) for k, v in partial.items()},
          "stray_files": stray}
''')

# ============================== §2 載入 ==============================
md(r"""
---
## 2. 載入與合併

每個 CSV 的**第一列資料是英文表頭**（不是資料列），必須丟掉。
用「`交易年月日` 是不是 6–7 位數字」來過濾，比用 `skiprows` 穩健。

載入時就做三件瘦身，因為 Colab 免費版只有 ~12 GB RAM，36 季全型態全欄位讀進來會 OOM：

1. **只留住宅類**（住宅大樓／華廈／公寓）——列數壓到約 1/3
2. **只讀用得到的欄位**（`USE_COLS`）——`移轉編號`、`主建物面積`、`附屬建物面積`、
   `陽台面積` 這幾個高基數欄位本專案一個都沒用到，卻各佔數百 MB
3. **數值欄與日期欄當場轉型**——字串型的 `總價元`／`單價元平方公尺` 是 `raw` 裡最貴的東西
   （每列都不同，pandas 的字串去重快取幫不上忙）；日期轉掉還順便省下 §3 與 §3.2
   各自再解析整份 `raw` 一次

載入後會印出 `raw` 的逐欄記憶體用量——這個數字是**實測**的，不要拿推估值決定要不要開跑。
""")

code(r'''
# 只列**下游真的會用到**的數值欄。曾經放過 單價元平方公尺 與 土地移轉總面積平方公尺，
# 但全 notebook 沒有任何地方讀它們、也不在 KEEP 裡（進不了 parquet）——在一個
# 存在目的就是省記憶體的白名單裡純屬浪費，而且是 clean_lvr KeyError 的唯一暴露面。
NUMERIC_COLS = [
    "總價元", "建物移轉總面積平方公尺",
    "車位移轉總面積平方公尺", "車位總價元",
    "建物現況格局-房", "建物現況格局-廳", "建物現況格局-衛",
]

# 實價登錄 2.0（民國110年7月）前後欄名有差異，統一到新版名稱。
# 例：106 年版是「車位移轉總面積(平方公尺)」，112 年版拿掉了括號。
COL_ALIASES = {
    "車位移轉總面積(平方公尺)": "車位移轉總面積平方公尺",
    "建物移轉總面積(平方公尺)": "建物移轉總面積平方公尺",
    "土地移轉總面積(平方公尺)": "土地移轉總面積平方公尺",
    "單價(元/平方公尺)": "單價元平方公尺",
    "單價元/平方公尺": "單價元平方公尺",
}

REQUIRED_COLS = [
    "鄉鎮市區", "交易標的", "土地位置建物門牌", "交易年月日", "交易筆棟數",
    "建物型態", "總價元", "建物移轉總面積平方公尺", "車位移轉總面積平方公尺",
    "車位總價元", "備註", "編號",
]

# read_csv 只讀這些欄（36 季全欄位讀進來會 OOM）。**由 §0 的 KEEP 推導**，
# 不是另一份手維護的清單——兩份清單走鐘時 §3.5 只會靜默少一欄，NB2 才會爆。
USE_COLS = list(dict.fromkeys(
    REQUIRED_COLS + NUMERIC_COLS + [c for c in KEEP if c not in DERIVED_COLS]))


def roc_to_datetime(s: pd.Series) -> pd.Series:
    """民國年月日字串（1051007 / 990101）→ Timestamp"""
    s = s.astype(str).str.strip()
    return pd.to_datetime(
        {
            "year": pd.to_numeric(s.str[:-4], errors="coerce") + 1911,
            "month": pd.to_numeric(s.str[-4:-2], errors="coerce"),
            "day": pd.to_numeric(s.str[-2:], errors="coerce"),
        },
        errors="coerce",
    )


_WARNED_ABSENT: set = set()   # 只對前幾個檔印「期待卻不存在的欄名」，不要洗版
LOAD_STATS: list = []      # 每個 csv 的列數保留情況，§3.2 Layer 2a′ 使用
# 建物型態的原始用語普查（在型態篩選**之前**），§3.2 Layer 2a″／2a‴ 使用。
# key 是 (發布季, 縣市, 建物型態)：
#   · 全域普查只能問「有沒有命中過」，對「某組從某季起消失」完全失明 → 需要「季」
#   · 「只有部分縣市改用語」在全國加總後被稀釋掉 → 需要「縣市」
BUILDING_SEEN: dict = {}


def load_one(path: Path) -> Optional[pd.DataFrame]:
    season, code = path.stem.split("_")
    # 先只讀表頭：usecols 要用檔案裡的原始欄名，而別名對照是在標準化之後才成立
    try:
        head = pd.read_csv(path, dtype=str, encoding="utf-8-sig", nrows=0)
    except Exception as exc:  # noqa: BLE001
        print(f"  讀取失敗 {path.name}: {exc}")
        return None

    renamed = [COL_ALIASES.get(c.strip(), c.strip()) for c in head.columns]
    missing = [c for c in REQUIRED_COLS if c not in renamed]
    if missing:
        print(f"  ⚠️  {path.name} 缺少欄位 {missing}，跳過。"
              "（若大量檔案出現，代表來源欄名又改了，須更新 COL_ALIASES）")
        return None

    # 別名衝突：來源同時保留舊欄名與新欄名（例如 單價(元/平方公尺) 與 單價元平方公尺
    # 並存）時，標準化後會出現兩個同名欄，之後 pd.to_numeric 會拿到 DataFrame 而不是
    # Series，炸在 try/except 之外、訊息完全指不出原因。走既有的「跳過檔案」路徑，
    # 讓 Layer 2a 用對的訊息中止。
    _dupes = {c for c in USE_COLS if renamed.count(c) > 1}
    if _dupes:
        print(f"  ⚠️  {path.name} 有 {len(_dupes)} 個欄名在標準化後重複：{sorted(_dupes)}，跳過。"
              "（來源同時保留了舊欄名與新欄名，須調整 COL_ALIASES 決定用哪一個）")
        return None

    use = [orig for orig, new in zip(head.columns, renamed) if new in USE_COLS]
    # 反向訊號：USE_COLS 要的欄名而來源沒有。少了這個，「衍生欄從 DERIVED_COLS 漏掉」
    # 會完全靜默（那個名字進了 USE_COLS，但 CSV 裡沒有，過濾器直接略過）。
    _absent = sorted(set(USE_COLS) - set(renamed))
    if _absent and path.name not in _WARNED_ABSENT:
        _WARNED_ABSENT.add(path.name)
        if len(_WARNED_ABSENT) <= 3:
            print(f"  ℹ️  {path.name}：USE_COLS 期待但來源沒有的欄名 {_absent}"
                  "（若那是清整產生的衍生欄，要加進 §0 的 DERIVED_COLS）")
    try:
        df = pd.read_csv(path, dtype=str, encoding="utf-8-sig", low_memory=False,
                         usecols=use)
    except Exception as exc:  # noqa: BLE001
        print(f"  讀取失敗 {path.name}: {exc}")
        return None

    df.columns = [COL_ALIASES.get(c.strip(), c.strip()) for c in df.columns]

    # 丟掉英文表頭列與任何日期格式不合的列。
    # **兩道日期閘門要分開量。** 「符合 \d{6,7}」只是字串形狀；`1051307`（13 月）
    # 通得過形狀卻解析不出日期。舊版把這種列留在 raw 裡、到 §3 的日期區間篩選才靜默
    # 丟掉——於是 Layer 2b/2b′/4/5 全部看不到它（實測可靜默吃掉 15% 的 lvr）。
    ok_fmt = df["交易年月日"].astype(str).str.strip().str.fullmatch(r"\d{6,7}")
    _dt = roc_to_datetime(df["交易年月日"])
    ok_date = ok_fmt & _dt.notna()
    # 只留住宅類（BUILDING_RELAXED 是嚴格版的超集），把 raw 的記憶體壓到 1/3
    _bt = df["建物型態"].fillna("")
    ok_type = _bt.str.startswith(BUILDING_RELAXED)
    ok_main = _bt.str.startswith(BUILDING_MAIN)

    # 這兩道篩選是「靜默丟列」——來源用語改版（實價登錄 2.0 就是 110S3 生效）
    # 會讓某季起的列被大量丟掉，而 load_one 照樣回傳一個看似正常的 DataFrame。
    # 逐檔記下保留情況，§3.2 Layer 2a′ 據此比對各季是否一致。
    # 在型態篩選之前先記下所有用語。篩選之後就只剩對得上的，看不到來源改成了什麼——
    # 而「來源改成了什麼」正是修正 BUILDING_RELAXED 時唯一需要的資訊。
    _city = CITY_CODES[code]
    for _t, _n in _bt.where(ok_date).fillna("(空白)").value_counts().items():
        if _t == "(空白)" and not (ok_date & (_bt == "")).any():
            continue            # where() 產生的 NaN，不是來源真的空白
        BUILDING_SEEN[(season, _city, _t)] = \
            BUILDING_SEEN.get((season, _city, _t), 0) + int(_n)

    LOAD_STATS.append({
        "發布季": season, "縣市": _city, "檔名": path.name,
        "原始列數": len(df),
        "日期格式合法": int(ok_fmt.sum()),
        "日期可解析": int(ok_date.sum()),
        "住宅類": int((ok_date & ok_type).sum()),
        "主型態": int((ok_date & ok_main).sum()),
    })

    _keep_rows = ok_date & ok_type
    df = df[_keep_rows].copy()

    # ---- 當場轉型（見本節說明的第 3 點）----
    for c in NUMERIC_COLS:
        if c in df.columns:
            # float64 而非 float32：總價元 可達 1e9，float32 的 24 bit 尾數放不下
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    # 交易年月日 只解析一次（上面已經算好），這裡直接帶走，並丟掉原字串。
    # 進到 raw 的每一列都保證有合法的 交易日，NaT 不會混進來。
    df["交易日"] = _dt[_keep_rows]
    df = df.drop(columns=["交易年月日"])
    # 這兩欄刻意**不**轉 category。整欄指向同一個字串物件，object dtype 只花每列 8 bytes
    # 的指標（省不到 20 MB）；但一旦是 category，凡是沒寫 observed=True 的多鍵 groupby
    # 都會改成產生完整笛卡兒積——§5–§7 有好幾處這樣的 groupby，語意會被悄悄改掉。
    df["縣市"] = CITY_CODES[code]
    df["發布季"] = season
    return df


LOAD_STATS.clear()          # 允許重跑本 cell
BUILDING_SEEN.clear()
frames, load_failed = [], []
for p in files:
    d = load_one(p)
    if d is None:
        load_failed.append(p.name)      # §3.2 Layer 2a 會據此中止
    else:
        frames.append(d)
raw = pd.concat(frames, ignore_index=True)
del frames
gc.collect()

print(f"合併後原始筆數：{len(raw):,}"
      f"（成功載入 {len(files) - len(load_failed)}/{len(files)} 檔）")
print(raw.groupby("縣市").size().reindex(CITY_ORDER).to_string())

# ---- raw 的實測記憶體（36 季會不會 OOM 只能量、不能推估）----
def rss_peak_gb() -> Optional[float]:
    """本行程至今的 RSS 高水位（GB）——高水位涵蓋 pd.concat 的瞬時峰值。
    ru_maxrss 的單位在 Linux（含 Colab）是 KB、在 macOS 是 bytes。"""
    try:
        import resource
        v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return v / 1e9 if sys.platform == "darwin" else v / 1e6
    except Exception:            # noqa: BLE001  Windows 沒有 resource
        return None


_mem = raw.memory_usage(deep=True).drop("Index").sort_values(ascending=False)
_peak = rss_peak_gb()
# deep=True 會把「多列共用同一個字串物件」重複計算（pandas 的 C parser 會去重
# 重複字串），所以這個數字是**上界**而不是真實佔用；真實佔用看下面的 RSS。
print(f"\nraw 記憶體：pandas 深度統計 {_mem.sum() / 1e9:.2f} GB"
      f"（{_mem.sum() / max(len(raw), 1):.0f} bytes/列，上界——共用字串被重複計算）")
print("\n".join(f"  {c:<16} {v / 1e6:>8.1f} MB" for c, v in _mem.head(6).items()))
if _peak is None:
    print("  （拿不到 RSS，只能參考上面的上界）")
else:
    print(f"  行程 RSS 高水位（已含 pd.concat 的瞬時峰值）：{_peak:.2f} GB")
    print(f"  Colab 免費版約 12.7 GB。若高水位已過半，先縮短 SEASONS 分批跑，"
          "或再砍 EXTRA_COLS。")
''')

# ============================== §3 清整 ==============================
md(r"""
---
## 3. 清整（計畫 4.1 規則）

依序套用，每一步都記錄剩餘筆數，方便稽核是哪一條規則砍掉最多。

1. 建物型態 = 住宅大樓（11層含以上有電梯）
2. 交易標的含「房地」（排除純土地、純車位、純建物）
3. 交易日期落在 2017-01-01 ~ 2024-12-31
4. 以 `縣市 + 編號` 去重（跨季重複發布）
5. 備註排除：親友／特殊關係／法拍／債權／非自願／預售
6. 拆解車位：總價與建物面積各扣掉車位部分
7. 總價、面積必須 > 0
8. 單價落在 5 萬 ~ 250 萬 元/坪
""")

code(r'''
# roc_to_datetime 定義在 §2（載入時就要用它把 交易年月日 轉掉）
_FW = str.maketrans("０１２３４５６７８９", "0123456789")
_ROAD = re.compile(r"^(.+?(?:大道|路|街|道))([一二三四五六七八九十]{1,3}段)?")


def street_key(addr: str, city: str, dist: str) -> str:
    """從門牌抽出『路名(+段)』，作為村里的替代空間單元。"""
    if not isinstance(addr, str) or not addr:
        return "(未解析)"
    s = addr.translate(_FW).replace(city, "").replace(city.replace("臺", "台"), "")
    if isinstance(dist, str):
        s = s.replace(dist, "", 1)
    m = _ROAD.match(s.strip())
    return (m.group(1) + (m.group(2) or "")) if m else "(未解析)"
''')

code(r'''
def clean_lvr(raw: pd.DataFrame, building_types: tuple[str, ...],
              verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """回傳 (清整後資料, 各步驟筆數表)"""
    steps: list[dict] = []

    def log(name: str, df: pd.DataFrame) -> pd.DataFrame:
        prev = steps[-1]["剩餘筆數"] if steps else len(df)
        steps.append({"步驟": name, "剩餘筆數": len(df), "本步驟移除": prev - len(df)})
        return df

    # 直接以布林遮罩取子集，不先整份 copy（raw 可達數 GB）
    df = raw[raw["建物型態"].fillna("").str.startswith(building_types)].copy()
    steps.append({"步驟": "0. 載入（已預篩住宅類）", "剩餘筆數": len(raw), "本步驟移除": 0})
    log("1. 建物型態", df)

    # 2. 交易標的含房地
    df = log("2. 交易標的含房地", df[df["交易標的"].fillna("").str.contains("房地")])

    # 3. 交易日期區間（交易日 已在 load_one 解析好，這裡不再重複解析整份 raw）
    df = log("3. 交易日期區間",
             df[df["交易日"].between(DATE_START, DATE_END)])

    # 4. 去重
    df = log("4. 編號去重", df.drop_duplicates(subset=["縣市", "編號"], keep="first"))

    # 5. 備註排除
    note = df["備註"].fillna("")
    df = log("5. 備註排除",
             df[~note.str.contains("|".join(NOTE_EXCLUDE), regex=True)])

    # ---- 數值欄位 ----
    # load_one 已經轉成 float64，這裡只補 NaN → 0。加 guard 是因為 NUMERIC_COLS 裡
    # 不在 REQUIRED_COLS 的欄位（如 建物現況格局-*）在某個來源版本可能整批不存在，
    # 而 load_one 那邊有 guard、這裡沒有的話會炸成一個看不出原因的 KeyError。
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    pens = df["交易筆棟數"].fillna("").str.extract(r"土地(\d+)建物(\d+)車位(\d+)")
    df["n_土地"] = pd.to_numeric(pens[0], errors="coerce").fillna(0).astype(int)
    df["n_建物"] = pd.to_numeric(pens[1], errors="coerce").fillna(0).astype(int)
    df["n_車位"] = pd.to_numeric(pens[2], errors="coerce").fillna(0).astype(int)

    # 6. 拆解車位
    df["房屋總價"] = df["總價元"] - df["車位總價元"]
    df["房屋面積坪"] = (df["建物移轉總面積平方公尺"] - df["車位移轉總面積平方公尺"]) / PING_M2
    # 有車位、但車位總價登記為 0 → 無法拆價，標記後保留（見下方診斷）
    df["車位價未拆"] = (df["n_車位"] > 0) & (df["車位總價元"] <= 0)
    log("6. 拆解車位（不刪列）", df)

    # 7. 總價 / 面積 > 0
    df = log("7. 總價與面積 > 0", df[(df["房屋總價"] > 0) & (df["房屋面積坪"] > 0)])

    # 8. 單價區間
    df["單價元每坪"] = df["房屋總價"] / df["房屋面積坪"]
    df = log("8. 單價 5萬~250萬/坪",
             df[df["單價元每坪"].between(PRICE_MIN, PRICE_MAX)])

    # ---- 分析用欄位 ----
    df["鄉鎮市區"] = df["鄉鎮市區"].astype(str).str.strip()
    df["district"] = df["縣市"] + df["鄉鎮市區"]
    df["month"] = df["交易日"].dt.to_period("M")
    df["quarter"] = df["交易日"].dt.to_period("Q")
    df["street"] = [
        street_key(a, c, d)
        for a, c, d in zip(df["土地位置建物門牌"], df["縣市"], df["鄉鎮市區"])
    ]
    df["street_key"] = df["district"] + df["street"]

    steps_df = pd.DataFrame(steps)
    if verbose:
        print(steps_df.to_string(index=False))
    return df.reset_index(drop=True), steps_df


lvr, clean_steps = clean_lvr(raw, BUILDING_MAIN)
print(f"\n清整後：{len(lvr):,} 筆")
''')

md(r"""
### 3.1 稽核：備註排除規則

備註是最容易誤殺的一條規則——關鍵字挑錯會整批砍掉正常交易。這一格把命中內容攤開來看。
""")

code(r'''
# 用與清整相同的前置條件（尚未套日期區間），單看備註規則的影響
_base = raw[raw["建物型態"].fillna("").str.startswith(BUILDING_MAIN)
            & raw["交易標的"].fillna("").str.contains("房地")]
_note = _base["備註"].fillna("")
_hit = _note.str.contains("|".join(NOTE_EXCLUDE), regex=True)

print(f"備註規則命中 {_hit.sum():,} / {len(_base):,} 筆（{_hit.mean():.1%}）\n")
print("各關鍵字命中數：")
for k in NOTE_EXCLUDE:
    print(f"  {k:8s} {_note.str.contains(k).sum():8,}")

print("\n【被排除】的備註 Top 8：")
print(_note[_hit].value_counts().head(8).to_string())
print("\n【保留】的備註 Top 8 — 確認沒有非市場交易漏網：")
print(_note[(~_hit) & (_note != "")].value_counts().head(8).to_string())
print("""
若『被排除』清單裡出現「預售屋、或土地及建物分件登記案件」這類制式敘述，
代表關鍵字設定有誤——那是一般成屋的登記方式註記，不是非市場交易。
""")
''')

code(r'''
# ---- 診斷：車位價未拆的比例與影響 ----
pk = lvr[lvr["n_車位"] > 0]
unsplit = lvr["車位價未拆"]
print(f"含車位交易：{len(pk):,} 筆，其中車位總價登記為 0（無法拆價）："
      f"{unsplit.sum():,} 筆（佔含車位交易 {unsplit.sum() / max(len(pk), 1):.1%}）\n")

print("兩組的單價分布（元/坪）：")
print(
    lvr.assign(組=np.where(unsplit, "車位價未拆", "車位價已拆或無車位"))
       .groupby("組")["單價元每坪"]
       .describe(percentiles=[0.25, 0.5, 0.75])[["count", "25%", "50%", "75%"]]
       .round(0)
       .to_string()
)
print("""
註：計畫 4.1 假設「車位價格有時包在總價裡」。實測顯示這組的單價中位數並未系統性偏高，
    代表多數情況下總價本來就不含該車位（增設車位、另案移轉等）。因此**不做車位價補插**，
    僅以 `車位價未拆` 欄位標記，交由 NB2/NB3 決定是否排除。
    對 NB0 而言影響有限——密度分析數的是交易「筆數」，單價只是用來擋離群值。
""")
''')

# ============================== §3.2 涵蓋率 ==============================
md(r"""
### 3.2 涵蓋率檢查

**這一節擋的是「看起來完全正常的錯誤數字」。**

下載範圍是**發布季**、分析窗口是**交易季**，兩者錯開。若某些發布季的資料沒進到 `raw`，
對應的交易月會被系統性少算——密度統計照樣算得出來、圖也畫得出來，只是憑空少一截。
（試跑時就踩過：月交易量圖出現漂亮的上升趨勢，其實是前半段根本沒被涵蓋。）

§1 的 Layer 1 已擋掉「檔案沒下載到」。這裡擋的是**檔案在、但沒有正確進到 `raw`**：

| 層 | 檢查 | 抓什麼 | 會中止 |
|---|---|---|---|
| **2a** | 每個 csv 都成功載入 | 來源**欄名**改版 → `load_one` 整檔跳過 | ✅ |
| **2a′** | 每檔的列數保留率一致 | 來源**用語**改版 → 列被靜默丟掉 | ✅ |
| **2a″** | 每**組**用語**逐季**都有命中 | 某種型態從某季起改了用語（恰好歸零） | ✅ |
| **2a‴** | 每（縣市 × 組）的季序列無階梯 | 型態用語**部分**改版（殘留非零值，2a″ 全盲） | ✅ |
| **2b** | 每個（發布季 × 縣市）筆數合理 | 檔案截斷、解析後幾乎空的 | ✅ |
| **2b′** | 季序列沒有持續性階梯落差 | 「從某季起一路少算」 | ✅ |
| **2c** | 發布季 × 交易月 貢獻矩陣 | 診斷：坑是哪一季造成的 | — |
| **3** | 逐月交易量離群（雙基準） | 前幾層沒抓到的殘留缺口 | ✅ |
| **4** | 對官方買賣移轉棟數的比值 | **等比例少算**——內部基準一起沉的那一類 | ✅ |
| **5** | 對上次成功執行的季筆數快照 | 「這次比上次少」 | ✅ |

中止行為由 `COVERAGE_STRICT` 控制（見 §0）。Layer 4 抓不到官方資料時預設只警告
（`ANCHOR_REQUIRED = False`），但會把 `external_anchor.status` 標成 `unavailable`——
**不是綠燈**。

**為什麼要 2a′ 和 2b′：** Layer 3 用置中滾動中位數當基準，對**短**缺口很敏銳，
但連續 ≥7 個月的部分性少算會讓基準跟著沉下去而完全失明。而「來源用語從某季起改版、
之後每季都少算一部分」恰好就是這個形狀（實價登錄 2.0 於 110S3 生效，是真實的前例）——
那種缺口 2a、2b、Layer 3 都攔不到，得靠 2a′ 在源頭量保留率、2b′ 在季序列上找階梯。

#### 為什麼需要 Layer 4：內部檢查有一個結構性的洞

除了 2a′ 的**日期解析率**（有絕對期望值：應該接近 1，只差一列英文表頭）之外，
**Layer 1 ~ 3 的基準全都是從資料自己算出來的**。這代表它們對「基準本身被污染」
沒有免疫力——最典型的是**全窗口等比例少算**：所有基準一起沉下去，相對值不變、全層綠燈。

已做的內部緩解：2a′ 的佔比基準改用**同縣市鄰近 7 季的置中滾動中位數**
（原本用全期 p90，但真實資料上臺南的住宅類佔比八年間從 0.29 漲到 0.50，
p90 由後段的高季決定 → 前段每一季都變成「p90 的 56–60%」，第一次跑真實資料就
誤報 4 個檔案；局部基準對趨勢免疫）；2a′ 同時量**主型態佔比**
（住宅類佔比的分子是三組聯集，而 Y 只用主型態組，訊號會被稀釋掉約 0.62 倍）；
2b′ **上下兩個方向都找**（只找下降的話，「序列前段被少算」呈現為上升而永遠看不到）；
2a″ 改成**逐季、以組為單位**的普查；2a‴ 補上「部分改版」——2a″ 只認恰好 0，
殘留一點舊用語就全盲（實測可讓清整後的 lvr 少 25% 而各層全綠）。但這些都還在資料內部。

**Layer 4 換一個分母。** 內政統計月報 4.5「辦理建物所有權登記」按縣市 × 登記月
公布**買賣移轉棟數**，跟我們的資料是兩條獨立的產出鏈。我們每個批次檔的可解析列數
÷ 官方同期棟數，長期應該穩定；比值掉下去就是我們掉了資料。批次檔的「發布季」
對應的是登記季（落後 `ANCHOR_LAG_Q` 季），兩者邊界對不齊，所以形狀檢查用
**四季滾動和**把錯位抵銷掉，絕對水準則用整個窗口的總和（首末季不計）。

官方那張表的欄位位置是寫死的，但每一列都驗
「移轉登記合計 == 買賣＋拍賣＋繼承＋贈與＋其他」——版面改了會 raise，
而不是靜默拿到錯的欄。**驗不出來的是口徑變更**，那會表現成全縣市同時偏移。

**Layer 5 是退一步的版本。** 把這次每個（發布季 × 縣市）的可解析列數存成快照，
下次執行拿來比。不需要任何外部來源，而且只要上一次是好的，連等比例少算都擋得住；
代價是第一次執行沒有基準，且快照會忠實地把一次錯誤的執行當成往後的基準——
所以只在 `coverage.passed` 時才寫入。

#### 還是抓不到什麼

| 仍然抓不到的形狀 | 為什麼 |
|---|---|
| 全窗口等比例少算 25% 以內 | Layer 4 的絕對下限即使照實測校準（0.61），實測解析度也只有 30% |
| 型態用語部分改版但只掉 < 45% | 2a′ 挖空基準的偵測深度下限是 45%（掉 30% → 比值約 0.70，門檻 0.55 接不到） |
| 單一縣市、**只有 1–3 季**、掉一半 | 實測十層全綠。真實的單季變異（主型態佔比局部最小 0.659）比這個訊號還大，任何門檻都分不開。絕對量小（約佔全國主型態 0.6%）且集中在少數月，是刻意接受的殘留 |
| 每年重複出現的長缺口 | 局部基準沉下去、跨年基準被翻轉；滾動四季比值也一起沉，只有絕對水準接得到 |
| 型態用語**全 36 季一起**部分改版 | 2a″ 要恰好 0、2a‴ 要有階梯、2a′ 的局部基準會一起沉。需要「某型態應佔住宅子集多少」這種絕對先驗才擋得住，目前沒有 |
| 官方表的口徑變更 | 算術恆等式與表級合理性檢查都驗不出來，會表現成 Layer 4 全縣市同時偏移 |
| Layer 4 缺席時的上面各項 | 抓不到官方資料就沒有外部錨。**先看 `external_anchor.status` 再看 `passed`** |

**門檻的餘裕會隨資料量反向變化，而且偏差有方向。** 合成資料（量體小）會讓健康資料
看起來離門檻更近（誤報餘裕被低估，保守），同時讓有缺陷的資料看起來離門檻更遠
（偵測邊界被高估，**危險**）。實測：同一個缺陷在 2% 量體下 2a′ 於 f=0.65 就中標，
在 20% 量體下要到 f≈0.99 才中標。所以任何「在合成資料上量到的偵測邊界」都要當成
樂觀值，只有真實資料上量到的誤報餘裕才可信。

**已用真實資料校準的門檻**（2026-08-25、36 個發布季）：
2a′ 的局部門檻 0.60 / 0.45（實測局部最小 0.840 / 0.659）、
`ANCHOR_RATIO_MIN/MAX` = 0.61 / 2.50（實測比值 0.876 ~ 1.404）。
仍未用真實資料校準的：2b 的 0.35、2b′ 與 2a‴ 的 0.55/2.5、Layer 3 的 0.40。
""")

code(r'''
# ---- Layer 2a：每個 csv 都要成功進到 raw ----
if load_failed:
    coverage_fail(
        f"❌ {len(load_failed)} 個檔案載入失敗、未進入 raw",
        "\n".join([f"    {n}" for n in load_failed] +
                  ["", "多半是來源欄名改版（往上找 load_one 印的「缺少欄位」警告）。",
                   "修正 COL_ALIASES / REQUIRED_COLS 後重跑 §2。"]))
else:
    print(f"✅ Layer 2a：{len(files)} 個檔案全數載入 raw")
''')

code(r'''
# ---- Layer 2a′：每個檔案的列數保留率 ----
# load_one 有三道靜默的列篩選（日期格式、日期可解析性、建物型態前綴）。來源改版時，
# 檔案讀得進來、欄位也齊全，但列被大量丟掉——這是 Layer 3 的滾動中位數
# 完全看不見的形狀，只能在源頭量。
ls = pd.DataFrame(LOAD_STATS).sort_values(["縣市", "發布季"]).reset_index(drop=True)
ls["日期格式率"] = ls["日期格式合法"] / ls["原始列數"].replace(0, np.nan)
ls["日期解析率"] = ls["日期可解析"] / ls["原始列數"].replace(0, np.nan)
ls["住宅類佔比"] = ls["住宅類"] / ls["日期可解析"].replace(0, np.nan)
ls["主型態佔比"] = ls["主型態"] / ls["日期可解析"].replace(0, np.nan)

# ---- 絕對檢查（不依賴任何從資料算出來的基準）----
# 兩道日期率都應該接近 1（只差一列英文表頭）。**分開量是必要的**：
#   · 日期格式率 只看字串形狀（\d{6,7}）
#   · 日期解析率 還要求真的解析得出日期——`1051307`（13 月）過得了形狀、解析不出來
# 舊版只量形狀，於是「月份欄變成 13」這一類可以靜默吃掉 15% 的 lvr 而九層全綠。
DATE_MIN = 0.95
# 佔比的絕對下限：低到這個地步代表前綴幾乎對不上任何列。真實資料實測
# 住宅類佔比 min = 0.261、主型態佔比 min = 0.137（都在臺南），各留約 2.5 倍餘裕。
RESI_FLOOR, MAIN_FLOOR = 0.10, 0.05

# ---- 相對檢查：基準是「同縣市鄰近季的置中滾動中位數」，不是全期 p90 ----
# 為什麼換掉全期 p90：真實資料上臺南的住宅類佔比有**長期趨勢**
# （2017 的 0.29 → 2024 的 0.50，1.7 倍）。全期 p90 由後段的高季決定，於是前段
# 每一季都變成「p90 的 56–60%」——第一次跑真實資料就誤報了 4 個檔案，全部是誤報。
# 置中滾動中位數對趨勢免疫（基準跟著趨勢走），對「某幾季突然掉下去」仍然敏銳。
# 代價是「很慢很長的污染」會讓基準一起沉，那一類交給 Layer 2a‴ 的階梯與 Layer 4 的外部錨。
#
# 兩個佔比的門檻**不同**，因為餘裕差很多（真實資料實測的局部最小值）：
#   · 住宅類佔比：最小 0.840（臺南）→ 門檻 0.60 還有 0.24 的餘裕
#   · 主型態佔比：最小 0.659（臺南 114S1，農曆年 + 政策低點撞上上升趨勢的鄰居）
#     → 門檻若也設 0.60 只剩 0.06，跟 Layer 2a′ 舊版被真實趨勢誤報是同一種薄餘裕，
#       所以放到 0.45（餘裕 0.21）。代價是主型態的單季偵測門檻變成「掉超過 55%」。
RESI_WIN = 7
RESI_LOCAL_MIN = 0.60          # 住宅類佔比（三組聯集）
MAIN_LOCAL_MIN = 0.45          # 主型態佔比（只有 MAIN_GROUPS，Y 真正用到的那一組）

# 置中滾動中位數有一個**純算術**的盲區：窗口 7、median 是第 4 小，所以只要缺口連續
# 佔滿 4 季以上，窗口內就有 ≥4 個缺口值 → median 本身就是缺口值 → 比值恰好 1.000，
# **與缺口深度完全無關**（掉 99% 也是 1.000）。而宣稱要接手的 2a‴ 對「中段」的缺口
# 同樣是盲的（level_shift 比的是切點前後的中位數，中段一段不動任何一側）。
# 所以「連續 4–12 季、任意深度」在只有 local 基準時是一條完整的盲帶。
#
# 補法：另一支基準只用「距離 PUNCH_INNER+1 ~ PUNCH_OUTER 季」的鄰居、**挖掉自己附近**。
# 對長度 ≤ 2×PUNCH_INNER+1 的缺口免疫（缺口全落在被挖掉的範圍內），
# 但仍然是局部的，對長期趨勢仍免疫。
#
# (7, 16) 是掃出來的。真實資料的誤報餘裕 vs 注入「臺中住宅大樓 -50%、連續 N 季」的偵測值：
#
# | inner/outer | 真實最小(主型態) | gap4 | gap6 | gap10 |
# |---|---|---|---|---|
# | 3/9   | 0.708 | 0.513 | 0.513 | 0.585 |  ← 偵測值隨缺口變長而升高
# | 5/16  | 0.711 | 0.499 | 0.499 | 0.538 |
# | **7/16** | **0.696** | **0.499** | **0.499** | **0.499** |  ← 偵測值與缺口長度無關
# | 7/20  | 0.671 | 0.514 | 0.514 | 0.514 |
# | 9/20  | 0.653 | 0.506 | 0.506 | 0.506 |  ← 誤報餘裕開始掉
#
# 7/16 幾乎不花誤報餘裕（0.708 → 0.696）就讓偵測力**不再隨缺口長度衰減**。
PUNCH_INNER, PUNCH_OUTER = 7, 16
PUNCH_MIN_N = 4                # 挖空後至少要有這麼多鄰居才算得出基準
# 挖空基準的門檻**和局部基準分開**，因為兩支的誤報側完全不同（實測最小值：
# 局部 住宅類 0.840 / 主型態 0.659；挖空 住宅類 0.744 / 主型態 0.696）。
RESI_PUNCH_MIN = 0.60          # 住宅類（餘裕 1.24×）
MAIN_PUNCH_MIN = 0.55          # 主型態（餘裕 1.27×），偵測「掉 ≥ 45%」
# 為什麼偵測要靠**主型態**而不是住宅類：住宅大樓只佔住宅子集約 67%，所以
# 「住宅大樓 -50%」在住宅類佔比上只剩 -33%。實測住宅類挖空的偵測值 0.730 而
# 誤報最小值 0.744——中間只有 0.014 的縫，**任何門檻都做不到**。
# 這正是這一版把「主型態佔比」單獨拉出來量的理由。


def local_ratio(g: pd.Series) -> pd.Series:
    """本季 / 鄰近 RESI_WIN 季（含自己）的中位數。對趨勢免疫、對短缺口敏銳。"""
    base = g.rolling(RESI_WIN, center=True, min_periods=3).median()
    return g / base.replace(0, np.nan)


def punched_ratio(g: pd.Series) -> pd.Series:
    """本季 / 「距離 PUNCH_INNER+1 ~ PUNCH_OUTER 季」鄰居的中位數（挖掉自己附近）。
    這一支的存在理由是 local_ratio 對連續 ≥4 季的缺口恆等於 1.000（見上方說明）。"""
    v = g.to_numpy(dtype=float)
    n = len(v)
    out = np.full(n, np.nan)
    for i in range(n):
        lo, hi = max(0, i - PUNCH_OUTER), min(n, i + PUNCH_OUTER + 1)
        idx = [j for j in range(lo, hi) if abs(j - i) > PUNCH_INNER]
        nb = v[idx]
        nb = nb[np.isfinite(nb)]
        if len(nb) >= PUNCH_MIN_N:
            base = float(np.median(nb))
            if base > 0:
                out[i] = v[i] / base
    return pd.Series(out, index=g.index)


for _c in ("住宅類佔比", "主型態佔比"):
    ls[_c + "_local"] = ls.groupby("縣市")[_c].transform(local_ratio)
    ls[_c + "_punch"] = ls.groupby("縣市")[_c].transform(punched_ratio)

print("列數保留率（各縣市）\n")
print(ls.groupby("縣市")
        .agg(日期格式率中位=("日期格式率", "median"),
             日期解析率中位=("日期解析率", "median"),
             住宅類佔比最小=("住宅類佔比", "min"),
             住宅類佔比中位=("住宅類佔比", "median"),
             住宅類佔比最大=("住宅類佔比", "max"),
             住宅類局部最小=("住宅類佔比_local", "min"),
             住宅類挖空最小=("住宅類佔比_punch", "min"),
             主型態佔比最小=("主型態佔比", "min"),
             主型態局部最小=("主型態佔比_local", "min"),
             主型態挖空最小=("主型態佔比_punch", "min"))
        .reindex(CITY_ORDER).round(3).to_string())
print(f"\n門檻校準參考（真實資料實測：局部 0.840/0.659、挖空 0.744/0.696）：")
print(f"  局部 住宅類 {RESI_LOCAL_MIN}／主型態 {MAIN_LOCAL_MIN}；"
      f"挖空 住宅類 {RESI_PUNCH_MIN}／主型態 {MAIN_PUNCH_MIN}。越接近越容易誤報。")
print("  **偵測「連續數季各掉一半」靠的是主型態的挖空那一支**"
      "（局部基準對連續 ≥4 季恆等於 1.000，住宅類則被稀釋到偵測不到）。")
print("  「佔比最小/最大」差得遠**不再**是問題（那是長期趨勢，局部基準對它免疫）；")
print(f"  但若有縣市的局部最小低於門檻 + 0.10，門檻就得放寬。")

# 兩支相對基準任一中標就報。**兩支都算不出來（全 NaN）也要中標**——
# 短 SEASONS 時 min_periods 讓 _local 全是 NaN，舊版沒 guard 它，於是相對檢查
# 整支靜默失效卻照樣印 ✅。
_rel_cols = ["住宅類佔比_local", "住宅類佔比_punch", "主型態佔比_local", "主型態佔比_punch"]
_rel_min = {"住宅類佔比_local": RESI_LOCAL_MIN, "住宅類佔比_punch": RESI_PUNCH_MIN,
            "主型態佔比_local": MAIN_LOCAL_MIN, "主型態佔比_punch": MAIN_PUNCH_MIN}
_bad = ((ls["日期格式率"] < DATE_MIN) | (ls["日期解析率"] < DATE_MIN)
        | (ls["住宅類佔比"] < RESI_FLOOR) | (ls["主型態佔比"] < MAIN_FLOOR)
        | ls["日期格式率"].isna() | ls["日期解析率"].isna()
        | ls["住宅類佔比"].isna() | ls["主型態佔比"].isna())
for _c in _rel_cols:
    _bad = _bad | (ls[_c] < _rel_min[_c])
n_rel_checked = int(ls[_rel_cols].notna().any(axis=1).sum())
if n_rel_checked < len(ls):
    print(f"⚠️  {len(ls) - n_rel_checked} 個檔案算不出任何相對基準"
          f"（季數太少：需要 local ≥ 3 個鄰居或 punch ≥ {PUNCH_MIN_N} 個）"
          "——那些檔案的相對檢查等於沒跑，已計入失敗。")
    _bad = _bad | ls[_rel_cols].notna().sum(axis=1).eq(0)
bad_ret = ls[_bad]
if len(bad_ret):
    coverage_fail(
        f"❌ {len(bad_ret)} 個檔案的列數保留率異常",
        "\n".join([f"    {r['檔名']}：{r['原始列數']:,} 列 → 格式合法 {r['日期格式合法']:,}"
                   f"（{r['日期格式率']:.1%}）→ 可解析 {r['日期可解析']:,}"
                   f"（{r['日期解析率']:.1%}）→ 住宅類 {r['住宅類']:,}"
                   f"（佔比 {r['住宅類佔比']:.1%}、局部 {r['住宅類佔比_local']:.0%}"
                   f"、挖空 {r['住宅類佔比_punch']:.0%}）"
                   f"、主型態 {r['主型態']:,}"
                   f"（佔比 {r['主型態佔比']:.1%}、局部 {r['主型態佔比_local']:.0%}"
                   f"、挖空 {r['主型態佔比_punch']:.0%}）"
                   for _, r in bad_ret.sort_values("檔名").head(30).iterrows()] +
                  ([f"    …另有 {len(bad_ret) - 30} 個"] if len(bad_ret) > 30 else []) +
                  ["",
                   "日期**格式率**偏低 → 交易年月日 的字串格式變了，檢查 load_one 的 fullmatch。",
                   "日期**解析率**偏低而格式率正常 → 格式對但值不合法（例如月份 13），"
                   "看得到的症狀是 lvr 少了一段而 raw 正常。",
                   "住宅類／主型態**佔比**低於絕對下限 → 建物型態的用語整批對不上，"
                   "看 Layer 2a″ 的逐季普查表。",
                   "**局部**比值偏低 → 那幾季突然掉下去（基準是鄰近 "
                   f"{RESI_WIN} 季的中位數）。",
                   "**挖空**比值偏低 → 連續好幾季一起掉（基準挖掉自己 ±"
                   f"{PUNCH_INNER} 季、只用距離 {PUNCH_INNER + 1}–{PUNCH_OUTER} 季的鄰居；"
                   "局部基準對連續 ≥4 季的缺口恆等於 1.000，這一支專門補那個盲區）。",
                   f"（門檻：局部 住宅類 {RESI_LOCAL_MIN}／主型態 {MAIN_LOCAL_MIN}；"
                   f"挖空 住宅類 {RESI_PUNCH_MIN}／主型態 {MAIN_PUNCH_MIN}）",
                   "這些都會讓對應的交易月被系統性少算，而統計量看起來完全正常。"]))
else:
    print(f"\n✅ Layer 2a′：{len(ls)} 個檔案的日期率與型態佔比都正常"
          f"（局部 {RESI_WIN} 季門檻 {RESI_LOCAL_MIN}/{MAIN_LOCAL_MIN}；"
          f"挖空 ±{PUNCH_INNER}–{PUNCH_OUTER} 季門檻 {RESI_PUNCH_MIN}/{MAIN_PUNCH_MIN}）")
''')

code(r'''
# ---- Layer 2a″：建物型態用語的逐季普查（以「組」為單位）----
# 保留率型的門檻對「某個用語從此對不上」本質上接不到——跨縣市的正常佔比差異
# （臺北 0.77 vs 臺南 0.32）比這個缺陷造成的訊號還大。名稱普查是絕對檢查：
# 不吃比例、不吃跨縣市差異，直接問「這一組命中了幾列」。
#
# **逐季**而不是全域：全域普查只問「有沒有命中過」，所以對最現實的劇本完全失明——
# 來源在某一季改用語、之後每季都少算一部分，只要還有任何一季留著舊用語，
# 全域命中數就 > 0。實價登錄 2.0 於 110S3 生效，正是這個形狀。
#
# **以組為單位**（見 §0 的 BUILDING_GROUPS）：改版後同一種建物型態的新舊用語各只在
# 半段序列出現，逐「前綴」要求會讓修好之後仍然硬中止、永遠沒有不動點。
# 檢查的是每一組**逐季的聯集**非零，所以「把新用語加進同一組」是可通過的修法。
MISSING_SENTINEL = -1

if not BUILDING_SEEN:
    # 只有所有檔案都載入失敗才會發生，而那已由 Layer 2a 攔下（除非 STRICT=False）
    print("⚠️  Layer 2a″ 跳過：BUILDING_SEEN 是空的（沒有任何檔案成功載入）")
    bs = pd.Series(dtype="int64")
    hits = {g: 0 for g in BUILDING_GROUPS}
    prefix_hits = {p: 0 for p in BUILDING_RELAXED}
    dead_groups = list(BUILDING_GROUPS)
    dead_prefixes_in_groups = list(BUILDING_RELAXED)
    prefix_runs, prefix_thin = [], []
    grp_season = pd.DataFrame(columns=list(BUILDING_GROUPS))
    grp_city_season = pd.DataFrame()
    seasons_seen = []
else:
    bt = pd.Series(BUILDING_SEEN, dtype="int64")
    bt.index = pd.MultiIndex.from_tuples(list(bt.index),
                                         names=["發布季", "縣市", "建物型態"])
    _names = bt.index.get_level_values("建物型態")
    bs = bt.groupby("建物型態").sum().sort_values(ascending=False)      # 全域用語普查

    def group_mask(g: str):
        return _names.str.startswith(tuple(BUILDING_GROUPS[g]))

    hits = {g: int(bt[group_mask(g)].sum()) for g in BUILDING_GROUPS}
    prefix_hits = {p: int(bs[bs.index.str.startswith(p)].sum()) for p in BUILDING_RELAXED}

    _seen = set(bt.index.get_level_values("發布季"))
    seasons_seen = [ss for ss in SEASONS if ss in _seen]
    _gone = [ss for ss in SEASONS if ss not in _seen]

    # 逐季（全國）與逐季 × 縣市 的組命中數。**缺季填哨兵 -1 而不是 0**：
    # 缺季是 Layer 1 的事；填 0 會讓缺季偽裝成用語改版（方向完全錯），
    # 還會把缺口兩側兩段真的零命中區間接成一段、季數也算錯。
    grp_season = pd.DataFrame(index=pd.Index(SEASONS, name="發布季"))
    for g in BUILDING_GROUPS:
        v = (bt[group_mask(g)].groupby("發布季").sum()
             .reindex(SEASONS).fillna(0).astype("int64"))
        if _gone:
            v.loc[_gone] = MISSING_SENTINEL
        grp_season[g] = v
    grp_city_season = pd.DataFrame(
        {g: (bt[group_mask(g)].groupby(["縣市", "發布季"]).sum()
             .reindex(pd.MultiIndex.from_product([CITY_ORDER, seasons_seen],
                                                 names=["縣市", "發布季"]))
             .fillna(0).astype("int64"))
         for g in BUILDING_GROUPS})

    print(f"建物型態用語普查（型態篩選前，共 {len(bs)} 種用語 / {bs.sum():,} 列）\n")
    print(bs.head(15).to_string())
    print(f"\n各組命中數：{hits}")
    print(f"各前綴命中數：{prefix_hits}")
    print(f"嚴格版 MAIN_GROUPS = {list(MAIN_GROUPS)} → 前綴 {list(BUILDING_MAIN)}")
    print(f"\n逐季命中數（{len(seasons_seen)} 季有資料"
          + (f"、{len(_gone)} 季缺檔以 {MISSING_SENTINEL} 標記" if _gone else "") + "）\n")
    print(grp_season.T.to_string())

    dead_groups = [g for g, n in hits.items() if n == 0]
    # 組內「全期零命中」的單一前綴：**刻意不當成失敗**——移到「組」語意的重點就是
    # 允許前綴半死（改版後新舊用語各只在半段序列出現），而「預先登記未來用語」
    # 也是合法用法。但完全沒有出口就無法跟「打錯字」區分，所以印出來。
    dead_prefixes_in_groups = [p_ for p_, n in prefix_hits.items() if n == 0]
    if dead_prefixes_in_groups:
        print(f"\n⚠️  {len(dead_prefixes_in_groups)} 個前綴全期零命中："
              f"{dead_prefixes_in_groups}")
        print("    不當成失敗（可能是預先登記的未來用語）。但如果不是刻意的，"
              "那就是打錯字——真實改版的舊前綴在前段一定還有命中數，不會是 0。")

    def zero_runs(v: np.ndarray) -> list:
        """零值的連續區間 [(起位置, 迄位置)]，兩端皆含。
        哨兵 -1 不等於 0，所以缺季會自然把區間切斷——不需要額外處理。"""
        runs, i = [], 0
        while i < len(v):
            if v[i] == 0:
                j = i
                while j + 1 < len(v) and v[j + 1] == 0:
                    j += 1
                runs.append((i, j))
                i = j + 1
            else:
                i += 1
        return runs

    # 量太小的組某季自然為 0 是正常的，只對有量的組查階梯。
    # 判定用**季 p90 而不是「非零季的中位數」**：後者會被健康季拉高，於是
    # 「某組 90% 被改掉、殘留量掉到門檻以下」會被歸成 thin 而整個關掉偵測——
    # 也就是缺陷越嚴重反而越安靜（把 f 從 0.9 調到 1.0 才會中標）。
    prefix_runs, prefix_thin = [], []
    for g in BUILDING_GROUPS:
        if g in dead_groups:
            continue                      # 全季零命中由 dead_groups 處理，不重複報
        v = grp_season[g].to_numpy()
        obs = v[v >= 0]
        p90 = float(np.percentile(obs, 90)) if len(obs) else 0.0
        if p90 < PREFIX_MIN_MEDIAN:
            prefix_thin.append({"組": g, "季 p90": round(p90, 1)})
            continue
        for a, b in zero_runs(v):
            if b - a + 1 >= PREFIX_ZERO_RUN:
                prefix_runs.append(
                    {"組": g, "起": SEASONS[a], "迄": SEASONS[b],
                     "季數": b - a + 1, "季 p90": round(p90, 1),
                     "位置": ("序列開頭" if a == 0 else
                              "序列結尾" if b == len(v) - 1 else "序列中段")})

if dead_groups or prefix_runs:
    detail = []
    if dead_groups:
        detail += [f"  · 全部發布季都零命中的組：{'、'.join(dead_groups)}"
                   f"（前綴 {[p for g in dead_groups for p in BUILDING_GROUPS[g]]}）",
                   "    來源的建物型態用語改版了，這一組的前綴對不上任何一列。"]
    if prefix_runs:
        detail += [f"  · {len(prefix_runs)} 段持續零命中的季區間："]
        detail += [f"    {r['組']}：{r['起']} ~ {r['迄']}（{r['季數']} 季，{r['位置']}），"
                   f"其他季 p90 {r['季 p90']:,.0f} 列" for r in prefix_runs]
        detail += ["    「序列結尾」＝來源從那一季起改了用語；"
                   "「序列開頭」＝那一段用的是更早的舊用語，我們接不到。"]
    coverage_fail(
        "❌ Layer 2a″：建物型態用語普查未通過"
        f"（零命中組 {len(dead_groups)} 個、持續零命中區間 {len(prefix_runs)} 段）",
        "\n".join(detail + [
            "",
            "  修法：上面的逐季命中表與全域用語表就是依據——找出來源改成了什麼，"
            "把**新用語加進 §0 的 BUILDING_GROUPS 同一組**（不要另開一組，"
            "另開一組的話新舊各有一半的季是 0，永遠修不好）。",
            f"  若受影響的是 MAIN_GROUPS（{list(MAIN_GROUPS)}），"
            "整份密度分析的 Y 會被削掉一段；",
            "  若只是華廈／公寓，影響範圍是 §6 的放寬敏感度分析。"]))
else:
    print(f"\n✅ Layer 2a″：{len(BUILDING_GROUPS)} 組在 {len(seasons_seen)} 季中"
          f"都沒有連續 {PREFIX_ZERO_RUN} 季以上的零命中"
          + (f"（量太小、只做全域普查：{'、'.join(r['組'] for r in prefix_thin)}）"
             if prefix_thin else ""))
''')

code(r'''
# ---- Layer 2a‴：每個（縣市 × 建物型態組）季序列的階梯 ----
# 2a″ 只認「恰好 0」。真實的改版很少那麼乾淨：過渡期兩種用語並存、各地政事務所
# 採用時間不一、只有部分縣市改——殘留一點舊用語，命中數就是個小的非零值，2a″ 全盲。
# 實測（合成資料、量體接近真實）：住宅大樓 50% 的列改用語、影響 18/36 季，
# 清整後的 lvr 少了 25%，而九層一個都沒中標。2a′ 也接不到，因為它的
# 住宅類佔比把訊號稀釋了（分子是三組的聯集，而 Y 只用主型態組）。
#
# 這一層對每個 (縣市 × 組) 的季命中數跑 Layer 2b′ 的 level_shift，實測接得到 f ≥ 0.45。
# 用**筆數**而不是佔比：臺南的住宅類佔比八年間從 0.29 漲到 0.50（真實市場變化），
# 佔比序列會誤報；筆數的前後段中位數比實測 1.6–1.9 倍，離上升門檻仍有餘裕。
grp_steps, grp_thin, grp_extremes, grp_unchecked = [], [], [], []
if len(grp_city_season):
    for city in CITY_ORDER:
        if city not in grp_city_season.index.get_level_values("縣市"):
            continue
        for g in BUILDING_GROUPS:
            v = grp_city_season.loc[city, g].to_numpy(dtype=float)
            p90 = float(np.percentile(v, 90)) if len(v) else 0.0
            if p90 < PREFIX_MIN_MEDIAN:
                grp_thin.append({"縣市": city, "組": g, "季 p90": round(p90, 1)})
                continue
            if len(v) < 8:
                # level_shift 兩側各需 4 點。**要記帳**：不記的話短 SEASONS 下
                # 每個組合都被跳過，卻印出「✅ 0 個序列均無持續性階梯」。
                grp_unchecked.append({"縣市": city, "組": g, "季數": len(v)})
                continue
            down, up = level_shift(v)
            if down:
                grp_extremes.append({"縣市": city, "組": g, "方向": "下降",
                                     "比值": round(down[3], 3)})
                if down[3] < GROUP_STEP_DOWN:
                    grp_steps.append({"縣市": city, "組": g,
                                      "切點季": seasons_seen[down[0]],
                                      "前段中位": round(down[1], 1),
                                      "後段中位": round(down[2], 1),
                                      "比值": round(down[3], 3), "方向": "該季起偏低"})
            if up:
                grp_extremes.append({"縣市": city, "組": g, "方向": "上升",
                                     "比值": round(up[3], 3)})
                if up[3] > GROUP_STEP_UP:
                    grp_steps.append({"縣市": city, "組": g,
                                      "切點季": seasons_seen[up[0]],
                                      "前段中位": round(up[1], 1),
                                      "後段中位": round(up[2], 1),
                                      "比值": round(up[3], 3), "方向": "該季之前偏低"})

    _ex = pd.DataFrame(grp_extremes)
    if len(_ex):
        print(f"(縣市 × 組) 季序列的階梯極值（門檻：下降 {GROUP_STEP_DOWN}、"
              f"上升 {GROUP_STEP_UP}）\n")
        print(_ex.pivot_table(index="縣市", columns=["組", "方向"], values="比值")
                 .reindex(CITY_ORDER).round(3).to_string())
        print("\n門檻校準參考：上表的「下降」離 "
              f"{GROUP_STEP_DOWN}、「上升」離 {GROUP_STEP_UP} 越近，誤判風險越高。"
              "\n  真實資料實測：上升最大 1.885（臺南華廈）、"
              "下降最小 0.681（臺中公寓，公寓長期衰退）。"
              "\n  **下降側會隨 SEASONS 延長繼續縮**（公寓還在跌），每次延長窗口都要回頭看這張表。")
    if grp_thin:
        print(f"\n量太小、跳過階梯檢查的組合 {len(grp_thin)} 個"
              f"（季 p90 < {PREFIX_MIN_MEDIAN}）："
              + "、".join(f"{r['縣市']}{r['組']}({r['季 p90']:.0f})" for r in grp_thin[:12])
              + (" …" if len(grp_thin) > 12 else ""))

if grp_unchecked and not grp_steps:
    coverage_fail(
        f"❌ Layer 2a‴：{len(grp_unchecked)} 個（縣市 × 組）季數不足 8，階梯檢查沒跑",
        "\n".join([f"    {r['縣市']} {r['組']}：只有 {r['季數']} 季"
                    for r in grp_unchecked[:12]] +
                   ["", "level_shift 兩側各需 4 個點。SEASONS 太短時這一層等於沒有——",
                    "而它擋的是『型態用語部分改版』，是十層裡唯一接得到那個形狀的。",
                    "要嘛把 SEASONS 補齊，要嘛明確接受（COVERAGE_STRICT=False）。"]))

if grp_steps:
    coverage_fail(
        f"❌ Layer 2a‴：{len(grp_steps)} 個（縣市 × 組）的季命中數出現持續性階梯",
        "\n".join([f"    {r['縣市']} {r['組']}：{r['切點季']} 為切點，"
                   f"季中位 {r['前段中位']:,.0f} → {r['後段中位']:,.0f}"
                   f"（{r['比值']:.0%}，{r['方向']}）" for r in grp_steps[:30]] +
                  ([f"    …另有 {len(grp_steps) - 30} 個"] if len(grp_steps) > 30 else []) +
                  ["",
                   "這是「某個建物型態的用語從某季起部分改掉」的形狀——2a″ 只認恰好 0，",
                   "所以殘留一點舊用語就會從那裡溜過去，只有這一層接得到。",
                   "先看 Layer 2a″ 印的逐季命中表與全域用語表，確認來源改成了什麼；",
                   "若確認是真實的市場變化（例如某縣市某型態的供給結構改變），",
                   f"把 GROUP_STEP_DOWN（{GROUP_STEP_DOWN}）／"
                   f"GROUP_STEP_UP（{GROUP_STEP_UP}）放寬，",
                   "但要先看上面印的極值表確認新門檻的餘裕。",
                   "切點只是「落差最大的分割位置」，實測會偏移最多 5 季。"]))
elif len(grp_city_season):
    print(f"\n✅ Layer 2a‴：{len(grp_extremes) // 2} 個（縣市 × 組）季序列均無持續性階梯")
''')

code(r'''
# ---- 涵蓋率的量測基底 ----
# 用 raw（住宅類全型態）而非 lvr（僅電梯大樓）：涵蓋率是「資料有沒有到齊」的性質，
# 與建物型態篩選無關；而全型態的筆數多好幾倍，月比值更穩，
# 較不會把臺南這種樣本本來就薄的縣市的正常波動誤判成缺口。
_m = raw["交易日"].dt.to_period("M")
_in = _m.notna() & (_m >= MONTHS[0]) & (_m <= MONTHS[-1])
# 縣市／發布季只有 6 與 36 種值，在這張衍生表裡轉 category；object dtype 在 250 萬列的
# 真實資料上會多吃數百 MB。（raw 本身不轉——見 load_one 的說明。）
cov_base = pd.DataFrame({
    "month": _m[_in],
    "縣市": raw["縣市"][_in].astype("category"),
    "發布季": raw["發布季"][_in].astype("category"),
})
_n_bad_date = int((~_m.notna()).sum())
print(f"涵蓋率基底：{len(cov_base):,} 筆住宅類交易落在分析窗口內"
      f"（raw 共 {len(raw):,} 筆；窗口外 {len(raw) - len(cov_base) - _n_bad_date:,} 筆、"
      f"日期無法解析 {_n_bad_date:,} 筆）")
del _m, _in
''')

code(r'''
# ---- 診斷：raw 的 (縣市, 編號) 重複率 ----
# 「同一批資料被載入兩次」的直接量測。不靠 Layer 4 的比值上限，因為真實比值在
# 六都之間就從 0.88 到 1.40（結構性差異），上限沒有足夠的解析度。
# 注意 clean_lvr 第 4 步本來就會對 (縣市, 編號) 去重，所以重複不影響決策數字，
# 但重複率突然變高是「檔案被重複解壓」的訊號。
_dup_rate = float(raw.duplicated(subset=["縣市", "編號"]).mean())
print(f"raw 的 (縣市, 編號) 重複率：{_dup_rate:.2%}"
      f"（一筆交易的土地/建物/車位分列登記時本來就會重複；"
      f"接近 50% 才代表整批被載入兩次）\n")

# ---- Layer 2b：每個（發布季 × 縣市）的筆數 ----
sc = (raw.groupby(["發布季", "縣市"], observed=True).size()
         .unstack("縣市").reindex(index=SEASONS, columns=CITY_ORDER).fillna(0).astype(int))
base_sc = sc.median(axis=0)

print("每個發布季 × 縣市的住宅類原始筆數（未套日期區間）\n")
print(sc.to_string())
print(f"\n各縣市每季中位筆數：{base_sc.astype(int).to_dict()}")

# 整個縣市一筆都沒有 → 縣市名對不上（臺/台 錯字、CITY_CODES 改版之類）。
# 這種情況不能只是跳過，否則下面的中位數基準是 0，反而印出假的綠燈。
dead_city = [c for c in sc.columns if base_sc[c] <= 0]
if dead_city:
    coverage_fail(
        f"❌ {len(dead_city)} 個縣市在所有發布季都是 0 筆：{'、'.join(dead_city)}",
        "    raw 的「縣市」值對不上 CITY_ORDER——多半是臺/台 用字或 CITY_CODES 改版。\n"
        f"    raw 實際出現的縣市：{sorted(raw['縣市'].unique())}")

# 低於同縣市季中位數的 35% 視為異常。發布季之間的量體差異來自市場循環與政策
# （房地合一 2.0、平均地權條例），實測落在中位的 0.76–1.42 倍；掉到 35% 以下
# 幾乎一定是檔案本身有問題。門檻不敢再放寬，是因為 20% 與 Layer 3 的 40%
# 之間會留一條「截斷到 20–25%」兩邊都躲掉的縫。
bad_sc = [(ss, c, int(sc.loc[ss, c]), float(base_sc[c]))
          for ss in sc.index for c in sc.columns          # 以「季」為主序，同一壞季的縣市列在一起
          if base_sc[c] > 0 and sc.loc[ss, c] < 0.35 * base_sc[c]]

if bad_sc:
    coverage_fail(
        f"❌ {len(bad_sc)} 個（發布季 × 縣市）筆數異常偏低（< 同縣市季中位數的 35%）",
        "\n".join([f"    {ss} {c}：{n:,} 筆（該縣市季中位 {b:,.1f}，門檻 {0.35 * b:,.1f}）"
                    for ss, c, n, b in bad_sc] +
                  ["", "檔案可能截斷或解析後幾乎為空。",
                   "刪掉該季的 .done 標記與 csv、重跑 §1 重新下載。"]))
elif not dead_city:          # dead_city 已在上面示警，別再印綠燈蓋掉它
    print(f"\n✅ Layer 2b：{sc.size} 個（發布季 × 縣市）組合筆數均正常")
''')

code(r'''
# ---- Layer 2b′：季序列的持續性階梯落差 ----
# level_shift 定義在 §0（Layer 2a‴ 比這裡更早用到它）
# STEP_RATIO / STEP_UP_RATIO 定義在 §0（Layer 2a‴ 比這裡更早用到）
steps, steps_unchecked = [], []
for c in list(CITY_ORDER) + ["六都合計"]:
    series = sc.sum(axis=1) if c == "六都合計" else sc[c]
    if len(series) < 8:
        # 同 2a‴：不記帳的話短 SEASONS 下整層靜默失效卻印綠燈
        steps_unchecked.append(c)
        continue
    down, up = level_shift(series.to_numpy(dtype=float))
    if down and down[3] < STEP_RATIO:
        steps.append((c, SEASONS[down[0]], down[1], down[2], "該季起偏低"))
    if up and up[3] > STEP_UP_RATIO:
        steps.append((c, SEASONS[up[0]], up[1], up[2], "該季之前偏低"))

if steps:
    coverage_fail(
        f"❌ {len(steps)} 個序列出現持續性的階梯落差",
        "\n".join([f"    {c}：{ss} 為切點，季中位 {b:,.0f} → {a:,.0f}"
                   f"（{a / b:.0%}，{w}）" for c, ss, b, a, w in steps] +
                  ["", "「某一段一路少算」的典型形狀。可能原因：",
                   "  · 來源用語改版（實價登錄 2.0 於 110S3 生效）→ 看 Layer 2a′ 的保留率",
                   "  · 那一段的季根本沒下載完整 → 看 Layer 1",
                   "注意這種缺口 Layer 3 的滾動中位數抓不到，別看它沒中標就放心。",
                   "切點只是「落差最大的分割位置」，實測會偏移最多 5 季"
                   "（受影響季數離全長一半越遠、偏得越多）；",
                   "要精確定位是哪幾季，看 Layer 2a′ 逐檔列出的保留率。"]))
elif steps_unchecked:
    coverage_fail(
        f"❌ Layer 2b′：{len(steps_unchecked)} 個序列的季數不足 8，階梯檢查沒跑",
        f"    {steps_unchecked}\n"
        "    level_shift 兩側各需 4 個點。SEASONS 太短時這一層等於沒有。")
else:
    print(f"✅ Layer 2b′：{len(CITY_ORDER) + 1} 個季序列均無持續性落差"
          f"（下降門檻 {STEP_RATIO:.0%}、上升門檻 {STEP_UP_RATIO:.1f}×）")
''')

code(r'''
# ---- Layer 2c：發布季 × 交易月 貢獻矩陣（診斷，不中止）----
cov = cov_base.groupby(["month", "發布季"], observed=True).size().rename("n").reset_index()
per_month = (cov.groupby("month", observed=True)
                .agg(貢獻發布季數=("發布季", "nunique"), 筆數=("n", "sum"))
                .reindex(MONTHS).fillna(0).astype(int))

print(f"分析窗口 {MONTHS[0]} ~ {MONTHS[-1]}（{len(MONTHS)} 個月）\n")
print(f"  每月的貢獻發布季數：中位 {per_month['貢獻發布季數'].median():.0f}"
      f"、最少 {per_month['貢獻發布季數'].min()}、最多 {per_month['貢獻發布季數'].max()}")
print(f"  完全沒有交易的月份：{int((per_month['筆數'] == 0).sum())} 個")

thin = per_month[per_month["貢獻發布季數"] < per_month["貢獻發布季數"].median()]
if len(thin):
    print(f"\n貢獻發布季數低於中位的月份（{len(thin)} 個，診斷用）：")
    print(thin.head(24).to_string())

print("\n每個發布季實際餵進窗口的交易月範圍：")
span = (cov.groupby("發布季", observed=True)["month"]
           .agg(最早="min", 最晚="max", 月數="nunique", ).reindex(SEASONS))
print(span.to_string())
''')

code(r'''
# ---- Layer 3：逐月交易量離群 ----
def _loo_median(g: pd.Series) -> pd.Series:
    """留一中位數：每個值對上「同一個日曆月、其他年份」的中位數。"""
    v = g.to_numpy(dtype=float)
    return pd.Series([np.median(np.delete(v, i)) for i in range(len(v))], index=g.index)


def monthly_flags(counts: pd.Series, scope: str) -> pd.DataFrame:
    """兩個獨立基準各判一次，任一中標即標記。門檻同為 COVERAGE_MIN_RATIO。

    **局部基準**（置中 13 個月滾動中位數）：對短缺口很敏銳，但連續 ≥7 個月的
    部分性少算會讓基準跟著沉下去而完全失明——13 月窗內只要有 7 個低值，
    中位數本身就是低值。單靠它會漏掉「從某季起一路少算」這種最現實的形狀。

    **同月跨年基準**（同一日曆月在其他 7 年的中位數）：補上上面的盲點。
    連續一整年以上的缺口用局部基準看不出來，用這個一定現形。代價是它會被
    長期趨勢稀釋（2017 低點 vs 2021 高點約 0.7 倍），所以同樣用 0.40 的寬門檻。

    0.40 的門檻是為了只抓「掉一大塊」等級的缺口：農曆年的真實低點實測落在基準的
    0.53–0.70（臺南最薄、最低 0.527），兩個基準下都不會誤報。注意農曆年落在
    **1 月**的年份（2017、2020、2023）會讓跨年基準的餘裕從 0.45 縮到 0.17——
    仍然安全，但門檻不宜再往上調。

    **兩個基準都是「相對」判斷，共有一組盲點**（見本節末的已知限制）：
      · 全窗口等比例少算 —— 兩個基準一起沉，抓不到
      · 每年重複出現的長缺口（如每年 6–12 月都少算）—— 局部基準沉下去，
        跨年基準因為「其他年份同月也低」而完全被翻轉，比值甚至會 > 1
      · 跨年基準的靈敏度隨市場循環浮動約 1.5 倍：缺口落在高點年時，
        LOO 中位被其他較低年份拉低，有效門檻等於從 0.40 縮到約 0.33
    """
    s = counts.reindex(MONTHS).fillna(0).astype(float)

    base_local = s.rolling(13, center=True, min_periods=7).median().bfill().ffill()
    base_yoy = (pd.Series(s.values, index=MONTHS.month)
                  .groupby(level=0).transform(_loo_median).to_numpy())

    def _ratio(base) -> np.ndarray:
        b = np.asarray(base, dtype=float)
        return np.divide(s.to_numpy(), b, out=np.full(len(s), np.nan), where=b > 0)

    out = pd.DataFrame({"範圍": scope, "交易月": MONTHS.astype(str), "筆數": s.values,
                        "局部基準": np.asarray(base_local, dtype=float),
                        "局部比值": _ratio(base_local),
                        "同月跨年基準": base_yoy,
                        "同月跨年比值": _ratio(base_yoy)})
    out["比值"] = out[["局部比值", "同月跨年比值"]].min(axis=1)
    out["旗標"] = ((out["局部比值"] < COVERAGE_MIN_RATIO)
                   | (out["同月跨年比值"] < COVERAGE_MIN_RATIO)
                   | (out["筆數"] == 0))
    return out


flags = pd.concat(
    [monthly_flags(cov_base.groupby("month", observed=True).size(), "六都整體")] +
    [monthly_flags(cov_base[cov_base["縣市"] == c].groupby("month", observed=True).size(), c)
     for c in CITY_ORDER],
    ignore_index=True)
flagged = flags[flags["旗標"]]

print(f"逐月交易量檢查：{len(flags)} 個（範圍 × 月）組合，"
      f"兩個基準的門檻同為 {COVERAGE_MIN_RATIO:.0%}\n")
print("各範圍的最低月比值（越接近 1 越健康；農曆年通常落在 0.55–0.70）：")
print(flags.groupby("範圍")[["局部比值", "同月跨年比值"]].min()
           .reindex(["六都整體"] + CITY_ORDER).round(2).to_string())

if len(flagged):
    coverage_fail(
        f"❌ {len(flagged)} 個（範圍 × 月）交易量異常偏低",
        "\n".join([f"    {r['範圍']} {r['交易月']}：{r['筆數']:,.0f} 筆"
                   f"（局部 {r['局部比值']:.2f} / 同月跨年 {r['同月跨年比值']:.2f}）"
                   for _, r in flagged.head(30).iterrows()] +
                  ([f"    …另有 {len(flagged) - 30} 個"] if len(flagged) > 30 else []) +
                  ["", "只有單一縣市中標 → 多半是該縣市某幾季的檔案有問題；",
                   "六都整體也中標 → 看 Layer 2c 的貢獻矩陣，找出是哪個發布季缺了。",
                   "只有『同月跨年』中標而局部沒中 → 是橫跨多個月的長缺口，",
                   "    局部基準對這種形狀本來就會失明，別以為它沒中標就沒事。",
                   f"確認是真實市場低點而非缺資料，可調高 COVERAGE_MIN_RATIO"
                   f"（目前 {COVERAGE_MIN_RATIO}）或設 COVERAGE_STRICT = False。"]))
else:
    print(f"\n✅ Layer 3：兩個基準下都沒有月份低於 {COVERAGE_MIN_RATIO:.0%}")
''')

code(r'''
# ---- Layer 4：外部絕對錨（內政統計月報 4.5「辦理建物所有權登記」）----
# 前面每一層的基準幾乎都是從資料自己算出來的，所以對「基準本身被污染」沒有免疫力：
# 全窗口等比例少算會讓所有基準一起沉下去，相對值不變、全層綠燈。
# 這一層的分母來自體制外——官方按縣市 × 登記月公布的**買賣移轉棟數**。
# 我們每個批次檔的列數 ÷ 官方同期棟數，長期應該穩定在某個水準；掉下去就是我們掉了資料。
_ODS_T = "urn:oasis:names:tc:opendocument:xmlns:table:1.0"
_ODS_TX = "urn:oasis:names:tc:opendocument:xmlns:text:1.0"
_ODS_O = "urn:oasis:names:tc:opendocument:xmlns:office:1.0"
# 資料列的欄位配置：col0 = 期別、col1 = 區域別，之後每個登記原因佔「棟數 / 面積」兩欄。
ANCHOR_COL_MOVE_ALL = 4                     # 移轉登記合計（棟數）
ANCHOR_COL_SALE, ANCHOR_COL_SALE_AREA = 6, 7   # 買賣（棟數 / 面積）
ANCHOR_COL_REASONS = (6, 8, 10, 12, 14)     # 買賣、拍賣、繼承、贈與、其他（棟數）
_ANCHOR_MONTH = re.compile(r"^\s*(\d{1,2})\s*月")
_ANCHOR_ROC = re.compile(r"民國\s*(\d{2,3})\s*年")
_ANCHOR_ZH = re.compile(r"^[　\s]*([一-鿿　\s]+?)[　\s]*(?=[A-Za-z(]|$)")


def _ods_cells(row) -> list:
    """ODS 的一列 → 值的 list（展開 number-columns-repeated）。"""
    out = []
    for c in row.findall(f"{{{_ODS_T}}}table-cell"):
        rep = int(c.get(f"{{{_ODS_T}}}number-columns-repeated", 1))
        v = c.get(f"{{{_ODS_O}}}value")
        if v is None:
            v = "".join("".join(pp.itertext())
                        for pp in c.findall(f"{{{_ODS_TX}}}p")).strip()
        out.extend([v] * min(rep, 64))
    return out


def parse_moi_ods(blob: bytes) -> tuple:
    """內政統計月報 4.5 的 ODS → (DataFrame, 逐工作表帳目)。

    用 odfpy／pandas 的 odf engine 會多一個相依（Colab 沒有預裝），這裡用標準庫解
    content.xml。每一列都驗「移轉登記合計 == 買賣+拍賣+繼承+贈與+其他」。

    **但恆等式不是萬能的**，實測可被繞過而不 raise：
      · 每個區塊內「棟數 ↔ 面積」對調（面積的合計恆等式同樣成立）
      · 五個登記原因欄任意排列（恆等式對排列不變）
    所以另外做**表級**的合理性檢查（sanity_official），用真實資料量過的先驗擋這兩類。
    另外回傳逐工作表帳目：版面改版時會表現成「某幾張工作表產出 0 列」這個對得出來的
    數字，而不是只有「總列數變少」——後者從輸出上看不出來。
    """
    root = ET.fromstring(zipfile.ZipFile(io.BytesIO(blob)).read("content.xml"))
    recs, checked, sheets = [], 0, []
    for tab in root.findall(f".//{{{_ODS_T}}}table"):
        name = (tab.get(f"{{{_ODS_T}}}name") or "").strip()
        rows = [_ods_cells(r) for r in tab.findall(f"{{{_ODS_T}}}table-row")]
        roc = None
        for r in rows[:6]:
            m = _ANCHOR_ROC.search("".join(str(x) for x in r[:3]))
            if m:
                roc = int(m.group(1))
                break
        if roc is None:
            sheets.append({"工作表": name, "民國年": None, "產出列數": 0,
                           "原因": "找不到「中華民國NNN年」"})
            continue
        month, n_before = None, len(recs)
        for r in rows:
            c0 = str(r[0]) if r else ""
            mm = _ANCHOR_MONTH.match(c0)
            if mm and "年" not in c0:
                month = int(mm.group(1))            # 進入某個月的區塊
                continue
            # month 還是 None 表示還在「全年合計」區塊（排在所有月份之前），跳過不然會重複計算
            if month is None or len(r) <= ANCHOR_COL_REASONS[-1]:
                continue
            zh = _ANCHOR_ZH.match(str(r[1]) if len(r) > 1 else "")
            city = re.sub(r"[　\s]", "", zh.group(1)) if zh else ""
            if not city.endswith(("市", "縣")):     # 「總計」「臺灣省」「福建省」等小計不要
                continue
            vals = [pd.to_numeric(str(r[i]).replace(",", ""), errors="coerce")
                    for i in (ANCHOR_COL_MOVE_ALL,) + ANCHOR_COL_REASONS]
            area = pd.to_numeric(str(r[ANCHOR_COL_SALE_AREA]).replace(",", ""),
                                 errors="coerce")
            if any(pd.isna(v) for v in vals) or pd.isna(area):
                continue
            if abs(sum(vals[1:]) - vals[0]) > max(2.0, 0.005 * float(vals[0])):
                raise ValueError(
                    f"欄位配置驗證失敗（{roc} 年 {month} 月 {city}）："
                    f"移轉合計 {vals[0]:,.0f} ≠ 買賣+拍賣+繼承+贈與+其他 {sum(vals[1:]):,.0f}。"
                    "來源版面改了，須更新 ANCHOR_COL_MOVE_ALL / ANCHOR_COL_REASONS。")
            checked += 1
            recs.append({"登記年月": pd.Period(f"{roc + 1911}-{month:02d}", freq="M"),
                         "縣市": city, "買賣棟數": int(vals[1]),
                         "移轉合計": float(vals[0]),
                         "其他原因最大": float(max(vals[2:])),
                         "買賣面積": float(area)})
        sheets.append({"工作表": name, "民國年": roc, "產出列數": len(recs) - n_before,
                       "原因": "" if len(recs) > n_before else "沒有任何「N月」區塊"})
    if not checked:
        raise ValueError("沒有任何一列通過欄位配置的算術驗證，來源版面可能改了")
    return pd.DataFrame(recs), sheets


# 表級合理性先驗，全部用真實檔案（1,976 筆 移轉合計 ≥ 500 的縣市月）量過：
#   · 買賣 / 移轉合計 的中位數 = 0.664（p5 0.478、p95 0.803）→ 門檻 [0.40, 0.85]
#   · 買賣 是五個原因中最大值的比例 = 0.9965（7/1976 例外，都是離島小月）→ 門檻 0.95
#   · 買賣面積 / 買賣棟數 的中位數 = 115.6 m²（p5 76.6、p95 187.0）→ 門檻 [30, 400]
# 「棟數↔面積 對調」會讓第三項變成 1/115.6 ≈ 0.009；「原因欄排列」會讓第一項掉到
# 0.03–0.10。兩者都被擋住，而真實資料離每個門檻都有 3 倍以上的餘裕。
ANCHOR_SANE = {"share": (0.40, 0.85), "max_frac": 0.95, "area_per_unit": (30.0, 400.0)}


def sanity_official(d: pd.DataFrame) -> dict:
    """對整張官方表做合理性檢查。不合理就 raise（版面錯位但恆等式仍成立的情形）。"""
    big = d[d["移轉合計"] >= 500]
    got = {"n_rows": int(len(d)), "n_big": int(len(big))}
    if len(big) < 30:
        got["skipped"] = "移轉合計 ≥ 500 的列不足 30，跳過表級檢查"
        return got
    got["share_median"] = round(float((big["買賣棟數"] / big["移轉合計"]).median()), 3)
    got["sale_is_max_frac"] = round(float((big["買賣棟數"] >= big["其他原因最大"]).mean()), 4)
    _s = d[d["買賣棟數"] >= 50]
    got["area_per_unit_median"] = round(
        float((_s["買賣面積"] / _s["買賣棟數"]).median()), 1) if len(_s) else None
    lo, hi = ANCHOR_SANE["share"]
    alo, ahi = ANCHOR_SANE["area_per_unit"]
    bad = []
    if not (lo <= got["share_median"] <= hi):
        bad.append(f"買賣/移轉合計 的中位數 {got['share_median']}（應在 {lo}–{hi}）"
                   "——登記原因的欄位順序可能變了")
    if got["sale_is_max_frac"] < ANCHOR_SANE["max_frac"]:
        bad.append(f"買賣是最大原因的比例 {got['sale_is_max_frac']}"
                   f"（應 ≥ {ANCHOR_SANE['max_frac']}）——同上")
    if got["area_per_unit_median"] is not None and not (
            alo <= got["area_per_unit_median"] <= ahi):
        bad.append(f"買賣面積/棟數 的中位數 {got['area_per_unit_median']} m²"
                   f"（應在 {alo}–{ahi}）——棟數與面積欄可能對調了")
    if bad:
        raise ValueError("官方表的表級合理性檢查未通過：\n  " + "\n  ".join(bad))
    return got


def fetch_moi_anchor() -> tuple:
    """抓（並快取）ANCHOR_URLS 的所有檔案。回傳 (官方資料, 逐檔說明)。"""
    frames, notes = [], []
    for url in ANCHOR_URLS:
        cache = RAW / ("moi_" + url.rsplit("/", 1)[-1])
        cached = cache.exists()
        if cached:
            blob, src = cache.read_bytes(), f"{cache.name}（快取）"
        else:
            try:
                resp = requests.get(url, headers=UA, timeout=180)
                resp.raise_for_status()
                blob = resp.content
                src = f"{cache.name}（下載 {len(blob) / 1e6:.1f} MB）"
            except Exception as exc:                      # noqa: BLE001
                notes.append(f"{url.rsplit('/', 1)[-1]}：抓取失敗 {type(exc).__name__}: {exc}")
                continue
        try:
            d, sheets = parse_moi_ods(blob)
            sane = sanity_official(d)
        except Exception as exc:                          # noqa: BLE001
            # **解析成功才寫快取。** 端點出錯時會回一頁 HTML 而不是 ODS；先寫快取的話
            # 那頁 HTML 會被當成快取永久留下，之後每次執行都讀到它、再也不會重抓。
            # 已存在的壞檔改名成 .bad 而不是刪掉——若是使用者手動放進來的檔案，
            # 刪掉會讓「手動補檔」這條路永遠走不通，而且看不到證據。
            if cached:
                bad = cache.with_suffix(cache.suffix + ".bad")
                cache.replace(bad)
                notes.append(f"{src}：解析失敗 {type(exc).__name__}: {exc}"
                             f"（已改名為 {bad.name}，下次執行會重抓）")
            else:
                notes.append(f"{src}：解析失敗 {type(exc).__name__}: {exc}")
            continue
        if not cached:
            cache.write_bytes(blob)
        _empty = [x["工作表"] for x in sheets if x["產出列數"] == 0]
        notes.append(
            f"{src}：{len(d):,} 列（{d['登記年月'].min()} ~ {d['登記年月'].max()}）"
            f"｜工作表 {len(sheets)} 張、產出 0 列的 {len(_empty)} 張"
            + (f"：{_empty[:6]}" if _empty else "")
            + f"｜買賣佔比中位 {sane.get('share_median')}"
              f"、面積/棟數中位 {sane.get('area_per_unit_median')}")
        frames.append(d)
    if not frames:
        return pd.DataFrame(columns=["登記年月", "縣市", "買賣棟數"]), notes
    # ANCHOR_URLS 第一個是現行檔。官方會回頭修正舊數字，重疊年份以現行檔為準。
    return (pd.concat(frames, ignore_index=True)
              .drop_duplicates(subset=["登記年月", "縣市"], keep="first")), notes


def season_reg_months(season: str, lag: int = None) -> list:
    """發布季 '106S1' → 對應的三個登記月。落後 lag 季（預設 ANCHOR_LAG_Q，見 §0）。"""
    lag = ANCHOR_LAG_Q if lag is None else lag
    q = pd.Period(f"{int(season[:3]) + 1911}Q{int(season[-1])}", freq="Q") - lag
    return list(pd.period_range(q.start_time, q.end_time, freq="M"))


def official_by_season(off_idx: pd.Series, lag: int) -> pd.DataFrame:
    """(發布季 × 縣市) 的官方買賣棟數；官方缺月的格子是 NaN。"""
    rec = {}
    for ss in SEASONS:
        for c in CITY_ORDER:
            vals = [off_idx.get((c, m)) for m in season_reg_months(ss, lag)]
            rec[(ss, c)] = np.nan if any(v is None for v in vals) else float(sum(vals))
    return (pd.Series(rec, dtype="float64").unstack()
              .reindex(index=SEASONS, columns=CITY_ORDER))
''')

code(r'''
anchor = {"status": "disabled", "notes": [], "lag_q": ANCHOR_LAG_Q}
anchor_bad_level, anchor_flagged, anchor_steps, anchor_unchecked = [], [], [], []

if not ANCHOR_ENABLE:
    print("Layer 4 已關閉（ANCHOR_ENABLE = False）——沒有任何外部錨，"
          "「全窗口等比例少算」這一類缺陷這次沒有人擋。")
else:
    official, anchor_notes = fetch_moi_anchor()
    anchor["notes"] = anchor_notes
    print("\n".join("  " + n for n in anchor_notes) or "  （沒有任何來源可用）")

    if official.empty:
        anchor["status"] = "unavailable"
        msg = ("⚠️  Layer 4 沒有跑：抓不到（或解析不了）內政統計月報 4.5。\n"
               "    這一層是唯一的外部錨，沒有它就擋不住「全窗口等比例少算」。\n"
               "    可手動下載後放進 data/raw/，檔名照 ANCHOR_URLS 的最後一段加 moi_ 前綴。")
        if ANCHOR_REQUIRED:
            coverage_fail("❌ Layer 4：外部錨不可用，且 ANCHOR_REQUIRED = True", msg)
        else:
            print(msg)
    else:
        # ---- 我方分子：每個批次檔的可解析列數（全建物型態，未經任何型態篩選）----
        ours = (ls.groupby(["發布季", "縣市"], observed=True)["日期可解析"].sum()
                  .unstack("縣市").reindex(index=SEASONS, columns=CITY_ORDER))
        off_idx = official.set_index(["縣市", "登記年月"])["買賣棟數"]

        # ---- 落後季數的診斷 ----
        # ANCHOR_LAG_Q 設錯的話，絕對水準只偏 ≤0.8%（實測），三支檢查全綠——
        # 也就是「設定錯了但看起來完全正常」。**未滾動**的逐季比值變異係數對 lag
        # 極度敏感（實測正確 lag 0.0017 vs 錯的 0.15+，分離 90 倍），拿它當診斷。
        _cv = {}
        for _lag in ANCHOR_LAG_CANDIDATES:
            _o = official_by_season(off_idx, _lag)
            _r = (ours / _o).replace([np.inf, -np.inf], np.nan)
            _v = _r.to_numpy(dtype=float).ravel()
            _v = _v[np.isfinite(_v)]
            _cv[_lag] = float(np.std(_v) / np.mean(_v)) if len(_v) > 4 and np.mean(_v) else np.nan
        anchor["lag_cv"] = {k: (None if pd.isna(v) else round(v, 4)) for k, v in _cv.items()}
        _best = min((k for k in _cv if not pd.isna(_cv[k])), key=lambda k: _cv[k],
                    default=ANCHOR_LAG_Q)
        print(f"\n落後季數診斷（逐季比值的變異係數，越小越對）："
              + "、".join(f"lag={k} {('—' if pd.isna(v) else format(v, '.4f'))}"
                          for k, v in _cv.items()))
        # 只在差 2 季以上才警告：登記季橫跨兩個季，相鄰 lag 的變異係數本來就分不開
        # （實測 lag=0 0.175 vs lag=1 0.215，而真正錯 2 季以上會差一個量級）。
        if abs(_best - ANCHOR_LAG_Q) >= 2:
            print(f"⚠️  變異係數最小的是 lag={_best}，與設定 ANCHOR_LAG_Q={ANCHOR_LAG_Q}"
                  " 差了 2 季以上。設錯 lag 不會讓下面的檢查中標，只會吃掉階梯的"
                  "誤報餘裕——請確認。")
        elif _best != ANCHOR_LAG_Q:
            print(f"   （最小的是 lag={_best}，與設定的 {ANCHOR_LAG_Q} 相鄰——"
                  "登記季橫跨兩個季，相鄰 lag 分不開，屬正常。）")
        anchor["lag_best_by_cv"] = int(_best)

        offq = official_by_season(off_idx, ANCHOR_LAG_Q)
        both = ours.notna() & offq.notna()
        n_all = len(SEASONS) * len(CITY_ORDER)
        n_cmp = int(both.to_numpy().sum())
        anchor["n_cells_compared"] = n_cmp
        anchor["n_cells_total"] = n_all
        anchor["cover"] = round(n_cmp / n_all, 3)
        _miss = [(ss, c) for ss in SEASONS for c in CITY_ORDER
                 if not bool(both.loc[ss, c])]
        if _miss:
            print(f"\n⚠️  {len(_miss)} / {n_all} 個（發布季 × 縣市）沒有對應的官方資料，"
                  f"不參與比對（例：{_miss[:3]}）")

        seasons_cmp = [ss for ss in SEASONS if bool(both.loc[ss].any())]
        anchor["seasons_compared"] = seasons_cmp

        # **「整個縣市一格都對不到」與「官方最後幾個月還沒發布」要分開判。**
        # 前者只可能是解析壞了（縣市名用字改了、版面變了），不可能是官方延遲；
        # 而它偏偏會讓 cover 掉到 0.80 以下 → 走進 insufficient → 只印警告。
        # 也就是「壞 1 個縣市會中止、壞 6 個縣市只警告」——嚴重度反轉。
        _dead_cities = [c for c in CITY_ORDER if not bool(both[c].any())]
        anchor["cities_no_official"] = _dead_cities
        if _dead_cities:
            anchor["status"] = "failed"
            coverage_fail(
                f"❌ Layer 4：{len(_dead_cities)} 個縣市在官方資料裡一格都對不到："
                f"{'、'.join(_dead_cities)}",
                "\n".join([
                    "    官方只是「最後幾個月還沒發布」的話，掉的是**季**不是**縣市**。",
                    "    整個縣市對不到，只可能是縣市名對不上（臺/台 用字）或版面改了。",
                    f"    官方表實際出現的縣市：{sorted(official['縣市'].unique())[:12]}",
                    f"    我們要找的：{list(CITY_ORDER)}"]))

        # ---- 涵蓋率閘門 ----
        # 沒有這道閘門的話，只要 3 個發布季可比（18/216 = 8.3% 的格子）就會印 ✅，
        # 而 rel 與 step 兩支根本沒跑。實測「官方年份表頭改用語」只要打到 8/10 張
        # 工作表就會落到這個狀態，而 notes 的日期範圍看起來還是完整的。
        if len(seasons_cmp) < 3 or anchor["cover"] < ANCHOR_MIN_COVER:
            anchor["status"] = "insufficient"
            msg = (f"⚠️  Layer 4 沒有跑：可比對的格子只有 {n_cmp}/{n_all}"
                   f"（{anchor['cover']:.1%}，需要 ≥ {ANCHOR_MIN_COVER:.0%}）、"
                   f"可比對發布季 {len(seasons_cmp)} 個。\n"
                   "    先看上面逐檔說明裡「產出 0 列的工作表」——官方表的版面若改了，"
                   "症狀就是某幾張工作表整張沒解出來。")
            if ANCHOR_REQUIRED:
                coverage_fail("❌ Layer 4：可比對範圍不足，且 ANCHOR_REQUIRED = True", msg)
            else:
                print(msg)
        else:
            # ---- (a) 絕對水準：整個窗口的總和 ----
            # 首末季不比：最後一季可能還在陸續公布，第一季的登記期會跨出窗口
            inner = seasons_cmp[1:-1]
            lvl = (ours.where(both).loc[inner].sum()
                   / offq.where(both).loc[inner].sum())
            print(f"\n絕對水準：我方筆數 / 官方買賣移轉棟數"
                  f"（{inner[0]} ~ {inner[-1]}，共 {len(inner)} 個發布季）\n")
            print(pd.DataFrame({"我方筆數": ours.where(both).loc[inner].sum().astype("int64"),
                                "官方棟數": offq.where(both).loc[inner].sum().astype("int64"),
                                "比值": lvl.round(3)}).reindex(CITY_ORDER).to_string())

            def _fin(x) -> bool:
                """比值算不出來（NaN／inf）不能當通過。
                inf 還會讓 coverage 的 json.dumps(allow_nan=False) 整份炸掉。"""
                return x is not None and np.isfinite(x)

            anchor["cities_uncomparable"] = [c for c in CITY_ORDER
                                             if not _fin(lvl.get(c))]
            if anchor["cities_uncomparable"]:
                print(f"\n⚠️  {len(anchor['cities_uncomparable'])} 個縣市算不出比值"
                      f"（官方分母為 0 或無可比格子）："
                      f"{'、'.join(anchor['cities_uncomparable'])}")
            anchor_bad_level = [
                {"縣市": c, "比值": (round(float(lvl[c]), 3) if _fin(lvl.get(c)) else None)}
                for c in CITY_ORDER
                if not _fin(lvl.get(c))
                or not (ANCHOR_RATIO_MIN <= float(lvl[c]) <= ANCHOR_RATIO_MAX)]

            _fin_lvl = [float(lvl[c]) for c in CITY_ORDER if _fin(lvl.get(c))]
            print(f"\n門檻校準參考：絕對區間目前是 "
                  f"[{ANCHOR_RATIO_MIN}, {ANCHOR_RATIO_MAX}]，兩端都是沒有實測依據的保守值。")
            if _fin_lvl:
                print(f"  本次實測範圍 {min(_fin_lvl):.3f} ~ {max(_fin_lvl):.3f}；"
                      f"建議下限 ≈ {0.7 * min(_fin_lvl):.3f}（實測最小 × 0.7）、"
                      f"上限 ≈ {1.8 * max(_fin_lvl):.3f}（實測最大 × 1.8）。")
                print("  注意：即使下限照這樣校準，實測解析度也只有「等比例少算 30%」"
                      "——少算 25% 仍會通過全部各層。")

            # ---- (b) 形狀：四季滾動和的比值 ----
            # 發布季與登記季的邊界本來就對不齊（一季的批次檔裡混著跨月登記的案件），
            # 逐季比值會有季節性抖動；滾動四季幾乎把錯位抵銷掉（實測：溢出比例
            # 從 10% 變到 50% 完全無感，因為溢出不改變四季總和）。
            roll = (ours.where(both).rolling(ANCHOR_ROLL, min_periods=ANCHOR_ROLL).sum()
                    / offq.where(both).rolling(ANCHOR_ROLL, min_periods=ANCHOR_ROLL).sum()
                    ).replace([np.inf, -np.inf], np.nan)
            _rep = {}
            for c in CITY_ORDER:
                sr = roll[c].dropna()
                med = float(sr.median()) if len(sr) else np.nan
                row = {"點數": len(sr), "中位": med,
                       "最小/中位": (float(sr.min()) / med if len(sr) and med > 0 else np.nan),
                       # 上界也要印。判定式是雙邊的（vv > med / ANCHOR_REL_MIN），
                       # 但舊版只印下界的校準值——整份設計的原則是每個門檻都要看得到餘裕。
                       "最大/中位": (float(sr.max()) / med if len(sr) and med > 0 else np.nan),
                       "階梯下降": np.nan, "階梯上升": np.nan}
                v = sr.to_numpy(dtype=float)
                if len(sr) >= 8 and med > 0:
                    down, up = level_shift(v)
                    if down:
                        row["階梯下降"] = down[3]
                    if up:
                        row["階梯上升"] = up[3]
                _rep[c] = row
                # **記下每個縣市到底有沒有被檢查到。** rel 需要 ≥1 個滾動點、
                # step 需要 ≥8 個；不記的話某個縣市可以整段靜默不跑而印出 ✅
                # （實測：官方隔季缺月就會讓某縣市的滾動比值全是 NaN）。
                if not len(sr) or not (med > 0):
                    anchor_unchecked.append({"縣市": c, "缺": "滾動比值（rel）與階梯",
                                             "滾動點數": len(sr)})
                    continue
                if len(sr) < 8:
                    anchor_unchecked.append({"縣市": c, "缺": "階梯（需 ≥ 8 個滾動點）",
                                             "滾動點數": len(sr)})
                for ss, vv in sr.items():
                    if vv < ANCHOR_REL_MIN * med or vv > med / ANCHOR_REL_MIN:
                        anchor_flagged.append(
                            {"縣市": c, "發布季": ss, "滾動比值": round(float(vv), 4),
                             "該縣市中位": round(med, 4),
                             "相對": round(float(vv) / med, 3)})
                if len(sr) >= 8:
                    down, up = level_shift(v)
                    if down and down[3] < ANCHOR_STEP_DOWN:
                        anchor_steps.append({"縣市": c, "切點季": str(sr.index[down[0]]),
                                             "前段": round(down[1], 4),
                                             "後段": round(down[2], 4),
                                             "比值": round(down[3], 3),
                                             "方向": "該季起偏低"})
                    if up and up[3] > ANCHOR_STEP_UP:
                        anchor_steps.append({"縣市": c, "切點季": str(sr.index[up[0]]),
                                             "前段": round(up[1], 4),
                                             "後段": round(up[2], 4),
                                             "比值": round(up[3], 3),
                                             "方向": "該季之前偏低"})
            print(f"\n{ANCHOR_ROLL} 季滾動比值（門檻：相對下限 {ANCHOR_REL_MIN:.0%}、"
                  f"階梯 {ANCHOR_STEP_DOWN}/{ANCHOR_STEP_UP}）\n")
            print(pd.DataFrame(_rep).T.reindex(CITY_ORDER).round(3).to_string())
            print(f"門檻校準參考：「最小/中位」離 {ANCHOR_REL_MIN} 越近、"
                  f"「最大/中位」離 {1 / ANCHOR_REL_MIN:.3f} 越近、"
                  f"「階梯下降」離 {ANCHOR_STEP_DOWN} 越近、"
                  f"「階梯上升」離 {ANCHOR_STEP_UP} 越近，誤判風險越高。")
            print("  真實資料實測：階梯下降最小 0.882（臺南）→ 對 0.75 的餘裕是 **17.6%**，"
                  "不是合成資料量到的 23%（那個數字來自上升側）。最緊的是下降側。")
            anchor["roll_report"] = {
                c: {k: (None if pd.isna(v) else round(float(v), 4)) for k, v in r.items()}
                for c, r in _rep.items()}

            anchor["status"] = ("failed" if (anchor_bad_level or anchor_flagged
                                             or anchor_steps or anchor_unchecked)
                                else "passed")
            anchor["level_by_city"] = {
                c: (round(float(lvl[c]), 4) if _fin(lvl.get(c)) else None)
                for c in CITY_ORDER}
            anchor["window"] = [inner[0], inner[-1]]

            if anchor["status"] == "failed":
                coverage_fail(
                    f"❌ Layer 4：與官方買賣移轉棟數比對不過"
                    f"（絕對水準 {len(anchor_bad_level)} 個縣市、"
                    f"滾動比值旗標 {len(anchor_flagged)} 個、階梯 {len(anchor_steps)} 個、"
                    f"沒被檢查到的縣市 {len(anchor_unchecked)} 個）",
                    "\n".join(
                        [f"    絕對水準 {r['縣市']}："
                         + ("算不出來（官方分母為 0 或無可比格子）" if r['比值'] is None
                            else f"{r['比值']}（允許 {ANCHOR_RATIO_MIN}–{ANCHOR_RATIO_MAX}）")
                         for r in anchor_bad_level] +
                        [f"    沒被檢查到 {r['縣市']}：缺 {r['缺']}"
                         f"（滾動點數 {r['滾動點數']}）——官方那幾期缺月，"
                         "這個縣市這次等於沒有外部錨" for r in anchor_unchecked] +
                        [f"    滾動比值 {r['縣市']} {r['發布季']}：{r['滾動比值']}"
                         f"（該縣市中位 {r['該縣市中位']} 的 {r['相對']:.0%}）"
                         for r in anchor_flagged[:20]] +
                        ([f"    …另有 {len(anchor_flagged) - 20} 個旗標"]
                         if len(anchor_flagged) > 20 else []) +
                        [f"    階梯 {r['縣市']}：{r['切點季']} 為切點，"
                         f"{r['前段']} → {r['後段']}（{r['比值']:.0%}，{r['方向']}）"
                         for r in anchor_steps] +
                        ["",
                         "怎麼讀：",
                         "  比值**偏低** → 我們少算了。先看 Layer 1（缺季／缺縣市）與 2a′。",
                         "  比值**偏高** → 同一批資料被算了兩次，或欄位配置錯位"
                         "（恆等式擋不住『棟數↔面積對調』與『原因欄排列』，"
                         "但 sanity_official 會；若那裡過了就先懷疑重複載入）。",
                         "  **整體**都偏離而不是某幾季 → 先看上面的落後季數診斷，",
                         "  再確認官方那張表的口徑沒改（口徑變更算術驗不出來）。",
                         "  階梯的切點只是「落差最大的分割位置」，實測會偏移最多 5 季；",
                         "  而滾動四季會把單季異常抹成 4 個連續點，所以階梯在序列"
                         "兩端第 4 點附近最敏感、中段幾乎全盲——要定位是哪幾季，"
                         "看上面的滾動比值表與 Layer 2a′ 的逐檔數字。"]))
            else:
                print(f"\n✅ Layer 4：{n_cmp}/{n_all} 個（發布季 × 縣市）"
                      f"（{anchor['cover']:.1%}）與官方買賣移轉棟數比對通過")
''')

code(r'''
# ---- Layer 5：跨次執行的季筆數快照 ----
# Layer 4 問「我們跟官方比對不對」，這一層問「這次跟上次比有沒有掉」。
# 兩者互補：快照不需要任何外部來源，而且只要上一次是好的，連等比例少算都擋得住。
# 快照只在本次全部檢查通過時才更新（見下一個 cell），不然壞的一次會把好的基準蓋掉。
#
# 存兩份東西：
#   1. counts —— 每個 (發布季 × 縣市) 的可解析列數（全建物型態）
#   2. group_counts —— 每季每個建物型態組的命中數
# 第 2 份是必要的：counts 的分子在型態篩選之前，對「型態用語改版」整條路徑結構上失明
# （實測用語改版 30% 時 counts 逐位元相同）。group_counts 讓這條路徑也有跨次基準。
SNAP_PATH = PROC / "coverage_snapshot.json"
snap = {"status": "disabled", "path": SNAP_PATH.name}
snap_fail, snap_warn, snap_group_fail = [], [], []

snap_counts = {f"{r['發布季']}|{r['縣市']}": int(r["日期可解析"])
               for _, r in ls.groupby(["發布季", "縣市"], as_index=False, observed=True)
                             ["日期可解析"].sum().iterrows()}
snap_groups = {f"{ss}|{g}": int(v) for ss, row in grp_season.to_dict("index").items()
               for g, v in row.items() if v >= 0}

if not SNAPSHOT_ENABLE:
    print("Layer 5 已關閉（SNAPSHOT_ENABLE = False）")
else:
    prev = None
    if SNAP_PATH.exists():
        try:
            _p = json.loads(SNAP_PATH.read_text(encoding="utf-8"))
            # 沒有 counts（或是空的）就不能當基準：否則下面一個格子都不比，
            # 卻會印出 passed——比沒有快照更糟。
            prev = _p if (_p.get("schema") == SNAPSHOT_SCHEMA and _p.get("counts")) else None
            if prev is None:
                print(f"快照不可用（schema={_p.get('schema')}、要 {SNAPSHOT_SCHEMA}；"
                      f"counts {len(_p.get('counts') or {})} 個），視為第一次執行")
        except Exception as exc:                      # noqa: BLE001
            print(f"⚠️  快照讀取失敗（{type(exc).__name__}: {exc}），視為第一次執行")

    if prev is None:
        snap["status"] = "first_run"
        print(f"Layer 5：還沒有基準快照（{SNAP_PATH.name}）。"
              "本次通過後會寫入，下次執行才有得比。")
    else:
        old = prev.get("counts", {})
        old_groups = prev.get("group_counts", {}) or {}
        gone = sorted(set(old) - set(snap_counts))
        added = sorted(set(snap_counts) - set(old))
        n_eff = 0                     # **真正進入比值計算**的格數
        for k in sorted(old):
            v0 = int(old[k])
            v1 = snap_counts.get(k)
            if v1 is None:
                snap_fail.append({"鍵": k, "上次": v0, "這次": 0, "比值": 0.0})
            elif v0 > 0:
                n_eff += 1
                ratio = v1 / v0
                if ratio < SNAPSHOT_DROP_FAIL:
                    snap_fail.append({"鍵": k, "上次": v0, "這次": v1, "比值": round(ratio, 3)})
                elif ratio < SNAPSHOT_DROP_WARN:
                    snap_warn.append({"鍵": k, "上次": v0, "這次": v1, "比值": round(ratio, 3)})
        for k, v0 in sorted(old_groups.items()):
            v1 = snap_groups.get(k)
            if v0 > 0 and v1 is not None and v1 / v0 < SNAPSHOT_DROP_FAIL:
                snap_group_fail.append({"鍵": k, "上次": int(v0), "這次": int(v1),
                                        "比值": round(v1 / v0, 3)})
        cover = n_eff / max(len(snap_counts), 1)
        snap.update({"baseline_created": prev.get("created"),
                     "baseline_anchor_status": prev.get("external_anchor_status"),
                     "n_compared": len(old), "n_effective": n_eff,
                     "cover": round(cover, 3),
                     "n_missing": len(gone), "n_added": len(added),
                     "n_dropped": len(snap_fail), "n_shrunk": len(snap_warn),
                     "n_group_dropped": len(snap_group_fail)})
        print(f"Layer 5：基準建於 {prev.get('created')}"
              f"（當時 Layer 4 狀態：{prev.get('external_anchor_status')}）")
        print(f"  上次 {len(old)} 格、實際比到 {n_eff} 格（{cover:.1%}）、"
              f"新增 {len(added)} 格、上次有這次沒有 {len(gone)} 格")
        print(f"  型態組：上次 {len(old_groups)} 格、下降 {len(snap_group_fail)} 格")
        # 鍵格式變更（CITY_CODES 臺→台、SEASONS 改動）會表現成「全部掉光」，
        # 讀起來像資料全滅。先把方向講對，再列細節。
        if gone and added and abs(len(gone) - len(added)) <= max(2, 0.1 * len(gone)):
            print(f"  ℹ️  上次的鍵有 {len(gone)} 個完全找不到、同時新增 {len(added)} 個新鍵"
                  "——這是**鍵格式變了**（CITY_CODES 用字或 SEASONS 範圍改過），"
                  f"不是資料掉了。確認無誤就刪掉 {SNAP_PATH.name} 重建基準。")
        if snap_warn:
            print(f"  ℹ️  {len(snap_warn)} 個小幅減少（來源確實會撤銷或修正個別案件）：")
            print("\n".join(f"     {r['鍵']}：{r['上次']:,} → {r['這次']:,}"
                            f"（{r['比值']:.1%}）" for r in snap_warn[:10]))
        # 比對涵蓋率不足時不能算通過：counts 全是 0／只剩一格（例如先用窄 SEASONS
        # 試跑一次就寫了基準）都會讓「比對 N 個」看起來正常而實際一格都沒比。
        _thin = cover < SNAPSHOT_MIN_COVER
        snap["status"] = ("failed" if (snap_fail or snap_group_fail or _thin) else "passed")
        if _thin:
            print(f"  ⚠️  只比到 {n_eff}/{len(snap_counts)} 格（{cover:.1%}），"
                  f"低於 {SNAPSHOT_MIN_COVER:.0%}——這個基準覆蓋不了現在的範圍。")
        if snap_fail or snap_group_fail or _thin:
            coverage_fail(
                f"❌ Layer 5：比上次執行少了（格子 {len(snap_fail)} 個、"
                f"型態組 {len(snap_group_fail)} 個、比對涵蓋率 {cover:.1%}）",
                "\n".join([f"    {r['鍵']}：{r['上次']:,} → {r['這次']:,}（{r['比值']:.1%}）"
                           for r in snap_fail[:30]] +
                          ([f"    …另有 {len(snap_fail) - 30} 個"]
                           if len(snap_fail) > 30 else []) +
                          [f"    型態組 {r['鍵']}：{r['上次']:,} → {r['這次']:,}"
                           f"（{r['比值']:.1%}）" for r in snap_group_fail[:20]] +
                          ([f"    比對涵蓋率只有 {cover:.1%}（需 ≥ "
                            f"{SNAPSHOT_MIN_COVER:.0%}）——基準的範圍比現在窄，"
                            f"刪掉 {SNAP_PATH.name} 重建"] if _thin else []) +
                          ["",
                           "上次執行拿得到、這次拿不到。可能原因：",
                           "  · data/raw/lvr 下的 csv 被部分刪掉或截斷 → 刪掉該季的 .done 重下載",
                           "  · 來源欄名／用語改版 → 看 Layer 2a / 2a′ / 2a″ / 2a‴",
                           "  · 型態組下降而格子沒下降 → 就是用語改版（counts 的分子在型態"
                           "篩選之前，看不到那條路徑，group_counts 才看得到）",
                           "  · SEASONS 或 CITY_CODES 被改窄了（那是預期行為，"
                           f"刪掉 {SNAP_PATH.name} 重建基準）"]))
        else:
            print(f"  ✅ 沒有任何格子或型態組比上次少超過 {1 - SNAPSHOT_DROP_FAIL:.0%}")
''')

code(r'''
def _cov_num(v) -> Optional[float]:
    """NaN → None，其餘四捨五入到小數三位。"""
    return None if pd.isna(v) else round(float(v), 3)


def _jsonable(recs: list) -> list:
    """NaN 不是合法 JSON（jq、JSON.parse 都會炸），統一轉成 null。"""
    return [{k: (None if isinstance(v, float) and np.isnan(v) else v)
             for k, v in r.items()} for r in recs]


coverage = {
    "strict": COVERAGE_STRICT,
    "layer1": LAYER1,
    "min_ratio": COVERAGE_MIN_RATIO,
    "step_ratio": STEP_RATIO,
    "step_up_ratio": STEP_UP_RATIO,
    "date_min": DATE_MIN,
    "resi_floor": RESI_FLOOR,
    "main_floor": MAIN_FLOOR,
    "resi_local_window": RESI_WIN,
    "resi_local_min": RESI_LOCAL_MIN,
    "main_local_min": MAIN_LOCAL_MIN,
    "punch_window": [PUNCH_INNER, PUNCH_OUTER],
    "resi_punch_min": RESI_PUNCH_MIN,
    "main_punch_min": MAIN_PUNCH_MIN,
    "n_rel_checked": n_rel_checked,
    # 門檻校準用。**看 local_min 不要看 min/max**：真實資料上臺南的住宅類佔比
    # 八年間從 0.29 漲到 0.50，min/max 比值 0.52 是真實趨勢而非缺陷——這正是
    # 舊版用全期 p90 當基準會誤報 4 個檔案的原因。局部基準對趨勢免疫。
    # 某縣市 36 季全部 NaN（完全沒有資料）時這些值都是 None。
    "share_by_city": {
        c: {"住宅類_min": _cov_num(g["住宅類佔比"].min()),
            "住宅類_max": _cov_num(g["住宅類佔比"].max()),
            "住宅類_local_min": _cov_num(g["住宅類佔比_local"].min()),
            "住宅類_punch_min": _cov_num(g["住宅類佔比_punch"].min()),
            "主型態_min": _cov_num(g["主型態佔比"].min()),
            "主型態_max": _cov_num(g["主型態佔比"].max()),
            "主型態_local_min": _cov_num(g["主型態佔比_local"].min()),
            "主型態_punch_min": _cov_num(g["主型態佔比_punch"].min())}
        for c, g in ls.groupby("縣市")},
    "n_seasons": len(SEASONS),
    "n_files": len(files),
    "n_files_failed": len(load_failed),
    "failed_files": load_failed,
    "bad_retention": _jsonable(
        bad_ret[["檔名", "原始列數", "日期格式率", "日期解析率",
                 "住宅類佔比", "住宅類佔比_local", "住宅類佔比_punch",
                 "主型態佔比", "主型態佔比_local",
                 "主型態佔比_punch"]].round(3).to_dict("records")),
    "bad_season_city": [{"發布季": ss, "縣市": c, "筆數": n, "季中位": round(b, 1)}
                        for ss, c, n, b in bad_sc],
    "level_shifts": [{"範圍": c, "切點季": ss, "前段中位": round(b, 1),
                      "後段中位": round(a, 1), "方向": w}
                     for c, ss, b, a, w in steps],
    "months_zero": int((per_month["筆數"] == 0).sum()),
    "min_ratio_by_scope": {
        k: {"局部": None if pd.isna(v["局部比值"]) else round(float(v["局部比值"]), 3),
            "同月跨年": None if pd.isna(v["同月跨年比值"]) else round(float(v["同月跨年比值"]), 3)}
        for k, v in flags.groupby("範圍")[["局部比值", "同月跨年比值"]].min().iterrows()},
    "flagged": _jsonable(
        flagged[["範圍", "交易月", "筆數", "局部比值", "同月跨年比值"]].round(3).to_dict("records")),
    "building_groups": {g: list(ps) for g, ps in BUILDING_GROUPS.items()},
    "building_group_hits": hits,
    "building_prefix_hits": prefix_hits,
    # 逐季命中數（-1 = 該季缺檔，不是零命中）。Layer 5 的快照也存這一份，
    # 這樣「型態用語改版」那條路徑才有跨次執行的比對基準。
    "group_hits_by_season": {
        ss: {g: int(v) for g, v in row.items()}
        for ss, row in grp_season.to_dict("index").items()},
    "dead_groups": dead_groups,
    "prefix_zero_runs": prefix_runs,
    "prefix_thin": prefix_thin,
    "prefix_zero_run_min": PREFIX_ZERO_RUN,
    "prefix_min_median": PREFIX_MIN_MEDIAN,
    # Layer 2a‴
    "group_level_shifts": grp_steps,
    "group_thin": grp_thin,
    "group_unchecked": grp_unchecked,
    "group_shift_extremes": grp_extremes,
    "group_step_down": GROUP_STEP_DOWN,
    "group_step_up": GROUP_STEP_UP,
    "dead_prefixes_in_groups": dead_prefixes_in_groups,
    "steps_unchecked": steps_unchecked,
    # Layer 4：外部絕對錨。status 有 passed / failed / unavailable / insufficient / disabled
    # 五種——**只有 passed 代表這一層真的跑過**，其餘都是「沒有人擋」。
    "external_anchor": {**anchor,
                        "required": ANCHOR_REQUIRED,
                        "enabled": ANCHOR_ENABLE,
                        "bad_level": anchor_bad_level,
                        "flagged": anchor_flagged,
                        "level_shifts": anchor_steps,
                        "unchecked_cities": anchor_unchecked,
                        "thresholds": {"ratio_min": ANCHOR_RATIO_MIN,
                                       "ratio_max": ANCHOR_RATIO_MAX,
                                       "rel_min": ANCHOR_REL_MIN,
                                       "roll": ANCHOR_ROLL,
                                       "step_down": ANCHOR_STEP_DOWN,
                                       "step_up": ANCHOR_STEP_UP,
                                       "min_cover": ANCHOR_MIN_COVER}},
    # Layer 5：跨次執行快照
    "snapshot": {**snap, "dropped": snap_fail, "shrunk": snap_warn,
                 "group_dropped": snap_group_fail,
                 "thresholds": {"fail": SNAPSHOT_DROP_FAIL, "warn": SNAPSHOT_DROP_WARN,
                                "min_cover": SNAPSHOT_MIN_COVER}},
    "passed": bool(not LAYER1["missing_seasons"] and not LAYER1["partial_seasons"]
                   and not load_failed and len(bad_ret) == 0 and not dead_groups
                   and not prefix_runs and not grp_steps and not grp_unchecked
                   and not dead_city
                   and not bad_sc and not steps and not steps_unchecked
                   and flagged.empty
                   and anchor["status"] != "failed" and snap["status"] != "failed"
                   # ANCHOR_REQUIRED=True 的時候，unavailable / insufficient 也算不過。
                   # 少了這一條，「ANCHOR_REQUIRED=True + COVERAGE_STRICT=False」會印出
                   # 「❌ Layer 4：外部錨不可用」緊接著「passed = True」。
                   and not (ANCHOR_REQUIRED and ANCHOR_ENABLE
                            and anchor["status"] != "passed")),
    # passed=True 的意思是「這九層都沒中標」，不是「資料一定完整」。
    # 把限制一起寫進 JSON，免得下游只看到 passed=True 就以為萬無一失。
    "limitations": [
        "內部各層（1 ~ 3）除 2a′ 的日期解析率外，基準都由資料自身算出，"
        "對「基準被污染」結構性無免疫力；能補上這一塊的只有 Layer 4 的外部錨",
        "Layer 4 的絕對下限已用真實資料校準（0.61 ≈ 實測最小 0.876 × 0.7），"
        "但實測解析度只有「等比例少算 30%」——少算 25% 仍會通過全部各層。"
        "「每年重複出現的長缺口」同理：滾動四季比值會一起沉，只有絕對水準接得到",
        "2a′ 的相對基準有兩支：局部（窗口 7）對**連續 ≥4 季**的缺口恆等於 1.000"
        "（純算術：median 是第 4 小，窗口內有 ≥4 個缺口值時 median 就是缺口值）。"
        "挖空基準（挖掉自己 ±7 季、只用距離 8–16 季的鄰居）補掉那條盲帶——"
        "實測注入「臺中住宅大樓 -50%」時，連續 4／6／10 季的偵測值都是 0.499"
        "（**與缺口長度無關**），對門檻 0.55 都接得到。"
        "偵測深度的下限是「掉 ≥ 45%」；掉 30% 時挖空比值約 0.70，接不到。"
        "缺口連續超過 15 季（2×PUNCH_INNER+1）時挖空基準也會開始失效。"
        "**另一端也有殘留**：只有 1–3 季的缺口，挖空基準的分母裡缺口佔比太低、"
        "局部基準又被真實單季變異（主型態佔比局部最小 0.659）蓋住——"
        "實測「臺中住宅大樓連續 2 季 -50%」十層全綠。那個量約佔全國主型態 0.6%，"
        "刻意接受",
        "型態用語**部分**改版：2a″ 只認恰好 0，2a‴ 接得到 f ≥ 0.45（18/36 季的形狀）。"
        "但**全 36 季一起**部分改版三層都接不到（2a′ 的局部基準會一起沉、"
        "2a‴ 沒有階梯、Layer 4/5 的分子在型態篩選之前所以結構上失明）。"
        "要擋住需要「某型態應佔住宅子集多少」這種絕對先驗，目前沒有",
        "Layer 4 缺席時（external_anchor.status 不是 passed：抓不到官方資料、"
        "ANCHOR_ENABLE=False、可比對季數不足）上面那一整類完全沒有人擋。"
        "看 passed 之前先看 external_anchor.status",
        "Layer 5 只答得出「這次比上次少」。第一次執行沒有基準；而基準本身是不是好的，"
        "要靠 Layer 4 保證——快照會忠實地把一次錯誤的執行當成往後的基準（故只在 passed 時才寫）",
        "「來源本身變少」路徑的溫和持續少算，內部各層兩個方向都有盲帶且不對稱："
        "序列後段掉到 55–70% 實測靜默損失達 23%；序列前段掉到 45–55%（2b′ 上升門檻 2.5 之下）"
        "實測靜默損失達 31–38%。用語改版路徑有 2a′ 與 2a″ 兜著，"
        "純檔案截斷路徑靠 Layer 4 的滾動比值階梯（0.80）接手，Layer 4 缺席就沒人接",
        "已用真實資料校準：2a′ 的 0.60/0.45（實測局部最小 0.840/0.659）、"
        "ANCHOR_RATIO_MIN/MAX 的 0.61/2.50（實測 0.876~1.404）、"
        "2a‴ 的 0.45（實測下降最小 0.681，臺中公寓）、"
        "Layer 4 階梯 0.75/1.50（實測下降最小 0.882，餘裕 17.6%——最緊的是下降側，"
        "不是註解一度寫的上升側 23%）。仍未校準：2b 的 0.35、2b′ 的 0.55/2.5、Layer 3 的 0.40",
        "**2a‴ 的下降門檻會隨 SEASONS 延長而變緊**：公寓的全國逐季命中數已從峰值 8,642"
        "掉到 3,972，而 level_shift 的最後一個切點只用最後 4 季當後段中位。"
        "每次把窗口往後延，都要回頭看 2a‴ 印的極值表",
        "門檻餘裕隨資料量反向變化且偏差有方向：合成資料會低估誤報餘裕（保守）、"
        "**高估偵測邊界（危險）**。實測同一缺陷在 2% 量體下 2a′ f=0.65 中標、"
        "20% 量體下要 f≈0.99。合成資料量到的偵測邊界一律當樂觀值看",
        "官方那張表的**口徑**若改了（例如買賣移轉棟數的定義變更），"
        "notebook 只驗證欄位配置的算術恆等式，驗不出口徑變更——會表現成 Layer 4 全縣市同時偏移",
    ],
}
assert json.dumps(coverage, ensure_ascii=False, allow_nan=False)   # 必須是合法 JSON
print(f"\ncoverage.passed = {coverage['passed']}")
print(f"external_anchor.status = {coverage['external_anchor']['status']}"
      f"（不是 passed 就代表這一層沒擋）")
print(f"snapshot.status = {coverage['snapshot']['status']}")
''')

code(r'''
# ---- Layer 5 的基準只在本次全部通過時才更新 ----
# 順序很重要：要等 coverage["passed"] 算出來。壞的一次若把快照蓋掉，
# 之後每一次都會拿壞的當基準、永遠綠燈——那比沒有快照更糟。
if SNAPSHOT_ENABLE and coverage["passed"]:
    _st = coverage["external_anchor"]["status"]
    SNAP_PATH.write_text(json.dumps(
        {"schema": SNAPSHOT_SCHEMA,
         "created": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
         "seasons": SEASONS, "cities": list(CITY_ORDER),
         "note": "counts = 每個 (發布季 × 縣市) 批次檔中「交易年月日 可解析」的列數"
                 "（全建物型態）；group_counts = 每季每個建物型態組的命中數",
         # **把基準當時的 Layer 4 狀態一起存下來。** 不存的話，往後任何一次執行都
         # 無法知道自己的基準有沒有被外部驗證過——「跟上次一樣」在上次就錯的時候
         # 是綠燈。
         "external_anchor_status": _st,
         "external_anchor_level": coverage["external_anchor"].get("level_by_city"),
         "counts": snap_counts,
         "group_counts": snap_groups}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"✅ 已更新基準快照 {SNAP_PATH}"
          f"（{len(snap_counts)} 格 + {len(snap_groups)} 個型態組格）")
    if _st != "passed":
        print(f"⚠️  但這次的 Layer 4 狀態是 `{_st}`（不是 passed）——這份基準"
              "沒有被外部錨驗證過，往後的「跟上次一樣」只保證一致、不保證正確。")
elif SNAPSHOT_ENABLE:
    print(f"⚠️  涵蓋率未通過，**不**更新 {SNAP_PATH.name}——"
          "免得把這次的缺漏變成往後的基準。")
''')

code(r'''
# ---- 存檔給 NB2 重用，避免重複清整 ----
# KEEP 定義在 §0（USE_COLS 由它推導）。這裡只剩不變量斷言：USE_COLS 既然是從 KEEP
# 推導的，缺欄就代表推導或清整壞了，不是設定寫錯——所以直接擋下來，不要靜默少一欄
# 讓 NB2 去踩。
_lost = [c for c in KEEP if c not in lvr.columns]
if _lost:
    raise RuntimeError(
        f"KEEP 有 {len(_lost)} 個欄位不在清整結果中：{_lost}\n"
        "USE_COLS 是由 KEEP 推導的（見 §0 的 DERIVED_COLS），所以這不該發生。\n"
        "依可能性排序：\n"
        f"  1. 那幾欄被**誤放進 DERIVED_COLS** → 就不會進 USE_COLS、也就沒被讀。"
        f"（目前 DERIVED_COLS 含 {sorted(DERIVED_COLS & set(_lost)) or '無交集'}）\n"
        "  2. 來源改了欄名 → 加進 COL_ALIASES（load_one 會印出「期待卻不存在的欄名」）\n"
        "  3. 真的是清整過程產生的衍生欄 → 才是加進 DERIVED_COLS")
out = lvr[KEEP].copy()
for c in ("month", "quarter"):
    out[c] = out[c].astype(str)
out.to_parquet(PROC / "lvr_clean.parquet", index=False)
print(f"已寫出 {PROC / 'lvr_clean.parquet'}（{len(out):,} 筆，"
      f"{(PROC / 'lvr_clean.parquet').stat().st_size / 1e6:.1f} MB）")
''')

# ============================== §4 單元母體 ==============================
md(r"""
---
## 4. 空間單元母體

密度的分母必須是**完整的單元母體**（含全期零交易的單元），不能只算資料裡出現過的單元。
從內政部戶政司開放資料 `ODRP019`（村里戶數、單一年齡人口數）取得六都的鄉鎮市區與村里清單，
順便拿到各村里戶數——後面「戶數比例分配」情境會用到。
""")

code(r'''
def fetch_villages(year: int = 113) -> pd.DataFrame:
    """內政部戶政司 ODRP019：村里戶數。回傳六都的 縣市/鄉鎮市區/村里/戶數。"""
    cache = RAW / f"villages_{year}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    records, page = [], 1
    while True:
        url = (f"https://www.ris.gov.tw/rs-opendata/api/v1/datastore/"
               f"ODRP019/{year}?page={page}")
        payload = requests.get(url, headers=UA, timeout=180).json()
        records.extend(payload["responseData"])
        total = int(payload["totalPage"])
        print(f"  page {page}/{total}")
        if page >= total:
            break
        page += 1

    v = pd.DataFrame(records)
    v["區域別"] = v["區域別"].str.replace("台", "臺", regex=False)
    v["縣市"] = v["區域別"].str[:3]
    v["鄉鎮市區"] = v["區域別"].str[3:]
    v = v[v["縣市"].isin(CITY_ORDER)].copy()

    for c in ("共同生活戶_戶數", "單獨生活戶_戶數"):
        v[c] = pd.to_numeric(v[c], errors="coerce").fillna(0)
    v["戶數"] = v["共同生活戶_戶數"] + v["單獨生活戶_戶數"]

    v = v[["縣市", "鄉鎮市區", "村里名稱", "戶數"]].rename(columns={"村里名稱": "村里"})
    v["district"] = v["縣市"] + v["鄉鎮市區"]
    v.to_parquet(cache, index=False)
    return v


villages = fetch_villages(113)
districts = (villages.groupby(["縣市", "district"], as_index=False)
                     .agg(村里數=("村里", "size"), 戶數=("戶數", "sum")))

print(f"\n六都村里總數：{len(villages):,}")
print(f"六都鄉鎮市區總數：{len(districts):,}\n")
print(districts.groupby("縣市")
               .agg(鄉鎮市區數=("district", "size"), 村里數=("村里數", "sum"))
               .reindex(CITY_ORDER).to_string())
''')

code(r'''
# 檢查實價登錄的 district 是否都能對到母體（抓錯字／改制／外島）
lvr_dist = set(lvr["district"].unique())
pop_dist = set(districts["district"])
orphan = lvr_dist - pop_dist
if orphan:
    n_orphan = lvr["district"].isin(orphan).sum()
    print(f"⚠️  {len(orphan)} 個 district 不在母體中，涉及 {n_orphan:,} 筆"
          f"（{n_orphan / len(lvr):.2%}）：{sorted(orphan)[:10]}")
else:
    print("✅ 實價登錄的所有 district 都對得上母體")

empty = pop_dist - lvr_dist
print(f"母體中全期零交易的 district：{len(empty)} 個"
      + (f" → {sorted(empty)}" if 0 < len(empty) <= 20 else ""))
''')

# ============================== §5 密度框架 ==============================
md(r"""
---
## 5. 密度分析

### 5.0 統計口徑

對每一種「空間單元 × 時間單元」方案，我們回報：

- `median / p25 / p75`：**含零 cell** 的交易筆數分布
- `zero_share`：零交易 cell 佔比
- `n1_share`：恰好 1 筆的 cell 佔比（中位數 = 那一筆本身，無平滑效果，計畫 5.0 要求標記）
- `ge5_share`：**決策指標**，n ≥ 5 的 cell 佔比
- `ge3_share`：n ≥ 3 的 cell 佔比（計畫 5.2 步驟 9 的排除門檻）

#### 分母的雙口徑（重要）

上面每個指標都算**兩次**，分子相同、分母不同，`_active` 後綴是第二種：

| 口徑 | 分母 | 立場 |
|---|---|---|
| **全母體**（無後綴，**主口徑**） | 六都全部鄉鎮市區 × 期數 | 計畫 §5.0 的字面讀法 |
| **子母體**（`_active`） | 全期至少一筆交易的鄉鎮市區 × 期數 | 只算「可能進模型」的地區 |

六都有相當比例的區（山地、偏鄉）八年來一筆電梯大樓交易都沒有。這些區在 NB2
（§5.2 步驟 9 排除 n<3）本來就會被丟掉，**不管選哪個粒度都不會進模型**——
把它們算進分母，等於四個方案被同一個係數一起稀釋。

這件事**不改變四個方案的排名**，只改變它們有沒有越過 50% 那條線；
而 50% 是計畫裡拍出來的門檻，本身沒有統計學依據。因此本 notebook 的做法是：

> **主決策維持全母體**（不替使用者搬球門），但兩個數字都算、都寫進報告與 JSON。
> 當兩個口徑給出**相反的 go/no-go** 時，該方案判定為 `CONFLICT`，
> 決策階梯就地停住、不自動選邊，交由人工拍板。

因子母體較小，`*_active` 的達標佔比恆 ≥ 對應的全母體值，所以衝突只會是
「全母體 FAIL、子母體 PASS」這一種方向。

村里方案的子母體 = 有交易的區底下的村里（實價登錄無村里欄位，無法在村里層級判定），
四個方案套用同一組區過濾，口徑一致。
""")

code(r'''
SHARE_COLS = ("zero_share", "n1_share", "ge3_share", "ge5_share")
NUM_COLS = ("median", "p25", "p75", "mean")


def active_districts(df: pd.DataFrame, dist_names) -> set:
    """子母體：全期至少一筆交易的鄉鎮市區。村里方案沿用同一組區過濾
    （實價登錄無村里欄位，無法在村里層級判定有無交易），四個方案口徑一致。"""
    have = set(df["district"].unique())
    return {d for d in dist_names if d in have}


def density_stats(counts: np.ndarray, n_units: int, n_periods: int,
                  label: str = "", scope: str = "六都",
                  n_units_active: Optional[int] = None) -> dict:
    """counts：非零 cell 的交易筆數陣列。零 cell 由 單元數 × 期數 補齊。

    兩個分母都算（見 5.0 節）。分子相同，只有分母不同：
      - 全母體 `n_units`：計畫 §5.0 字面讀法，**主口徑**
      - 子母體 `n_units_active`：全期至少一筆交易的單元，欄位加 `_active` 後綴
    因子母體較小，`*_active` 的達標佔比恆 ≥ 對應的全母體值。
    """
    counts = np.asarray(counts, dtype=float)
    n_units_active = int(n_units if n_units_active is None else n_units_active)
    if n_units_active > n_units:
        raise ValueError(f"{label}: 子母體({n_units_active}) 大於全母體({n_units})")

    def block(units: int, sfx: str) -> dict:
        total = int(units) * int(n_periods)
        if total == 0:                      # 該範圍全期零交易（只可能發生在分市報告）
            return {f"單元數{sfx}": 0, f"cell 數{sfx}": 0,
                    **{f"{k}{sfx}": np.nan for k in NUM_COLS + SHARE_COLS}}
        if counts.size > total:
            raise ValueError(f"{label}{sfx}: 非零 cell({counts.size}) 超過總 cell({total})")
        n_zero = total - counts.size
        full = np.concatenate([counts, np.zeros(n_zero)])
        return {
            f"單元數{sfx}": int(units),
            f"cell 數{sfx}": total,
            f"median{sfx}": float(np.median(full)),
            f"p25{sfx}": float(np.percentile(full, 25)),
            f"p75{sfx}": float(np.percentile(full, 75)),
            f"mean{sfx}": float(full.mean()),
            f"zero_share{sfx}": n_zero / total,
            f"n1_share{sfx}": float((counts == 1).sum()) / total,
            f"ge3_share{sfx}": float((counts >= 3).sum()) / total,
            f"ge5_share{sfx}": float((counts >= MIN_N).sum()) / total,
        }

    return {"方案": label, "範圍": scope, "期數": int(n_periods),
            "交易筆數": int(counts.sum()),
            **block(n_units, ""), **block(n_units_active, "_active")}


def show(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    fmt = df.copy()
    for c in df.columns:
        if c.startswith(SHARE_COLS):
            fmt[c] = df[c].map(lambda x: "—" if pd.isna(x) else f"{x * 100:.1f}%")
        elif c.startswith(NUM_COLS):
            fmt[c] = df[c].map(lambda x: "—" if pd.isna(x) else f"{x:.2f}")
        elif c.startswith("cell 數") or c == "交易筆數":
            fmt[c] = df[c].map("{:,}".format)
    return fmt
''')

md(r"""
### 5.1 四種方案的密度估計

`鄉鎮市區` 為精確計算；`村里` 三種為上限／情境估計（見開頭說明）。
""")

code(r'''
PERIOD_COL = {"月": "month", "季": "quarter"}


def _cell_counts(df: pd.DataFrame, freq: str, extra: Optional[list] = None) -> pd.DataFrame:
    keys = ["district", PERIOD_COL[freq]] + (extra or [])
    return df.groupby(keys, observed=True).size().rename("n").reset_index()


# ---------- 鄉鎮市區：精確 ----------
def district_density(df, dist_pop, freq, scope="六都") -> dict:
    g = _cell_counts(df, freq)
    act = active_districts(df, dist_pop["district"])
    return density_stats(g["n"].values, len(dist_pop), N_PERIODS[freq],
                         f"鄉鎮市區 × {freq}", scope, n_units_active=len(act))


# ---------- 村里 A：理論上限 ----------
def village_ceiling(df, dist_pop, freq, scope="六都") -> dict:
    """任意分配方式下，n≥k 的 cell 數不可能超過 Σ min(V, ⌊T/k⌋)。這是嚴格上界。"""
    V = dist_pop.set_index("district")["村里數"]
    g = _cell_counts(df, freq)
    g["V"] = g["district"].map(V)
    g = g.dropna(subset=["V"])

    act = active_districts(df, dist_pop["district"])
    n_periods = N_PERIODS[freq]
    n_units = int(V.sum())
    n_units_active = int(V[V.index.isin(act)].sum())  # 有交易的區底下的村里數

    def hit(k: int) -> float:
        return float(np.minimum(g["V"], g["n"] // k).sum())

    def block(units: int, sfx: str) -> dict:
        total = int(units) * n_periods
        if total == 0:                      # 該範圍全期零交易（只可能發生在分市報告）
            return {f"單元數{sfx}": 0, f"cell 數{sfx}": 0,
                    **{f"{k}{sfx}": np.nan for k in NUM_COLS + SHARE_COLS}}
        # 中位數上限：最大的 m 使得「n≥m 的 cell」還能佔到一半以上
        med_ceiling = 0
        for m in range(1, 201):
            if hit(m) / total >= 0.5:
                med_ceiling = m
            else:
                break
        return {
            f"單元數{sfx}": int(units), f"cell 數{sfx}": total,
            f"median{sfx}": float(med_ceiling),
            f"p25{sfx}": np.nan, f"p75{sfx}": np.nan, f"mean{sfx}": np.nan,
            f"zero_share{sfx}": 1.0 - hit(1) / total,
            f"n1_share{sfx}": np.nan,
            f"ge3_share{sfx}": hit(3) / total,
            f"ge5_share{sfx}": hit(MIN_N) / total,
        }

    return {"方案": f"村里 × {freq}（A 理論上限）", "範圍": scope,
            "期數": n_periods, "交易筆數": int(g["n"].sum()),
            **block(n_units, ""), **block(n_units_active, "_active")}


# ---------- 村里 B：戶數比例分配（無聚集虛無情境） ----------
def village_multinomial(df, vill_pop, freq, scope="六都", seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    weights = {d: g["戶數"].to_numpy(dtype=float)
               for d, g in vill_pop.groupby("district", observed=True)}
    chunks = []
    for (dist, _), n in _cell_counts(df, freq).set_index(["district", PERIOD_COL[freq]])["n"].items():
        w = weights.get(dist)
        if w is None or len(w) == 0:
            continue
        p = w / w.sum() if w.sum() > 0 else np.full(len(w), 1.0 / len(w))
        draw = rng.multinomial(int(n), p)
        chunks.append(draw[draw > 0])
    counts = np.concatenate(chunks) if chunks else np.array([])
    act = active_districts(df, vill_pop["district"])
    return density_stats(counts, len(vill_pop), N_PERIODS[freq],
                         f"村里 × {freq}（B 戶數分配）", scope,
                         n_units_active=int(vill_pop["district"].isin(act).sum()))


# ---------- 村里 C：街道聚集情境 ----------
def village_street(df, dist_pop, freq, scope="六都") -> dict:
    """用門牌路段當代理單元：每個(區,期)取交易量前 V 大的路段填滿 V 個村里槽位。
    保留真實的空間聚集特性，是帶聚集的樂觀估計（恆 ≤ A 的上限）。"""
    V = dist_pop.set_index("district")["村里數"]
    g = _cell_counts(df, freq, extra=["street_key"])
    g["V"] = g["district"].map(V)
    g = g.dropna(subset=["V"]).sort_values("n", ascending=False)
    g["rank"] = g.groupby(["district", PERIOD_COL[freq]], observed=True).cumcount()
    keep = g[g["rank"] < g["V"]]
    act = active_districts(df, dist_pop["district"])
    return density_stats(keep["n"].values, int(V.sum()), N_PERIODS[freq],
                         f"村里 × {freq}（C 街道聚集）", scope,
                         n_units_active=int(V[V.index.isin(act)].sum()))


def all_schemes(df, dist_pop, vill_pop, scope="六都") -> list[dict]:
    rows = []
    for freq in ("月", "季"):
        rows.append(village_ceiling(df, dist_pop, freq, scope))
        rows.append(village_street(df, dist_pop, freq, scope))
        rows.append(village_multinomial(df, vill_pop, freq, scope))
        rows.append(district_density(df, dist_pop, freq, scope))
    return rows
''')

code(r'''
# 只保留對得上母體的交易，確保分母一致
lvr_m = lvr[lvr["district"].isin(set(districts["district"]))].copy()
print(f"納入密度分析：{len(lvr_m):,} 筆（清整後 {len(lvr):,} 筆）\n")

# ---- 兩個分母的規模（決策衝突的來源）----
ACTIVE_DIST = active_districts(lvr_m, districts["district"])
ACTIVE_VILL = int(districts[districts["district"].isin(ACTIVE_DIST)]["村里數"].sum())
print(f"全母體：{len(districts):,} 區 / {len(villages):,} 村里")
print(f"子母體：{len(ACTIVE_DIST):,} 區 / {ACTIVE_VILL:,} 村里"
      f"（全期至少一筆電梯大樓交易；佔全母體 {len(ACTIVE_DIST) / len(districts):.1%} 的區）")
print(f"→ 子母體口徑的達標佔比會是全母體的 {len(districts) / max(len(ACTIVE_DIST), 1):.2f}× 左右\n")
print("全期零交易的區（依縣市）：")
print((districts.assign(有交易=districts["district"].isin(ACTIVE_DIST))
                .groupby("縣市")["有交易"]
                .agg(區數="size", 有交易="sum")
                .assign(零交易=lambda d: d["區數"] - d["有交易"])
                .reindex(CITY_ORDER).to_string()))

overall = all_schemes(lvr_m, districts, villages, "六都")
show(overall).to_string(index=False)
''')

code(r'''
display_cols = ["方案", "單元數", "cell 數", "median", "p25", "p75",
                "zero_share", "n1_share", "ge3_share", "ge5_share"]
print("【主口徑：全母體】計畫 §5.0 字面讀法\n")
print(show(overall)[display_cols].to_string(index=False))
''')

code(r'''
# ---- 雙口徑對照：決策指標 ge5_share ----
O_all = pd.DataFrame(overall).set_index("方案")
dual = pd.DataFrame({
    "全母體 n≥5": O_all["ge5_share"],
    "子母體 n≥5": O_all["ge5_share_active"],
    "全母體 單元數": O_all["單元數"],
    "子母體 單元數": O_all["單元數_active"],
})
dual["口徑衝突"] = np.where(
    (O_all["ge5_share"] <= PASS_SHARE) & (O_all["ge5_share_active"] > PASS_SHARE),
    "⚠️ 翻轉", "")
fmt = dual.copy()
for c in ("全母體 n≥5", "子母體 n≥5"):
    fmt[c] = dual[c].map(lambda x: "—" if pd.isna(x) else f"{x:.1%}")
for c in ("全母體 單元數", "子母體 單元數"):
    fmt[c] = dual[c].map("{:,}".format)
print(f"兩個分母的決策指標對照（門檻 > {PASS_SHARE:.0%}）\n")
print(fmt.to_string())

n_flip = int((dual["口徑衝突"] != "").sum())
print(f"\n口徑會翻轉結論的方案數：{n_flip} / {len(dual)}")
if n_flip:
    print("⚠️  下方決策階梯會在第一個翻轉的方案就地停住，不自動選邊。")
''')

code(r'''
# ---- 代理單元的粒度檢查：路段數 vs 村里數 ----
s_per_dist = lvr_m.groupby("district")["street_key"].nunique().rename("活躍路段數")
gran = (districts.set_index("district")[["縣市", "村里數"]]
        .join(s_per_dist).fillna({"活躍路段數": 0}))
print("代理單元粒度（比值 >1 表示路段比村里細，C 情境會低估村里密度；<1 則高估）\n")
gran_city = (gran.groupby("縣市")[["村里數", "活躍路段數"]].sum().reindex(CITY_ORDER))
gran_city["路段/村里"] = (gran_city["活躍路段數"] / gran_city["村里數"]).round(2)
# 全期零交易的區（多為山地／偏鄉）會把比值拉到 0，另外標示出來
gran_city["有交易的區"] = (gran[gran["活躍路段數"] > 0].groupby("縣市").size()
                            .reindex(CITY_ORDER).fillna(0).astype(int))
gran_city["區數"] = gran.groupby("縣市").size().reindex(CITY_ORDER)
print(gran_city[["區數", "有交易的區", "村里數", "活躍路段數", "路段/村里"]].to_string())
print(f"\n六都整體：村里 {gran['村里數'].sum():,} 個，活躍路段 {int(gran['活躍路段數'].sum()):,} 個，"
      f"比值 {gran['活躍路段數'].sum() / gran['村里數'].sum():.2f}")
print(f"門牌未能解析出路名的比例：{(lvr_m['street'] == '(未解析)').mean():.2%}")
''')

md(r"""
### 5.2 六都分別報告

台北市的村里交易密度遠高於台南——整體數字會掩蓋這個差異，必須分開看。
若只有部分縣市撐得起某個粒度，可以考慮「分市採用不同粒度」，但那會讓 NB3 的跨市比較失去共同基準，
本計畫不採用。
""")

code(r'''
by_city_rows = []
for city in CITY_ORDER:
    d_sub = lvr_m[lvr_m["縣市"] == city]
    dist_sub = districts[districts["縣市"] == city]
    vill_sub = villages[villages["縣市"] == city]
    by_city_rows += all_schemes(d_sub, dist_sub, vill_sub, city)

by_city = pd.DataFrame(by_city_rows)

def city_pivot(col: str) -> pd.DataFrame:
    p = (by_city.pivot(index="方案", columns="範圍", values=col)
                .reindex(columns=CITY_ORDER) * 100).round(1)
    p["六都整體"] = (O_all[col] * 100).round(1)
    return p


pivot = city_pivot("ge5_share")
pivot_active = city_pivot("ge5_share_active")

for title, p in (("全母體（主口徑）", pivot), ("子母體 _active", pivot_active)):
    print(f"【{title}】各方案 n≥{MIN_N} 的 cell 佔比（%），決策門檻 > {PASS_SHARE:.0%}\n")
    print(p.to_string())
    print()

flip = (pivot <= PASS_SHARE * 100) & (pivot_active > PASS_SHARE * 100)
if flip.values.any():
    print(f"⚠️  口徑翻轉的（方案 × 範圍）格數：{int(flip.values.sum())}")
    print(flip.replace({True: "⚠️", False: ""}).to_string())
else:
    print("✅ 兩個口徑在所有方案 × 範圍上結論一致")
''')

code(r'''
print(f"各方案的 cell 交易筆數中位數（含零 cell）\n")
med = (by_city.pivot(index="方案", columns="範圍", values="median")
              .reindex(columns=CITY_ORDER))
med["六都整體"] = pd.DataFrame(overall).set_index("方案")["median"]
print(med.round(2).to_string())
''')

# ============================== §6 決策 ==============================
md(r"""
---
## 6. 粒度決策

依計畫 5.0 的決策規則逐條檢查。村里方案用 **A 理論上限**判定：
上限本身就 ≤ 50% 時，無論交易在村里間怎麼分布都不可能達標，可以在不做 geocoding 的情況下**直接排除**。
若上限過關但 C 街道聚集情境沒過，代表結論取決於實際空間分布，標記為「需 geocoding 才能定案」。

**每個方案在兩個分母口徑下各判一次**（見 5.0）：

| 判定 | 意義 |
|---|---|
| `PASS` / `PASS?` | 兩個口徑都過（`?` = 村里方案，須 geocoding 實測確認） |
| `FAIL` | 兩個口徑都不過 |
| `UNDETERMINED` | 兩口徑一致過了上限，但街道情境沒過，須 geocoding |
| `CONFLICT` | ⚠️ **兩個口徑結論相反**——決策階梯就地停住，不自動選邊 |

村里方案的衝突判定**只看 A 理論上限**（C 街道聚集的兩口徑差異記在細節裡不觸發停機），
理由與「決策以 A 為準」一致：A 是唯一不會因估計方法失準而翻案的數字。
""")

code(r'''
O = pd.DataFrame(overall).set_index("方案")


def pick(full, active):
    """取主口徑的值（DENOMINATOR_PRIMARY 決定）。衝突判定本身與主口徑無關。"""
    return full if DENOMINATOR_PRIMARY == "full" else active


def _fp(x) -> str:
    """給人看的百分比。NaN 代表「該範圍沒有活躍單元、無從計算」，不是 0。"""
    return "無法計算" if pd.isna(x) else f"{x:.1%}"


def tri(full: float, active: float) -> str:
    """兩個分母各判一次。因子母體恆 ≥ 全母體，衝突只有「全母體不過、子母體過」一種方向。"""
    # NaN（該範圍無活躍單元）比較永遠是 False，等於被讀成「不過」。
    # 在「完全沒有資料」的情境下判 FAIL 是合理結論，但要知道那不是「無法判定」。
    ok_f, ok_a = full > PASS_SHARE, active > PASS_SHARE
    if ok_f and ok_a:
        return "PASS_BOTH"
    return "FAIL_BOTH" if not ok_a else "CONFLICT"


def verdict_village(freq: str) -> dict:
    A, C, B = (f"村里 × {freq}（A 理論上限）", f"村里 × {freq}（C 街道聚集）",
               f"村里 × {freq}（B 戶數分配）")
    ceil_f, ceil_a = O.loc[A, "ge5_share"], O.loc[A, "ge5_share_active"]
    street_f, street_a = O.loc[C, "ge5_share"], O.loc[C, "ge5_share_active"]
    hh_f, hh_a = O.loc[B, "ge5_share"], O.loc[B, "ge5_share_active"]

    state = tri(ceil_f, ceil_a)          # 村里方案只認 A 理論上限，衝突判定亦然
    if state == "FAIL_BOTH":
        status = "FAIL"
        note = (f"理論上限 全母體 {_fp(ceil_f)} / 子母體 {_fp(ceil_a)} 皆 ≤ {PASS_SHARE:.0%}，"
                "兩個口徑下任何空間分布都不可能達標")
    elif state == "CONFLICT":
        status = "CONFLICT"
        note = (f"⚠️ 口徑翻轉：理論上限 全母體 {_fp(ceil_f)} ≤ {PASS_SHARE:.0%} < 子母體 {_fp(ceil_a)}。"
                "全母體口徑下此方案已可直接排除，子母體口徑下它仍在候選中（待 geocoding），需人工拍板")
    elif pick(street_f, street_a) > PASS_SHARE:
        status = "PASS?"
        note = (f"上限 {_fp(pick(ceil_f, ceil_a))}、街道情境 {_fp(pick(street_f, street_a))} "
                f"皆過關（主口徑＝{DENOMINATOR_PRIMARY}，兩口徑同結論），但須 geocoding 實測確認")
    else:
        status = "UNDETERMINED"
        note = (f"上限 {_fp(pick(ceil_f, ceil_a))} 過關但街道情境僅 "
                f"{_fp(pick(street_f, street_a))}，結論取決於實際村里分布，須 geocoding")
    return {"方案": f"村里 × {freq}", "ge5_share": ceil_f, "ge5_share_active": ceil_a,
            "主口徑值": pick(ceil_f, ceil_a), "判定": status, "說明": note, "精確": False,
            "細節": {"上限A": ceil_f, "上限A_active": ceil_a,
                     "街道C": street_f, "街道C_active": street_a,
                     "戶數B": hh_f, "戶數B_active": hh_a}}


def verdict_district(freq: str) -> dict:
    name = f"鄉鎮市區 × {freq}"
    share_f, share_a = O.loc[name, "ge5_share"], O.loc[name, "ge5_share_active"]
    state = tri(share_f, share_a)
    if state == "CONFLICT":
        status = "CONFLICT"
        note = (f"⚠️ 口徑翻轉：全母體 {_fp(share_f)} ≤ {PASS_SHARE:.0%} < 子母體 {_fp(share_a)}。"
                "分母選哪個會給出相反的 go/no-go，需人工拍板")
    else:
        status = "PASS" if state == "PASS_BOTH" else "FAIL"
        op = ">" if state == "PASS_BOTH" else "≤"
        note = (f"精確值 全母體 {_fp(share_f)} / 子母體 {_fp(share_a)}，"
                f"兩口徑皆 {op} {PASS_SHARE:.0%}")
    return {"方案": name, "ge5_share": share_f, "ge5_share_active": share_a,
            "主口徑值": pick(share_f, share_a), "判定": status, "說明": note, "精確": True, "細節": {}}


ladder = [verdict_village("月"), verdict_village("季"),
          verdict_district("月"), verdict_district("季")]

MARK = {"PASS": "✅", "PASS?": "🟡", "UNDETERMINED": "🟡", "CONFLICT": "⚠️", "FAIL": "❌"}

chosen = halted = None
for v in ladder:
    print(f"{MARK[v['判定']]} {v['方案']:<12} "
          f"全母體={_fp(v['ge5_share']):>8} 子母體={_fp(v['ge5_share_active']):>8}  "
          f"{v['判定']:<13} {v['說明']}")
    if chosen or halted:
        continue                      # 階梯已定案；後面的方案只印數字、不採用
    if v["判定"] == "CONFLICT":
        halted = v
    elif v["判定"] in ("PASS", "PASS?"):
        chosen = v

print()
if chosen:
    print(f"👉 建議粒度：**{chosen['方案']}**"
          + ("（精確判定）" if chosen["精確"] else "（需 geocoding 確認）"))
elif halted:
    bar = "=" * 78
    print(bar)
    print(f"⚠️⚠️  決策階梯停在「{halted['方案']}」——兩個分母口徑給出相反的 go/no-go。")
    print(f"      全母體（{len(districts)} 區，計畫 §5.0 字面讀法）：{halted['ge5_share']:.1%} → 不過")
    print(f"      子母體（{len(ACTIVE_DIST)} 區，全期有交易的區）  ：{halted['ge5_share_active']:.1%} → 過")
    print()
    print("      差異來自八年零交易的區（山地、偏鄉）。它們不管選哪個粒度都不會進 NB2 的模型，")
    print("      所以算不算進分母是「口徑選擇」而非統計問題——而 50% 這條線本身也沒有統計學依據。")
    print()
    print("      本 notebook 不自動選邊：decision.scheme = null、denominator.conflict = true。")
    print("      拍板後的做法：")
    print("        · 維持全母體 → 視此方案為 FAIL，往下看更粗的粒度或放寬建物型態（下一節）")
    print("        · 改採子母體 → 把 DENOMINATOR_PRIMARY 改成 \"active\" 重跑本 notebook")
    print(bar)
else:
    print("👉 嚴格建物型態（僅電梯大樓）下四種方案全部不過 → 進入決策樹第 4 分支：放寬建物型態重跑")
''')

code(r'''
# ---- 決策樹第 4 分支：放寬建物型態（住宅大樓 + 華廈 + 公寓）----
# BUILDING_MAIN ⊂ BUILDING_RELAXED，所以「放寬版 = 嚴格版 ∪ 只有放寬版才收的型態」。
# 不重跑整份 clean_lvr：那會讓 street_key 的逐列迴圈把嚴格版那批列再算一次，
# 而它是本 notebook 最慢也最吃記憶體的一段。
BUILDING_EXTRA = tuple(t for t in BUILDING_RELAXED if not t.startswith(BUILDING_MAIN))
# 拆解只有在「沒有任何 (縣市, 編號) 同時出現在兩個型態子集」時才與單次清整等價。
# 有交集時兩者會不一樣，而且方向不只一種（實測差 0.25% 的筆數、0.22% 的鍵內容）：
#   · clean_lvr 的去重在第 4 步，備註／價格／日期篩選在它之後——單次版可能因為
#     代表列被後面的步驟刪掉而整個編號消失，拆解版則另一個子集裡的那列還活著
#   · concat 把 lvr 放前面 + keep="first" → 拆解版永遠偏好主型態，單次版偏好 raw 順序
# 所以先問這個交集在不在，有就老實跑單次（慢，但正確）。
# 一個 (縣市, 編號) 只可能同時落在兩個型態子集 ⟺ 它有 ≥2 列且型態不同
# （前綴互斥、一列只有一個型態）。所以交集必為「重複的 (縣市, 編號)」的子集——
# 先用 duplicated 把候選縮到那幾列，不要對全部 143 萬列建兩個 set
# （那要約 150 MB 與數秒；真實資料上重複率是 0.00%，等於白花）。
_dupmask = raw.duplicated(subset=["縣市", "編號"], keep=False)
_cand = raw.loc[_dupmask, ["縣市", "編號", "建物型態"]]
_bt_raw = _cand["建物型態"].fillna("")
_ids_main = set(map(tuple, _cand.loc[_bt_raw.str.startswith(BUILDING_MAIN),
                                     ["縣市", "編號"]].to_numpy()))
_ids_extra = (set(map(tuple, _cand.loc[_bt_raw.str.startswith(BUILDING_EXTRA),
                                       ["縣市", "編號"]].to_numpy()))
              if BUILDING_EXTRA else set())
_overlap = _ids_main & _ids_extra
print(f"（重複的 (縣市, 編號) 候選 {int(_dupmask.sum()):,} 列 → "
      f"跨型態交集 {len(_overlap):,} 個）")
if not BUILDING_EXTRA:
    lvr_relaxed = lvr.copy()          # BUILDING_RELAXED 等於 BUILDING_MAIN
elif _overlap:
    print(f"ℹ️  有 {len(_overlap):,} 個 (縣市, 編號) 同時出現在主型態與放寬型態，"
          "拆解會與單次清整不等價 → 改跑單次完整清整（較慢）")
    lvr_relaxed, _ = clean_lvr(raw, BUILDING_RELAXED, verbose=False)
else:
    # 兩個子集的 (縣市, 編號) 不相交 → 拆解與單次清整逐列等價，
    # 而且省掉對主型態那批列重跑一次 street_key 的逐列迴圈（本 notebook 最慢的一段）
    lvr_extra, _ = clean_lvr(raw, BUILDING_EXTRA, verbose=False)
    lvr_relaxed = pd.concat([lvr, lvr_extra], ignore_index=True)
del _ids_main, _ids_extra, _overlap, _bt_raw, _dupmask, _cand
lvr_relaxed = lvr_relaxed[lvr_relaxed["district"].isin(set(districts["district"]))]
print(f"放寬後：{len(lvr_relaxed):,} 筆（嚴格 {len(lvr_m):,} 筆，"
      f"倍數 {len(lvr_relaxed) / max(len(lvr_m), 1):.2f}×）\n")

relaxed = all_schemes(lvr_relaxed, districts, villages, "六都")
R = pd.DataFrame(relaxed).set_index("方案")

# 放寬版的子母體自成一套（放寬後有交易的區會變多），故與嚴格版的 _active 分母不同，
# 但同一欄內的嚴格 vs 放寬比較仍是同口徑對同口徑。
cmp = pd.DataFrame({
    "嚴格 全母體": O["ge5_share"],
    "放寬 全母體": R["ge5_share"],
    "嚴格 子母體": O["ge5_share_active"],
    "放寬 子母體": R["ge5_share_active"],
})
print(f"n≥{MIN_N} cell 佔比比較（門檻 > {PASS_SHARE:.0%}）\n")
print(cmp.apply(lambda col: col.map(
    lambda x: "—" if pd.isna(x) else f"{x:.1%}")).to_string())

LADDER_ORDER = ["村里 × 月（A 理論上限）", "村里 × 季（A 理論上限）",
                "鄉鎮市區 × 月", "鄉鎮市區 × 季"]


def first_passing(col: str):
    for name in LADDER_ORDER:
        if R.loc[name, col] > PASS_SHARE:
            return name
    return None


relaxed_chosen = first_passing("ge5_share")
relaxed_chosen_active = first_passing("ge5_share_active")
print(f"\n放寬建物型態後第一個過關的方案（全母體）：{relaxed_chosen or '無'}")
print(f"放寬建物型態後第一個過關的方案（子母體）：{relaxed_chosen_active or '無'}")
if relaxed_chosen != relaxed_chosen_active:
    print("⚠️  放寬情境下兩個分母口徑也給出不同答案，同樣需人工拍板。")
''')

# ============================== §7 圖表 ==============================
md(r"""
---
## 7. 圖表
""")

code(r'''
import matplotlib
import matplotlib.pyplot as plt


def setup_cjk_font() -> bool:
    """讓 matplotlib 顯示中文。成功回傳 True，失敗則圖表改用英文標籤。"""
    from matplotlib import font_manager

    candidates = ["Noto Sans CJK TC", "Noto Sans CJK JP", "PingFang TC",
                  "Heiti TC", "Microsoft JhengHei", "Arial Unicode MS"]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            return True

    if IN_COLAB:
        import subprocess
        subprocess.run(["apt-get", "-qq", "install", "-y", "fonts-noto-cjk"],
                       capture_output=True)
        font_manager.fontManager.__init__()
        if any(f.name.startswith("Noto Sans CJK") for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = ["Noto Sans CJK TC", "Noto Sans CJK JP"]
            plt.rcParams["axes.unicode_minus"] = False
            return True
    return False


CJK = setup_cjk_font()
LBL = (lambda zh, en: zh) if CJK else (lambda zh, en: en)
print("中文字型：" + ("OK" if CJK else "不可用，圖表改用英文標籤"))
''')

code(r'''
scheme_short = {
    "村里 × 月（A 理論上限）": LBL("村里×月 上限", "Village×M ceil"),
    "村里 × 月（C 街道聚集）": LBL("村里×月 街道", "Village×M street"),
    "村里 × 月（B 戶數分配）": LBL("村里×月 戶數", "Village×M hh"),
    "鄉鎮市區 × 月": LBL("鄉鎮市區×月", "District×M"),
    "村里 × 季（A 理論上限）": LBL("村里×季 上限", "Village×Q ceil"),
    "村里 × 季（C 街道聚集）": LBL("村里×季 街道", "Village×Q street"),
    "村里 × 季（B 戶數分配）": LBL("村里×季 戶數", "Village×Q hh"),
    "鄉鎮市區 × 季": LBL("鄉鎮市區×季", "District×Q"),
}
ORDER = list(scheme_short)

fig, ax = plt.subplots(figsize=(12.5, 5))
x = np.arange(len(ORDER))
w = 0.38
v_full = (O.loc[ORDER, "ge5_share"] * 100).values
v_act = (O.loc[ORDER, "ge5_share_active"] * 100).values
colors = ["#c44e52" if "村里" in s else "#4c72b0" for s in ORDER]

b_full = ax.bar(x - w / 2, v_full, w, color=colors,
                label=LBL(f"全母體 {len(districts)} 區（主口徑）",
                          f"full universe, {len(districts)} districts"))
b_act = ax.bar(x + w / 2, v_act, w, facecolor="none", edgecolor=colors,
               hatch="//", linewidth=1.4,
               label=LBL(f"子母體 {len(ACTIVE_DIST)} 區（全期有交易）",
                         f"active universe, {len(ACTIVE_DIST)} districts"))
ax.axhline(PASS_SHARE * 100, color="k", ls="--", lw=1.2,
           label=LBL(f"決策門檻 {PASS_SHARE:.0%}", f"threshold {PASS_SHARE:.0%}"))

for bars, vals in ((b_full, v_full), (b_act, v_act)):
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}", ha="center", fontsize=8)

# 標出口徑會翻轉結論的方案
for i in range(len(ORDER)):
    if v_full[i] <= PASS_SHARE * 100 < v_act[i]:
        ax.annotate(LBL("口徑翻轉", "flips"), (x[i], max(v_full[i], v_act[i]) + 6),
                    ha="center", fontsize=9, color="#b00020", weight="bold")

ax.set_xticks(x)
ax.set_xticklabels([scheme_short[s] for s in ORDER], rotation=20, ha="right")
ax.set_ylabel(LBL(f"n≥{MIN_N} 的 cell 佔比 (%)", f"share of cells with n>={MIN_N} (%)"))
ax.set_title(LBL("六都整體：各粒度方案是否達到 n≥5 門檻（兩種分母口徑）",
                 "All 6 cities: clearing the n>=5 bar under both denominators"))
ax.set_ylim(0, max(np.nanmax(np.concatenate([v_full, v_act])) * 1.3, 60))
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIG_DIR / "ge5_share_by_scheme.png", dpi=150)
plt.show()
''')

code(r'''
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
CITY_COL = "ge5_share" if DENOMINATOR_PRIMARY == "full" else "ge5_share_active"
for ax, freq in zip(axes, ("月", "季")):
    subset = [s for s in ORDER if f"× {freq}" in s]
    x = np.arange(len(CITY_ORDER))
    w = 0.8 / len(subset)
    for i, s in enumerate(subset):
        v = [by_city[(by_city["方案"] == s) & (by_city["範圍"] == c)][CITY_COL].iloc[0] * 100
             for c in CITY_ORDER]
        ax.bar(x + i * w - 0.4 + w / 2, v, w, label=scheme_short[s])
    ax.axhline(PASS_SHARE * 100, color="k", ls="--", lw=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(CITY_ORDER if CJK else ["TPE", "NTP", "TYN", "TXG", "TNN", "KHH"])
    ax.set_title(LBL(f"時間粒度＝{freq}", f"time unit = {'month' if freq == '月' else 'quarter'}"))
    ax.legend(fontsize=8)
axes[0].set_ylabel(LBL(f"n≥{MIN_N} 的 cell 佔比 (%)", f"share of cells with n>={MIN_N} (%)"))
_dn = LBL("全母體" if DENOMINATOR_PRIMARY == "full" else "子母體",
          "full" if DENOMINATOR_PRIMARY == "full" else "active")
plt.suptitle(LBL(f"六都分別：密度達標比例（分母＝{_dn}）",
                 f"By city: share of cells clearing the bar ({_dn} denominator)"))
plt.tight_layout()
plt.savefig(FIG_DIR / "ge5_share_by_city.png", dpi=150)
plt.show()
''')

code(r'''
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

# (左) 鄉鎮市區 cell 的交易筆數分布（含零 cell）
ax = axes[0]
for freq, color in (("月", "#4c72b0"), ("季", "#dd8452")):
    g = _cell_counts(lvr_m, freq)
    n_zero = len(districts) * N_PERIODS[freq] - len(g)
    full = np.concatenate([g["n"].values, np.zeros(max(n_zero, 0))])
    ax.hist(np.clip(full, 0, 60), bins=np.arange(0, 62, 2), alpha=0.55,
            color=color, label=LBL(f"鄉鎮市區 × {freq}", f"District x {freq}"))
ax.axvline(MIN_N, color="k", ls="--", lw=1.2, label=f"n={MIN_N}")
ax.set_xlabel(LBL("cell 內交易筆數（≥60 併入最後一格）", "txns per cell (60+ clipped)"))
ax.set_ylabel(LBL("cell 數", "cells"))
ax.set_title(LBL("鄉鎮市區 cell 密度分布", "District cell density"))
ax.legend()

# (右) 月交易量時間序列
ax = axes[1]
ts = lvr_m.groupby(["month", "縣市"]).size().unstack("縣市").reindex(MONTHS).fillna(0)
ts.index = ts.index.to_timestamp()
for c in CITY_ORDER:
    if c in ts:
        ax.plot(ts.index, ts[c], lw=1.2, label=c if CJK else c[:2])
# 標出涵蓋率檢查旗標的月份（§3.2 Layer 3）
_fl = flags[(flags["範圍"] == "六都整體") & flags["旗標"]]["交易月"]
for _m in _fl:
    ax.axvspan(pd.Period(_m, "M").to_timestamp(),
               (pd.Period(_m, "M") + 1).to_timestamp(), color="#b00020", alpha=0.18, lw=0)
if len(_fl):
    ax.plot([], [], color="#b00020", alpha=0.4, lw=8,
            label=LBL("涵蓋率旗標月", "coverage flag"))
ax.set_title(LBL("六都電梯大樓月交易量", "Monthly elevator-tower txn volume"))
ax.set_ylabel(LBL("筆數", "txns"))
ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig(FIG_DIR / "density_distribution.png", dpi=150)
plt.show()
''')

# ============================== §8 產出 ==============================
md(r"""
---
## 8. 產出決策檔

寫出兩個檔案，後續 notebook 直接讀取，不要在 NB1–NB4 裡重新硬編粒度：

- `data/processed/density_decision.json` — 機器可讀的粒度決策
- `data/processed/density_report.md` — 人看的密度報告
""")

code(r'''
def _nan_to_none(v):
    """裸 NaN 不是合法 JSON。單元數為 0 的範圍（該縣市全期零交易）會產生 NaN。"""
    return None if isinstance(v, float) and np.isnan(v) else v


decision = {
    "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    "spec": "taiwan-housets-mvp-plan_3.md §5.0",
    "window": {"start": str(DATE_START.date()), "end": str(DATE_END.date()),
               "n_months": N_PERIODS["月"], "n_quarters": N_PERIODS["季"]},
    "criteria": {"min_n": MIN_N, "pass_share": PASS_SHARE,
                 "denominator_primary": DENOMINATOR_PRIMARY},
    "universe": {"cities": CITY_ORDER,
                 "n_districts": int(len(districts)),
                 "n_villages": int(len(villages))},
    "denominator": {
        "primary": DENOMINATOR_PRIMARY,
        "definition": {
            "full": "六都全部鄉鎮市區 × 期數（計畫 §5.0 字面讀法）",
            "active": "全期至少一筆交易的鄉鎮市區 × 期數（村里方案為這些區底下的村里）",
        },
        "full": {"n_districts": int(len(districts)), "n_villages": int(len(villages))},
        "active": {"n_districts": int(len(ACTIVE_DIST)), "n_villages": int(ACTIVE_VILL)},
        "empty_districts": sorted(set(districts["district"]) - ACTIVE_DIST),
        "conflict": bool(halted is not None),
        "conflict_schemes": [v["方案"] for v in ladder if v["判定"] == "CONFLICT"],
        "halted_at": halted["方案"] if halted else None,
    },
    "coverage": coverage,
    "cleaning": {"building_types": list(BUILDING_MAIN),
                 "n_raw": int(len(raw)), "n_clean": int(len(lvr_m)),
                 "steps": clean_steps.to_dict("records")},
    "schemes": {r["方案"]: {k: _nan_to_none(v) for k, v in r.items()
                            if k not in ("方案", "範圍")}
                for r in overall},
    # 某縣市全期零筆時 ge5_share_active 是 NaN。裸 NaN 不是合法 JSON
    # （jq 容忍，但 JS 的 JSON.parse 會炸），一律轉 null——schemes 早就這樣做了，
    # by_city 與 relaxed 當初漏掉。
    "by_city": {c: {r["方案"]: {k: _nan_to_none(r[k])
                                for k in ("ge5_share", "ge5_share_active")}
                    for r in by_city_rows if r["範圍"] == c} for c in CITY_ORDER},
    # ladder 也要洗 NaN：全期零活躍單元時 *_active 與 主口徑值 都會是 NaN，
    # 而 allow_nan=False 會讓整個寫檔動作炸掉、什麼都不留。
    "ladder": [{k: _nan_to_none(v) for k, v in item.items() if k != "細節"}
               | {"細節": {k: _nan_to_none(v) for k, v in item["細節"].items()}}
               for item in ladder],
    "relaxed": {"building_types": list(BUILDING_RELAXED),
                "n_clean": int(len(lvr_relaxed)),
                "schemes": {r["方案"]: {k: _nan_to_none(r[k])
                                        for k in ("ge5_share", "ge5_share_active")}
                            for r in relaxed},
                "first_passing": relaxed_chosen,
                "first_passing_active": relaxed_chosen_active},
    "decision": {
        "spatial_unit": chosen["方案"].split(" × ")[0] if chosen else None,
        "time_unit": chosen["方案"].split(" × ")[1] if chosen else None,
        "scheme": chosen["方案"] if chosen else None,
        "exact": bool(chosen["精確"]) if chosen else None,
        "needs_geocoding": (not chosen["精確"]) if chosen else None,
        "blocked_by_denominator_conflict": bool(chosen is None and halted is not None),
        "note": (chosen["說明"] if chosen else
                 (f"決策階梯停在「{halted['方案']}」：兩個分母口徑給出相反的 go/no-go，"
                  "需人工拍板，見 denominator 區塊" if halted else
                  "四種方案全部不過，見 relaxed 區塊")),
    },
}

path = PROC / "density_decision.json"
# allow_nan=False：寧可在這裡炸掉，也不要寫出一個 JS／嚴格 parser 讀不了的檔
payload = json.dumps(decision, ensure_ascii=False, indent=2, allow_nan=False)
path.write_text(payload, encoding="utf-8")
print(f"已寫出 {path}")
print(json.dumps(decision["decision"], ensure_ascii=False, indent=2))
''')

code(r'''
def pct(x) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.1%}"


def _r(x) -> str:
    """比值：None／NaN 印成破折號，其餘兩位小數。"""
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}"


def md_table(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    body = ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


tbl = pd.DataFrame([{
    "方案": r["方案"], "單元數": f"{r['單元數']:,}", "cell 數": f"{r['cell 數']:,}",
    "中位數": "—" if np.isnan(r["median"]) else f"{r['median']:.1f}",
    "零 cell": pct(r["zero_share"]), "n=1": pct(r["n1_share"]),
    "n≥3": pct(r["ge3_share"]), "n≥5": pct(r["ge5_share"]),
} for r in overall])

dual_tbl = pd.DataFrame([{
    "方案": r["方案"],
    "全母體 單元數": f"{r['單元數']:,}", "全母體 n≥5": pct(r["ge5_share"]),
    "子母體 單元數": f"{r['單元數_active']:,}", "子母體 n≥5": pct(r["ge5_share_active"]),
    "口徑翻轉": "⚠️" if r["ge5_share"] <= PASS_SHARE < r["ge5_share_active"] else "",
} for r in overall])

if chosen is None and halted is not None:
    conflict_block = "\n".join([
        "> ⚠️ **兩個分母口徑給出相反的 go/no-go——本報告不自動選邊，需人工拍板。**",
        ">",
        f"> 方案「{halted['方案']}」：全母體口徑 {halted['ge5_share']:.1%} → 不過；"
        f"子母體口徑 {halted['ge5_share_active']:.1%} → 過。",
        ">",
        f"> 差異來自 {len(districts) - len(ACTIVE_DIST)} 個全期零交易的區（山地、偏鄉）。"
        "它們不管選哪個粒度都不會進 NB2 的模型（§5.2 步驟 9 排除 n<3），"
        "所以算不算進分母是**口徑選擇**、不是統計問題；而 50% 這條線本身也沒有統計學依據。",
        ">",
        "> 決策階梯在此就地停住，未再往下檢查更粗的粒度。拍板方式："
        "維持全母體 → 視此方案為 FAIL、往下看更粗的粒度或放寬建物型態；"
        "改採子母體 → 把 `DENOMINATOR_PRIMARY` 改成 `\"active\"` 重跑 NB0。",
    ])
elif decision["decision"].get("needs_geocoding"):
    conflict_block = "⚠️ 此結論尚需 geocoding 實測確認（實價登錄無村里欄位）。"
else:
    conflict_block = ""

coverage_banner = ("" if coverage["passed"] else
                   "> 🚨 **涵蓋率檢查未通過——以下所有密度數字都可能被系統性低估。**"
                   "詳見下方「涵蓋率」章節。")

if coverage["passed"]:
    coverage_note = (f"十層檢查全數通過（外部錨 Layer 4 的狀態："
                     f"`{coverage['external_anchor']['status']}`）。"
                     "注意這代表「十層都沒中標」，"
                     "不是「資料一定完整」——見下方抓不到什麼。")
else:
    coverage_note = "\n".join(
        ["> ⚠️ **涵蓋率檢查未通過**（`COVERAGE_STRICT=False` 才會走到這裡）。",
         "> 以下密度數字很可能被系統性低估——受影響的月份少了一部分交易，",
         "> 但統計量看起來完全正常。定案前務必先補齊資料重跑。"] +
        [f"> - {r['範圍']} {r['交易月']}：{r['筆數']:,.0f} 筆"
         f"（局部 {_r(r['局部比值'])} / 同月跨年 {_r(r['同月跨年比值'])}）"
         for r in coverage["flagged"][:10]] +
        ([f"> - …另有 {len(coverage['flagged']) - 10} 個旗標月未列出"]
         if len(coverage["flagged"]) > 10 else []) +
        [f"> - 階梯落差：{s['範圍']} 以 {s['切點季']} 為切點 "
         f"{s['前段中位']:,.0f} → {s['後段中位']:,.0f}（{s['方向']}）"
         for s in coverage["level_shifts"][:5]] +
        ([f"> - **建物型態組零命中：{'、'.join(coverage['dead_groups'])}**"
          f"——來源用語改版了，見 notebook §3.2 Layer 2a″ 印出的普查表找新用語"]
         if coverage["dead_groups"] else []) +
        [f"> - **{r['組']} 在 {r['起']} ~ {r['迄']} 連續 {r['季數']} 季零命中"
         f"（{r['位置']}）**——那幾季的交易被系統性少算"
         for r in coverage["prefix_zero_runs"][:5]] +
        [f"> - **{r['縣市']} {r['組']} 的季命中數在 {r['切點季']} 出現階梯"
         f"（{r['前段中位']:,.0f} → {r['後段中位']:,.0f}，{r['方向']}）**"
         for r in coverage["group_level_shifts"][:5]] +
        # 以下這幾條原本漏了。若唯一的失敗來自它們，說明區塊會一個原因都沒列。
        ([f"> - **{len(coverage['bad_retention'])} 個檔案的日期率或型態佔比異常（2a′）**"
          f"：{'、'.join(r['檔名'] for r in coverage['bad_retention'][:6])}"
          + ("…" if len(coverage["bad_retention"]) > 6 else "")]
         if coverage["bad_retention"] else []) +
        ([f"> - **{len(coverage['bad_season_city'])} 個（發布季 × 縣市）筆數異常偏低（2b）**"
          f"：{'、'.join(f"{r['發布季']}{r['縣市']}" for r in coverage['bad_season_city'][:6])}"]
         if coverage["bad_season_city"] else []) +
        ([f"> - **{len(coverage['group_unchecked'])} 個（縣市 × 組）季數不足、"
          f"2a‴ 的階梯檢查沒跑**"] if coverage["group_unchecked"] else []) +
        ([f"> - **2b′ 有 {len(coverage['steps_unchecked'])} 個序列季數不足、階梯檢查沒跑**"]
         if coverage["steps_unchecked"] else []) +
        ([f"> - **外部錨的階梯 {len(coverage['external_anchor']['level_shifts'])} 個**："
          + "、".join(f"{r['縣市']} {r['切點季']} {r['比值']:.0%}"
                      for r in coverage["external_anchor"]["level_shifts"][:5])]
         if coverage["external_anchor"]["level_shifts"] else []) +
        ([f"> - **{len(coverage['external_anchor']['unchecked_cities'])} 個縣市"
          f"沒有被外部錨檢查到**："
          + "、".join(r["縣市"] for r in coverage["external_anchor"]["unchecked_cities"])]
         if coverage["external_anchor"]["unchecked_cities"] else []) +
        ([f"> - **{len(coverage['external_anchor'].get('cities_no_official') or [])} 個縣市"
          f"在官方資料裡一格都對不到**（縣市名對不上或版面改了）"]
         if coverage["external_anchor"].get("cities_no_official") else []) +
        ([f"> - **型態組比上次執行少了 {len(coverage['snapshot']['group_dropped'])} 格**"
          f"（counts 看不到型態路徑，group_counts 才看得到）"]
         if coverage["snapshot"]["group_dropped"] else []) +
        ([f"> - **Layer 4 的狀態是 `{coverage['external_anchor']['status']}` 而"
          f"ANCHOR_REQUIRED = True**"]
         if (coverage["external_anchor"]["status"] != "passed"
             and coverage["external_anchor"].get("required")) else []) +
        ([f"> - **外部錨比值越界或算不出來**："
          + "、".join(f"{r['縣市']} {r['比值'] if r['比值'] is not None else '無法比對'}"
                      for r in coverage["external_anchor"]["bad_level"])
          + f"（允許 {coverage['external_anchor']['thresholds']['ratio_min']}"
            f"–{coverage['external_anchor']['thresholds']['ratio_max']}）"]
         if coverage["external_anchor"]["bad_level"] else []) +
        ([f"> - **外部錨滾動比值旗標 {len(coverage['external_anchor']['flagged'])} 個**"
          f"（最低 {min(r['相對'] for r in coverage['external_anchor']['flagged']):.0%} 於中位）"]
         if coverage["external_anchor"]["flagged"] else []) +
        ([f"> - **比上次執行少了超過 {1 - coverage['snapshot']['thresholds']['fail']:.0%} 的格子"
          f" {len(coverage['snapshot']['dropped'])} 個**"]
         if coverage["snapshot"]["dropped"] else []) +
        ([f"> - 缺少發布季：{'、'.join(coverage['layer1']['missing_seasons'])}"]
         if coverage["layer1"]["missing_seasons"] else []) +
        [f"> - {ss} 缺主檔：{'、'.join(cs)}"
         for ss, cs in coverage["layer1"]["partial_seasons"].items()])

city_tbl = (by_city.pivot(index="方案", columns="範圍", values="ge5_share")
                   .reindex(index=ORDER, columns=CITY_ORDER)
                   .apply(lambda col: col.map(lambda x: f"{x:.1%}")).reset_index())

ladder_tbl = pd.DataFrame([{
    "順位": i + 1, "方案": v["方案"],
    "全母體 n≥5": pct(v["ge5_share"]), "子母體 n≥5": pct(v["ge5_share_active"]),
    "判定": v["判定"], "說明": v["說明"],
} for i, v in enumerate(ladder)])

relaxed_tbl = pd.DataFrame({
    "方案": O.index,
    "嚴格 全母體": [pct(x) for x in O["ge5_share"]],
    "放寬 全母體": [pct(R.loc[s, "ge5_share"]) for s in O.index],
    "嚴格 子母體": [pct(x) for x in O["ge5_share_active"]],
    "放寬 子母體": [pct(R.loc[s, "ge5_share_active"]) for s in O.index],
})

# 只把非空的警告區塊串起來，避免全部通過時在結論段後留一串空行
_blocks = "\n\n".join(x for x in (coverage_banner, conflict_block) if x)
notes_block = f"\n\n{_blocks}" if _blocks else ""

report = f"""# NB0 密度分析報告

- 產生時間：{decision['generated_at']}
- 規格來源：`taiwan-housets-mvp-plan_3.md` §5.0
- 分析期間：{DATE_START:%Y-%m} ~ {DATE_END:%Y-%m}（{N_PERIODS['月']} 個月 / {N_PERIODS['季']} 季）
- 建物型態：{' + '.join(BUILDING_MAIN)}
- 涵蓋率檢查：{'✅ 通過' if coverage['passed'] else '⚠️ 未通過（COVERAGE_STRICT=False 下仍產出）'}
  （{coverage['n_files'] - coverage['n_files_failed']}/{coverage['layer1']['n_expected']} 檔載入、
  發布季缺 {len(coverage['layer1']['missing_seasons'])} 個、旗標月份 {len(coverage['flagged'])} 個）
- 決策門檻：n ≥ {MIN_N} 的 cell 佔比 > {PASS_SHARE:.0%}（分母含零交易 cell）
- 分母口徑：主口徑＝**{'全母體' if DENOMINATOR_PRIMARY == 'full' else '子母體'}**
  ｜全母體 {len(districts):,} 區 / {len(villages):,} 村里
  ｜子母體 {len(ACTIVE_DIST):,} 區 / {ACTIVE_VILL:,} 村里（全期至少一筆交易的區）

## 結論

**{decision['decision']['scheme'] or ('⚠️ 分母口徑衝突，決策未定案' if halted else '四種方案全部不過關')}**

{decision['decision']['note']}{notes_block}

## 資料

原始 {len(raw):,} 筆 → 清整後 {len(lvr_m):,} 筆
（六都 {len(districts):,} 個鄉鎮市區、{len(villages):,} 個村里）

### 清整步驟

{md_table(clean_steps.assign(**{c: clean_steps[c].map('{:,}'.format) for c in ('剩餘筆數', '本步驟移除')}))}

## 密度統計（六都整體）

{md_table(tbl)}

村里三種估計的意義：**A 理論上限**是嚴格數學上界（任何空間分布都不可能超過）；
**B 戶數分配**是「交易完全無空間聚集」的虛無情境；**C 街道聚集**用門牌路段的實際聚集程度模擬。
恆有 A ≥ C。決策以 A 為準。

## 分母的雙口徑對照

同一個分子（n≥5 的 cell 數），除以兩個不同分母：

- **全母體**：六都全部 {len(districts):,} 個鄉鎮市區 × 期數 — 計畫 §5.0 的字面讀法
- **子母體**：{len(ACTIVE_DIST):,} 個全期至少一筆交易的區 × 期數 — 只算可能進 NB2 模型的地區

被排除的 {len(districts) - len(ACTIVE_DIST):,} 個區（{(len(districts) - len(ACTIVE_DIST)) / len(districts):.1%}）
多為山地與偏鄉，八年來一筆電梯大樓交易都沒有。它們**不管選哪個粒度都不會進模型**，
所以這個選擇不改變四個方案的排名，只改變它們有沒有越過 {PASS_SHARE:.0%} 那條線。

{md_table(dual_tbl)}

## 各縣市 n≥{MIN_N} 佔比（分母＝全母體）

{md_table(city_tbl)}

## 決策階梯

{md_table(ladder_tbl)}

## 敏感度：放寬建物型態（決策樹第 4 分支）

放寬後樣本 {len(lvr_relaxed):,} 筆（{len(lvr_relaxed) / max(len(lvr_m), 1):.2f}× 於嚴格版）。
第一個過關的方案：全母體口徑 **{relaxed_chosen or '無'}**、子母體口徑 **{relaxed_chosen_active or '無'}**
{'（⚠️ 放寬情境下兩個口徑也給出不同答案）' if relaxed_chosen != relaxed_chosen_active else ''}

放寬版的子母體自成一套（放寬後有交易的區會變多），故其 `_active` 分母與嚴格版不同；
同一欄內的嚴格 vs 放寬比較仍是同口徑對同口徑。

{md_table(relaxed_tbl)}

## 涵蓋率

下載範圍是發布季、分析窗口是交易季，兩者錯開——缺季會讓對應的交易月被系統性少算，
而密度數字照樣算得出來、看起來完全正常。因此在算密度之前先擋十層（見 notebook §1、§3.2）。

| 項目 | 值 |
|---|---|
| 發布季（1） | {coverage['layer1']['n_seasons']} 季｜缺季 {len(coverage['layer1']['missing_seasons'])} 個、缺部分縣市 {len(coverage['layer1']['partial_seasons'])} 季 |
| 檔案數（1） | {coverage['layer1']['n_files']} / {coverage['layer1']['n_expected']} |
| 檔案載入（2a） | {coverage['n_files'] - coverage['n_files_failed']} / {coverage['n_files']} |
| 日期率／型態佔比異常（2a′） | {len(coverage['bad_retention'])} 個檔案（日期率下限 {coverage['date_min']}、佔比絕對下限 住宅類 {coverage['resi_floor']}／主型態 {coverage['main_floor']}、局部基準 {coverage['resi_local_window']} 季、門檻 住宅類 {coverage['resi_local_min']}／主型態 {coverage['main_local_min']}） |
| 建物型態組零命中（2a″） | {len(coverage['dead_groups'])} 組{'：' + '、'.join(coverage['dead_groups']) if coverage['dead_groups'] else ''}（命中數 {' / '.join(f'{k} {v:,}' for k, v in coverage['building_group_hits'].items())}） |
| 組逐季零命中區間（2a″） | {len(coverage['prefix_zero_runs'])} 段（門檻：連續 {coverage['prefix_zero_run_min']} 季）{'｜' + '、'.join(f"{r['組']} {r['起']}~{r['迄']}" for r in coverage['prefix_zero_runs'][:4]) if coverage['prefix_zero_runs'] else ''} |
| （縣市 × 組）季序列階梯（2a‴） | {len(coverage['group_level_shifts'])} 個{'｜' + '、'.join(f"{r['縣市']}{r['組']} {r['比值']:.0%}" for r in coverage['group_level_shifts'][:4]) if coverage['group_level_shifts'] else ''}（跳過量太小的 {len(coverage['group_thin'])} 個組合） |
| （發布季 × 縣市）筆數異常（2b） | {len(coverage['bad_season_city'])} 個 |
| 季序列階梯落差（2b′） | {len(coverage['level_shifts'])} 個序列（下降 {coverage['step_ratio']:.0%}／上升 {coverage['step_up_ratio']:.1f}×） |
| 零交易月份 | {coverage['months_zero']} 個 |
| 逐月量體旗標（3） | {len(coverage['flagged'])} 個（門檻 {coverage['min_ratio']:.0%}） |
| 外部錨（4） | **{coverage['external_anchor']['status']}**｜比對 {coverage['external_anchor'].get('n_cells_compared', 0)}/{coverage['external_anchor'].get('n_cells_total', 0)} 格（門檻 {coverage['external_anchor']['thresholds']['min_cover']:.0%}）{'｜比值 ' + '、'.join(f"{k} {v}" for k, v in (coverage['external_anchor'].get('level_by_city') or {}).items() if v is not None) if coverage['external_anchor'].get('level_by_city') else ''} |
| 外部錨旗標（4） | 絕對水準越界 {len(coverage['external_anchor']['bad_level'])} 個縣市、滾動比值旗標 {len(coverage['external_anchor']['flagged'])} 個、階梯 {len(coverage['external_anchor']['level_shifts'])} 個、沒被檢查到的縣市 {len(coverage['external_anchor']['unchecked_cities'])} 個 |
| 跨次快照（5） | **{coverage['snapshot']['status']}**{f"｜實際比到 {coverage['snapshot'].get('n_effective', 0)}/{coverage['snapshot'].get('n_compared', 0)} 格（{coverage['snapshot'].get('cover', 0):.0%}）、下降 {len(coverage['snapshot']['dropped'])} 格、型態組下降 {len(coverage['snapshot']['group_dropped'])} 格" if coverage['snapshot']['status'] in ('passed', 'failed') else ''} |
| **結論** | **{'✅ 通過' if coverage['passed'] else '⚠️ 未通過'}** |

各範圍的最低月比值（農曆年的真實低點通常落在 0.55–0.70）：

{md_table(pd.DataFrame([{"範圍": k, "局部基準": _r(coverage["min_ratio_by_scope"][k]["局部"]), "同月跨年基準": _r(coverage["min_ratio_by_scope"][k]["同月跨年"])} for k in ["六都整體"] + CITY_ORDER if k in coverage["min_ratio_by_scope"]]))}

Layer 3 用兩個獨立基準：**局部**（置中 13 個月滾動中位數）對短缺口敏銳，但連續 ≥7 個月的
部分性少算會讓它失明；**同月跨年**（同一日曆月的其他 7 年中位數）補上這個盲點。

### 這個檢查抓不到什麼

`passed = ✅` 的意思是「十層都沒中標」，**不是「資料一定完整」**。
Layer 1 ~ 3 的基準都是從資料自己算出來的，對「基準本身被污染」沒有免疫力；
補上這一塊的是 Layer 4（外部分母＝官方買賣移轉棟數）與 Layer 5（跨次快照）。
**Layer 4 這次的狀態是 `{coverage['external_anchor']['status']}`**——
不是 `passed` 就代表那一整類缺陷這次沒有人擋。

{md_table(pd.DataFrame([{"仍然抓不到": x} for x in coverage["limitations"][1:]]))}

{coverage_note}

## 圖表

- `figs/ge5_share_by_scheme.png`
- `figs/ge5_share_by_city.png`
- `figs/density_distribution.png`

## 對後續 notebook 的意涵

- NB1：分析單元邊界用 `{decision['decision']['spatial_unit'] or '待定'}`，GEE reduceRegion 次數 ≈ 單元數 × {N_PERIODS['季']} 季
- NB2：房價聚合粒度 `{decision['decision']['scheme'] or '待定'}`；若時間粒度為「月」，遙測特徵每季 3 個月重複，須依 §5.3 處理
- NB3：paired t-test 的 pairing 單位固定為「分析單元 × 季」
- NB4：VLM 影像對數量 ≈ 單元數，成本依計畫 §9 估算
"""

(PROC / "density_report.md").write_text(report, encoding="utf-8")
print(f"已寫出 {PROC / 'density_report.md'}（{len(report):,} 字元）")
print(report[:1500])
''')

md(r"""
---
## 下一步

1. 檢視 `data/processed/density_report.md`，確認粒度決策
2. 若 `decision.blocked_by_denominator_conflict` 為 `true` → **先拍板分母口徑**再往下：
   維持全母體就把該方案當 FAIL、繼續看更粗的粒度或放寬建物型態；
   改採子母體就把 `DENOMINATOR_PRIMARY` 設成 `"active"` 重跑本 notebook
3. 若決策為村里粒度 → 需先補 geocoding（TGOS 門牌坐標 或 國土測繪中心村里界圖）才能定案
4. 若決策為鄉鎮市區粒度 → 直接進 **NB1**：上傳鄉鎮市區邊界為 GEE Asset，跑 Sentinel-2 遙測指數
5. `data/processed/lvr_clean.parquet` 已備妥，NB2 直接讀取，不需重跑清整
""")

# ============================== 組裝 ==============================
def cell_id(kind: str, source: str, seen: dict) -> str:
    """由內容決定的 cell id：沒改到的 cell 重產後位元組相同，diff 才讀得出改了什麼。
    （用隨機 id 的話每次重產整份 notebook 都會翻新。）"""
    h = hashlib.sha1(f"{kind}\n{source}".encode()).hexdigest()[:12]
    seen[h] = seen.get(h, 0) + 1
    return h if seen[h] == 1 else f"{h}-{seen[h]}"


_seen: dict = {}
nb = {
    "cells": [
        {"cell_type": t, "id": cell_id(t, s, _seen), "metadata": {},
         **({"source": s.splitlines(keepends=True)} if t == "markdown"
            else {"execution_count": None, "outputs": [], "source": s.splitlines(keepends=True)})}
        for t, s in CELLS
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
        "colab": {"provenance": [], "toc_visible": True},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

try:                       # cell id 已由 cell_id() 給定，這裡只做格式驗證
    import nbformat

    n_changes, nb = nbformat.validator.normalize(nb, version=4, version_minor=5)
    nbformat.validate(nb)
    print(f"nbformat: valid (normalize made {n_changes} fixes)")
except ImportError:
    print("WARNING: nbformat not installed — skipping validation")

out = (pathlib.Path(__file__).resolve().parent.parent
       / "notebooks" / "00_density_check.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"wrote {out}  cells={len(CELLS)}")
