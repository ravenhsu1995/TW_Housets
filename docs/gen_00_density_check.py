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
import io
import json
import re
import time
import zipfile
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

# 建物型態（前綴比對）
BUILDING_MAIN = ("住宅大樓",)                      # 住宅大樓(11層含以上有電梯)
BUILDING_RELAXED = ("住宅大樓", "華廈", "公寓")      # 決策樹第 4 分支用
# ⚠️ 不要在這裡放「舊版用語安全網」前綴。§3.2 的 Layer 2a″ 要求**每一個**前綴都必須
#    在來源中有命中，多放一個沒命中的會讓完全健康的資料硬中止，而且錯誤訊息會說
#    「來源用語改版了」——方向完全相反、極難 debug。
#    （注意這跟下面 NOTE_EXCLUDE 的慣例相反：那裡刻意保留舊用語當安全網。）

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


class CoverageError(RuntimeError):
    """涵蓋率檢查失敗。數字會被系統性低估但看起來正常，故預設中止。"""


def coverage_fail(title: str, detail: str) -> None:
    """涵蓋率檢查不通過時的統一出口：預設 raise，COVERAGE_STRICT=False 時降為警告。"""
    msg = f"{title}\n{detail}"
    if COVERAGE_STRICT:
        raise CoverageError(
            msg + "\n\n若已確認可以接受不完整的涵蓋，把 COVERAGE_STRICT 設為 False 再重跑。")
    print(f"{msg}\n\n（COVERAGE_STRICT = False，僅警告不中止——數字可能被系統性低估）\n")

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

載入時就先篩掉非住宅類（只留住宅大樓／華廈／公寓），把 `raw` 的記憶體佔用壓到約 1/3——
Colab 免費版只有 ~12 GB RAM，全型態全欄位讀進來會很緊。
""")

code(r'''
NUMERIC_COLS = [
    "總價元", "單價元平方公尺", "建物移轉總面積平方公尺",
    "車位移轉總面積平方公尺", "車位總價元", "土地移轉總面積平方公尺",
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


LOAD_STATS: list = []      # 每個 csv 的列數保留情況，§3.2 Layer 2a′ 使用
BUILDING_SEEN: dict = {}   # 建物型態的原始用語普查（在型態篩選**之前**），§3.2 Layer 2a″ 使用


def load_one(path: Path) -> Optional[pd.DataFrame]:
    season, code = path.stem.split("_")
    try:
        df = pd.read_csv(path, dtype=str, encoding="utf-8-sig", low_memory=False)
    except Exception as exc:  # noqa: BLE001
        print(f"  讀取失敗 {path.name}: {exc}")
        return None

    df.columns = [COL_ALIASES.get(c.strip(), c.strip()) for c in df.columns]
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        print(f"  ⚠️  {path.name} 缺少欄位 {missing}，跳過。"
              "（若大量檔案出現，代表來源欄名又改了，須更新 COL_ALIASES）")
        return None

    # 丟掉英文表頭列與任何日期格式不合的列
    ok_date = df["交易年月日"].astype(str).str.strip().str.fullmatch(r"\d{6,7}")
    # 只留住宅類（BUILDING_RELAXED 是嚴格版的超集），把 raw 的記憶體壓到 1/3
    ok_type = df["建物型態"].fillna("").str.startswith(BUILDING_RELAXED)

    # 這兩道篩選是「靜默丟列」——來源用語改版（實價登錄 2.0 就是 110S3 生效）
    # 會讓某季起的列被大量丟掉，而 load_one 照樣回傳一個看似正常的 DataFrame。
    # 逐檔記下保留情況，§3.2 Layer 2a′ 據此比對各季是否一致。
    # 在型態篩選之前先記下所有用語。篩選之後就只剩對得上的，看不到來源改成了什麼——
    # 而「來源改成了什麼」正是修正 BUILDING_RELAXED 時唯一需要的資訊。
    for _bt, _n in df.loc[ok_date, "建物型態"].fillna("(空白)").value_counts().items():
        BUILDING_SEEN[_bt] = BUILDING_SEEN.get(_bt, 0) + int(_n)

    LOAD_STATS.append({
        "發布季": season, "縣市": CITY_CODES[code], "檔名": path.name,
        "原始列數": len(df),
        "日期可解析": int(ok_date.sum()),
        "住宅類": int((ok_date & ok_type).sum()),
    })

    df = df[ok_date & ok_type].copy()
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

print(f"合併後原始筆數：{len(raw):,}"
      f"（成功載入 {len(files) - len(load_failed)}/{len(files)} 檔）")
print(raw.groupby("縣市").size().reindex(CITY_ORDER).to_string())
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

    # 3. 交易日期區間
    df["交易日"] = roc_to_datetime(df["交易年月日"])
    df = log("3. 交易日期區間",
             df[df["交易日"].between(DATE_START, DATE_END)])

    # 4. 去重
    df = log("4. 編號去重", df.drop_duplicates(subset=["縣市", "編號"], keep="first"))

    # 5. 備註排除
    note = df["備註"].fillna("")
    df = log("5. 備註排除",
             df[~note.str.contains("|".join(NOTE_EXCLUDE), regex=True)])

    # ---- 數值欄位 ----
    for c in NUMERIC_COLS:
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
| **2a″** | 每個建物型態前綴都有命中 | 某個前綴從此對不上（保留率門檻接不到） | ✅ |
| **2b** | 每個（發布季 × 縣市）筆數合理 | 檔案截斷、解析後幾乎空的 | ✅ |
| **2b′** | 季序列沒有持續性階梯落差 | 「從某季起一路少算」 | ✅ |
| **2c** | 發布季 × 交易月 貢獻矩陣 | 診斷：坑是哪一季造成的 | — |
| **3** | 逐月交易量離群（雙基準） | 前幾層沒抓到的殘留缺口 | ✅ |

中止行為由 `COVERAGE_STRICT` 控制（見 §0）。

**為什麼要 2a′ 和 2b′：** Layer 3 用置中滾動中位數當基準，對**短**缺口很敏銳，
但連續 ≥7 個月的部分性少算會讓基準跟著沉下去而完全失明。而「來源用語從某季起改版、
之後每季都少算一部分」恰好就是這個形狀（實價登錄 2.0 於 110S3 生效，是真實的前例）——
那種缺口 2a、2b、Layer 3 都攔不到，得靠 2a′ 在源頭量保留率、2b′ 在季序列上找階梯。

#### 已知限制（請當成這一節的使用說明）

除了 2a′ 的**日期解析率**（有絕對期望值：應該接近 1，只差一列英文表頭）之外，
**其餘每一層的基準都是從資料自己算出來的**。這代表它們對「基準本身被污染」沒有免疫力：

| 仍然抓不到的形狀 | 為什麼 |
|---|---|
| 全窗口等比例少算 | 所有基準一起等比例沉下去，相對值不變 |
| 每年重複出現的長缺口 | 局部基準沉下去；跨年基準因「其他年份同月也低」而翻轉，比值甚至 > 1 |
| 持續但溫和的少算（掉到 55–70%） | 幅度落在 2b/2b′/L3 三個門檻的餘裕帶內 |

已做的緩解：2a′ 的住宅類佔比基準改用該縣市的**季最大值**（用語改版只會讓佔比往下掉，
不會往上，所以最大值不會被污染）；2b′ **上下兩個方向都找**（只找下降的話，
「序列前段被少算」呈現為上升而永遠看不到）。

要真正關掉這一類，需要一個**外部絕對錨**——例如拿內政部公布的建物買賣移轉棟數
做跨來源比對，或把上次成功執行的季筆數矩陣存成快照做跨次比對。目前兩者都沒做。
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
# load_one 有兩道靜默的列篩選（日期格式、建物型態前綴）。來源用語改版時，
# 檔案讀得進來、欄位也齊全，但列被大量丟掉——這是 Layer 3 的滾動中位數
# 完全看不見的形狀，只能在源頭量。
ls = pd.DataFrame(LOAD_STATS)
ls["日期解析率"] = ls["日期可解析"] / ls["原始列數"].replace(0, np.nan)
ls["住宅類佔比"] = ls["住宅類"] / ls["日期可解析"].replace(0, np.nan)

# 日期解析率有**絕對**期望值：應該接近 1，只差一列英文表頭。這一支不依賴任何
# 從資料算出來的基準，是本節唯一不會被污染的檢查。
#
# 住宅類佔比因縣市而異（臺北高、臺南低），只能跟同縣市其他季比。基準的選擇是
# 「抗污染」與「抗離群」的取捨：
#   · 中位數 → 受影響季數一過半就自己變低值，整個檢查靜默失效
#   · 最大值 → 抗污染到 100%，但分母由單一最高季決定、毫無 robustness，
#              某季偏高就會讓大批健康檔案誤報，而且訊息指的方向是反的
#   · p90（採用）→ 要 >90% 的季受污染才失效，同時對單季離群免疫
RESI_Q = 0.90
q_resi = ls.groupby("縣市")["住宅類佔比"].transform(lambda g: g.quantile(RESI_Q))
ls["住宅類佔比相對"] = ls["住宅類佔比"] / q_resi.replace(0, np.nan)

# 絕對下限：佔比低到這個地步，代表 BUILDING_RELAXED 的前綴幾乎對不上任何列。
# 不依賴資料自身基準，所以是少數能擋「全部 36 季都受同一次用語改版影響」的檢查
# （那種情況下相對值會全是 1.0）。0.10 對健康的臺南（實測約 0.26）仍有 2.6 倍餘裕，
# 覆蓋到「保留率 < 36%」。再往上調就會逼近臺南的正常值，不安全。
# 「部分前綴對不上」不是調這個數字能解的——跨縣市的佔比差異比那個缺陷的訊號還大，
# 要靠下面的建物型態名稱普查。
RESI_FLOOR = 0.10

print("列數保留率（各縣市）\n")
print(ls.groupby("縣市")
        .agg(日期解析率中位=("日期解析率", "median"),
             住宅類佔比最小=("住宅類佔比", "min"),
             住宅類佔比中位=("住宅類佔比", "median"),
             住宅類佔比最大=("住宅類佔比", "max"),
             住宅類佔比相對最小=("住宅類佔比相對", "min"))
        .reindex(CITY_ORDER).round(3).to_string())
print(f"\n門檻校準參考：上表「住宅類佔比相對最小」離門檻 0.60 越近，誤判風險越高。")
print(f"  若有縣市的 住宅類佔比 min/max 低於 0.70，代表季間變異比預期大，門檻須放寬。")
print(f"  （此門檻尚未用真實資料校準過——目前的餘裕是在合成資料上量的。）")

bad_ret = ls[(ls["日期解析率"] < 0.95) | (ls["住宅類佔比相對"] < 0.60)
             | (ls["住宅類佔比"] < RESI_FLOOR)
             | ls["日期解析率"].isna() | ls["住宅類佔比"].isna()]
if len(bad_ret):
    coverage_fail(
        f"❌ {len(bad_ret)} 個檔案的列數保留率異常",
        "\n".join([f"    {r['檔名']}：{r['原始列數']:,} 列 → 日期可解析 {r['日期可解析']:,}"
                    f"（{r['日期解析率']:.1%}）→ 住宅類 {r['住宅類']:,}"
                    f"（佔可解析 {r['住宅類佔比']:.1%}，該縣市中位的 {r['住宅類佔比相對']:.0%}）"
                    for _, r in bad_ret.sort_values("檔名").head(30).iterrows()] +
                  ([f"    …另有 {len(bad_ret) - 30} 個"] if len(bad_ret) > 30 else []) +
                  ["",
                   "日期解析率偏低 → 交易年月日 的格式變了，檢查 load_one 的 fullmatch 樣式。",
                   "住宅類佔比偏低 → 建物型態 的用語變了，檢查 BUILDING_RELAXED 的前綴。",
                   f"（基準是該縣市季的 p{int(RESI_Q * 100)}，要超過 {1 - RESI_Q:.0%} 的季受污染才會失效；"
                   f"另有絕對下限 {RESI_FLOOR}，擋全部 36 季一起改版的情況。）",
                   "這兩者都會讓對應的交易月被系統性少算，而統計量看起來完全正常。"]))
else:
    print(f"\n✅ Layer 2a′：{len(ls)} 個檔案的列數保留率均與同縣市其他季一致")
''')

code(r'''
# ---- Layer 2a″：建物型態的名稱層級普查 ----
# 保留率型的門檻對「某一個前綴從此對不上」本質上接不到——跨縣市的正常佔比差異
# （臺北 0.6 vs 臺南 0.26）比這個缺陷造成的訊號還大。名稱普查是絕對檢查：
# 不吃比例、不吃跨縣市差異，直接問「這個前綴到底有沒有命中過任何一列」。
bs = pd.Series(BUILDING_SEEN).sort_values(ascending=False)
hits = {p: int(bs[bs.index.str.startswith(p)].sum()) for p in BUILDING_RELAXED}

print(f"建物型態用語普查（型態篩選前，共 {len(bs)} 種用語 / {bs.sum():,} 列）\n")
print(bs.head(15).to_string())
print(f"\nBUILDING_RELAXED 各前綴的命中數：{hits}")
print(f"其中嚴格版 BUILDING_MAIN = {list(BUILDING_MAIN)}")

dead_prefix = [p for p, n in hits.items() if n == 0]
if dead_prefix:
    coverage_fail(
        f"❌ BUILDING_RELAXED 有 {len(dead_prefix)} 個前綴在來源中零命中：{'、'.join(dead_prefix)}",
        "\n".join(["    來源的建物型態用語改版了，這個前綴從此對不上任何一列。",
                    "    上表是來源實際出現的用語（篩選前），據此更新 BUILDING_RELAXED。",
                    "",
                    f"    若零命中的是 BUILDING_MAIN（{list(BUILDING_MAIN)}）之一，"
                    "整份密度分析的 Y 會是空的；",
                    "    若是華廈／公寓，只影響 §6 的放寬敏感度分析。"]))
else:
    print(f"\n✅ Layer 2a″：{len(BUILDING_RELAXED)} 個建物型態前綴都有命中")
''')

code(r'''
# ---- 涵蓋率的量測基底 ----
# 用 raw（住宅類全型態）而非 lvr（僅電梯大樓）：涵蓋率是「資料有沒有到齊」的性質，
# 與建物型態篩選無關；而全型態的筆數多好幾倍，月比值更穩，
# 較不會把臺南這種樣本本來就薄的縣市的正常波動誤判成缺口。
_m = roc_to_datetime(raw["交易年月日"]).dt.to_period("M")
_in = _m.notna() & (_m >= MONTHS[0]) & (_m <= MONTHS[-1])
# 縣市／發布季只有 6 與 36 種值，用 category 存；object dtype 在 250 萬列的
# 真實資料上會多吃數百 MB，而 raw 本身已經很佔記憶體。
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


# 0.55：市場循環（2017 低 → 2021 高 → 2023 回落）造成的前後段中位數比，實測落在
# 0.94 以上（中位數把循環大幅平滑掉了），餘裕充足；掉到 0.55 以下是資料問題不是市場。
# 上升方向**不**取倒數。實測「階梯式的真實成長」只要 2.0× 就會撞到 1.82
# （前段可以只有 4 季，容易被序列開頭的真實低谷帶低），而真缺陷造成的上升是
# 3.2–3.5×，餘裕很足。所以上升門檻另外設在 2.5，換掉那段誤報空間。
STEP_RATIO = 0.55
STEP_UP_RATIO = 2.5
steps = []
for c in list(CITY_ORDER) + ["六都合計"]:
    series = sc.sum(axis=1) if c == "六都合計" else sc[c]
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
else:
    print(f"✅ Layer 2b′：7 個季序列均無持續性落差"
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
    "resi_quantile": RESI_Q,
    "resi_floor": RESI_FLOOR,
    # 門檻校準用：真實資料跑過一次後，看各縣市 住宅類佔比 的 min/max 比值，
    # 低於 0.70 代表季間變異比預期大、2a′ 的 0.60 門檻須放寬。
    # 某縣市 36 季全部 NaN（該縣市完全沒有資料）時 min/quantile/max 都是 NaN。
    # coverage 裡其他可能含 NaN 的欄位都有防護，這個新加的當初漏了。
    "resi_share_by_city": {
        c: {k: _cov_num(v) for k, v in
            (("min", g.min()), ("p90", g.quantile(RESI_Q)), ("max", g.max()))}
        for c, g in ls.groupby("縣市")["住宅類佔比"]},
    "n_seasons": len(SEASONS),
    "n_files": len(files),
    "n_files_failed": len(load_failed),
    "failed_files": load_failed,
    "bad_retention": _jsonable(
        bad_ret[["檔名", "原始列數", "日期解析率", "住宅類佔比相對"]].round(3).to_dict("records")),
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
    "building_type_hits": hits,
    "dead_prefixes": dead_prefix,
    "passed": bool(not LAYER1["missing_seasons"] and not LAYER1["partial_seasons"]
                   and not load_failed and len(bad_ret) == 0 and not dead_prefix
                   and not dead_city
                   and not bad_sc and not steps and flagged.empty),
    # passed=True 的意思是「這六層都沒中標」，不是「資料一定完整」。
    # 把限制一起寫進 JSON，免得下游只看到 passed=True 就以為萬無一失。
    "limitations": [
        "除 2a′ 的日期解析率外，所有基準都由資料自身算出，對「基準被污染」無免疫力",
        "全窗口等比例少算：所有基準一起沉，抓不到",
        "每年重複出現的長缺口：局部基準沉下去、跨年基準被翻轉（比值可能 > 1）",
        "「來源本身變少」路徑的溫和持續少算，兩個方向都有盲帶且不對稱："
        "序列後段掉到 55–70% 實測靜默損失達 23%；序列前段掉到 45–55%（2b′ 上升門檻 2.5 之下）"
        "實測靜默損失達 31–38%。用語改版路徑有 2a′ 兜著，純檔案截斷路徑則沒人接",
        "用語改版只影響**部分**發布季且保留率 > 65%：三層都接不到。"
        "2a′ 的嚴重度不夠（偵測邊界 65%）、絕對下限 0.10 也躲得過、"
        "而 2a″ 是全域普查，只要有任何一季還留著舊用語就 hits > 0。"
        "最現實的劇本正是這一格（實價登錄 2.0 於 110S3 起改用語、影響 18/36 季）。"
        "解法是把 2a″ 改成逐季普查——偵測「某前綴從某季起命中數變成 0」的階梯，"
        "而非只問「有沒有命中過」。尚未實作",
        "門檻尚未用真實資料校準：目前的誤報餘裕都是在合成資料上量的",
        "尚未接外部絕對錨（內政部建物買賣移轉棟數／跨次執行快照）",
    ],
}
assert json.dumps(coverage, ensure_ascii=False, allow_nan=False)   # 必須是合法 JSON
print(f"\ncoverage.passed = {coverage['passed']}")
''')

code(r'''
# ---- 存檔給 NB2 重用，避免重複清整 ----
KEEP = [
    "縣市", "鄉鎮市區", "district", "交易日", "month", "quarter",
    "土地位置建物門牌", "street", "street_key", "編號",
    "建物型態", "交易標的", "總價元", "車位總價元", "房屋總價",
    "建物移轉總面積平方公尺", "車位移轉總面積平方公尺", "房屋面積坪", "單價元每坪",
    "n_土地", "n_建物", "n_車位", "車位價未拆",
    "移轉層次", "總樓層數", "建築完成年月", "都市土地使用分區",
    "建物現況格局-房", "建物現況格局-廳", "建物現況格局-衛", "備註", "發布季",
]
out = lvr[[c for c in KEEP if c in lvr.columns]].copy()
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
lvr_relaxed, relaxed_steps = clean_lvr(raw, BUILDING_RELAXED, verbose=False)
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
    coverage_note = ("六層檢查全數通過。注意這代表「六層都沒中標」，"
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
        ([f"> - **建物型態前綴零命中：{'、'.join(coverage['dead_prefixes'])}**"
          f"——來源用語改版了，見 notebook §3.2 Layer 2a″ 印出的普查表找新用語"]
         if coverage["dead_prefixes"] else []) +
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
而密度數字照樣算得出來、看起來完全正常。因此在算密度之前先擋六層（見 notebook §1、§3.2）。

| 項目 | 值 |
|---|---|
| 發布季（1） | {coverage['layer1']['n_seasons']} 季｜缺季 {len(coverage['layer1']['missing_seasons'])} 個、缺部分縣市 {len(coverage['layer1']['partial_seasons'])} 季 |
| 檔案數（1） | {coverage['layer1']['n_files']} / {coverage['layer1']['n_expected']} |
| 檔案載入（2a） | {coverage['n_files'] - coverage['n_files_failed']} / {coverage['n_files']} |
| 列數保留率異常（2a′） | {len(coverage['bad_retention'])} 個檔案（門檻：相對 p{int(coverage['resi_quantile'] * 100)} 的 60%、絕對下限 {coverage['resi_floor']}） |
| 建物型態前綴零命中（2a″） | {len(coverage['dead_prefixes'])} 個{'：' + '、'.join(coverage['dead_prefixes']) if coverage['dead_prefixes'] else ''}（命中數 {' / '.join(f'{k} {v:,}' for k, v in coverage['building_type_hits'].items())}） |
| （發布季 × 縣市）筆數異常（2b） | {len(coverage['bad_season_city'])} 個 |
| 季序列階梯落差（2b′） | {len(coverage['level_shifts'])} 個序列（下降 {coverage['step_ratio']:.0%}／上升 {coverage['step_up_ratio']:.1f}×） |
| 零交易月份 | {coverage['months_zero']} 個 |
| 逐月量體旗標（3） | {len(coverage['flagged'])} 個（門檻 {coverage['min_ratio']:.0%}） |
| **結論** | **{'✅ 通過' if coverage['passed'] else '⚠️ 未通過'}** |

各範圍的最低月比值（農曆年的真實低點通常落在 0.55–0.70）：

{md_table(pd.DataFrame([{"範圍": k, "局部基準": _r(coverage["min_ratio_by_scope"][k]["局部"]), "同月跨年基準": _r(coverage["min_ratio_by_scope"][k]["同月跨年"])} for k in ["六都整體"] + CITY_ORDER if k in coverage["min_ratio_by_scope"]]))}

Layer 3 用兩個獨立基準：**局部**（置中 13 個月滾動中位數）對短缺口敏銳，但連續 ≥7 個月的
部分性少算會讓它失明；**同月跨年**（同一日曆月的其他 7 年中位數）補上這個盲點。

### 這個檢查抓不到什麼

`passed = ✅` 的意思是「六層都沒中標」，**不是「資料一定完整」**。除了 2a′ 的日期解析率
（有絕對期望值）之外，其餘每一層的基準都是從資料自己算出來的，對「基準本身被污染」
沒有免疫力：

{md_table(pd.DataFrame([{"仍然抓不到": x} for x in coverage["limitations"][1:]]))}

要真正關掉這一類需要外部絕對錨（內政部建物買賣移轉棟數比對、或跨次執行的快照比對），
目前沒有做。

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
