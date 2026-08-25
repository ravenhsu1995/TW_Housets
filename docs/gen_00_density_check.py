# -*- coding: utf-8 -*-
"""產生 notebooks/00_density_check.ipynb"""
import json, pathlib

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

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"}

print(f"分析期間 {DATE_START:%Y-%m} ~ {DATE_END:%Y-%m}"
      f"（{(DATE_END.year - DATE_START.year) * 12 + DATE_END.month - DATE_START.month + 1} 個月 / "
      f"{len(pd.period_range(DATE_START, DATE_END, freq='Q'))} 季）")
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
files = sorted(LVR_DIR.glob("*_[afhbde].csv"))
print(f"已取得 {len(files)} 個檔案（預期 {len(SEASONS) * 6}），"
      f"共 {sum(f.stat().st_size for f in files) / 1e6:.0f} MB")

missing = [s for s in SEASONS if not (LVR_DIR / f".{s}.done").exists()]
if missing:
    print(f"⚠️  缺少發布季：{missing}")
else:
    print("✅ 全部發布季齊備")
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
    keep = df["交易年月日"].astype(str).str.strip().str.fullmatch(r"\d{6,7}")
    # 只留住宅類（BUILDING_RELAXED 是嚴格版的超集），把 raw 的記憶體壓到 1/3
    keep &= df["建物型態"].fillna("").str.startswith(BUILDING_RELAXED)
    df = df[keep].copy()

    df["縣市"] = CITY_CODES[code]
    df["發布季"] = season
    return df


frames = [d for p in files if (d := load_one(p)) is not None]
raw = pd.concat(frames, ignore_index=True)
del frames

print(f"合併後原始筆數：{len(raw):,}")
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
""")

code(r'''
MONTHS = pd.period_range(DATE_START, DATE_END, freq="M")
QUARTERS = pd.period_range(DATE_START, DATE_END, freq="Q")
N_PERIODS = {"月": len(MONTHS), "季": len(QUARTERS)}


def density_stats(counts: np.ndarray, n_units: int, n_periods: int,
                  label: str = "", scope: str = "六都") -> dict:
    """counts：非零 cell 的交易筆數陣列。零 cell 由 n_units*n_periods 補齊。"""
    counts = np.asarray(counts, dtype=float)
    total_cells = int(n_units) * int(n_periods)
    n_zero = max(total_cells - counts.size, 0)
    if counts.size > total_cells:
        raise ValueError(f"{label}: 非零 cell({counts.size}) 超過總 cell({total_cells})")

    full = np.concatenate([counts, np.zeros(n_zero)])
    return {
        "方案": label,
        "範圍": scope,
        "單元數": int(n_units),
        "期數": int(n_periods),
        "cell 數": total_cells,
        "交易筆數": int(counts.sum()),
        "median": float(np.median(full)),
        "p25": float(np.percentile(full, 25)),
        "p75": float(np.percentile(full, 75)),
        "mean": float(full.mean()),
        "zero_share": n_zero / total_cells,
        "n1_share": float((counts == 1).sum()) / total_cells,
        "ge3_share": float((counts >= 3).sum()) / total_cells,
        "ge5_share": float((counts >= MIN_N).sum()) / total_cells,
    }


def show(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    fmt = df.copy()
    for c in ("zero_share", "n1_share", "ge3_share", "ge5_share"):
        fmt[c] = df[c].map(lambda x: "—" if pd.isna(x) else f"{x * 100:.1f}%")
    for c in ("median", "p25", "p75", "mean"):
        fmt[c] = df[c].map(lambda x: "—" if pd.isna(x) else f"{x:.2f}")
    fmt["cell 數"] = df["cell 數"].map("{:,}".format)
    fmt["交易筆數"] = df["交易筆數"].map("{:,}".format)
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
    return density_stats(g["n"].values, len(dist_pop), N_PERIODS[freq],
                         f"鄉鎮市區 × {freq}", scope)


# ---------- 村里 A：理論上限 ----------
def village_ceiling(df, dist_pop, freq, scope="六都") -> dict:
    """任意分配方式下，n≥k 的 cell 數不可能超過 Σ min(V, ⌊T/k⌋)。這是嚴格上界。"""
    V = dist_pop.set_index("district")["村里數"]
    g = _cell_counts(df, freq)
    g["V"] = g["district"].map(V)
    g = g.dropna(subset=["V"])

    n_units, n_periods = int(V.sum()), N_PERIODS[freq]
    total_cells = n_units * n_periods

    def ceil_share(k: int) -> float:
        return float(np.minimum(g["V"], g["n"] // k).sum()) / total_cells

    # 中位數上限：最大的 m 使得「n≥m 的 cell」還能佔到一半以上
    med_ceiling = 0
    for m in range(1, 201):
        if ceil_share(m) >= 0.5:
            med_ceiling = m
        else:
            break

    min_zero = 1.0 - float(np.minimum(g["V"], g["n"]).sum()) / total_cells
    return {
        "方案": f"村里 × {freq}（A 理論上限）", "範圍": scope,
        "單元數": n_units, "期數": n_periods, "cell 數": total_cells,
        "交易筆數": int(g["n"].sum()),
        "median": float(med_ceiling), "p25": np.nan, "p75": np.nan, "mean": np.nan,
        "zero_share": min_zero, "n1_share": np.nan,
        "ge3_share": ceil_share(3), "ge5_share": ceil_share(MIN_N),
    }


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
    return density_stats(counts, len(vill_pop), N_PERIODS[freq],
                         f"村里 × {freq}（B 戶數分配）", scope)


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
    return density_stats(keep["n"].values, int(V.sum()), N_PERIODS[freq],
                         f"村里 × {freq}（C 街道聚集）", scope)


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

overall = all_schemes(lvr_m, districts, villages, "六都")
show(overall).to_string(index=False)
''')

code(r'''
display_cols = ["方案", "單元數", "cell 數", "median", "p25", "p75",
                "zero_share", "n1_share", "ge3_share", "ge5_share"]
print(show(overall)[display_cols].to_string(index=False))
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

pivot = (by_city.pivot(index="方案", columns="範圍", values="ge5_share")
                .reindex(columns=CITY_ORDER) * 100).round(1)
pivot["六都整體"] = (pd.DataFrame(overall).set_index("方案")["ge5_share"] * 100).round(1)
print(f"各方案 n≥{MIN_N} 的 cell 佔比（%），決策門檻 > {PASS_SHARE:.0%}\n")
print(pivot.to_string())
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
""")

code(r'''
O = pd.DataFrame(overall).set_index("方案")


def verdict_village(freq: str) -> dict:
    ceil = O.loc[f"村里 × {freq}（A 理論上限）", "ge5_share"]
    street = O.loc[f"村里 × {freq}（C 街道聚集）", "ge5_share"]
    hh = O.loc[f"村里 × {freq}（B 戶數分配）", "ge5_share"]
    if ceil <= PASS_SHARE:
        status, note = "FAIL", f"理論上限 {ceil:.1%} ≤ {PASS_SHARE:.0%}，任何空間分布都不可能達標"
    elif street > PASS_SHARE:
        status, note = "PASS?", f"上限 {ceil:.1%}、街道情境 {street:.1%} 皆過關，但須 geocoding 實測確認"
    else:
        status, note = "UNDETERMINED", (f"上限 {ceil:.1%} 過關但街道情境僅 {street:.1%}，"
                                        "結論取決於實際村里分布，須 geocoding")
    return {"方案": f"村里 × {freq}", "ge5_share": ceil, "判定": status,
            "說明": note, "精確": False,
            "細節": {"上限A": ceil, "街道C": street, "戶數B": hh}}


def verdict_district(freq: str) -> dict:
    share = O.loc[f"鄉鎮市區 × {freq}", "ge5_share"]
    ok = share > PASS_SHARE
    return {"方案": f"鄉鎮市區 × {freq}", "ge5_share": share,
            "判定": "PASS" if ok else "FAIL",
            "說明": f"精確值 {share:.1%} {'>' if ok else '≤'} {PASS_SHARE:.0%}",
            "精確": True, "細節": {}}


ladder = [verdict_village("月"), verdict_village("季"),
          verdict_district("月"), verdict_district("季")]

chosen = None
for v in ladder:
    mark = {"PASS": "✅", "PASS?": "🟡", "UNDETERMINED": "🟡", "FAIL": "❌"}[v["判定"]]
    print(f"{mark} {v['方案']:<12} ge5={v['ge5_share']:6.1%}  {v['判定']:<13} {v['說明']}")
    if chosen is None and v["判定"] in ("PASS", "PASS?"):
        chosen = v

print()
if chosen:
    print(f"👉 建議粒度：**{chosen['方案']}**"
          + ("（精確判定）" if chosen["精確"] else "（需 geocoding 確認）"))
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

cmp = pd.DataFrame({
    "嚴格（僅電梯大樓）": O["ge5_share"],
    "放寬（+華廈+公寓）": R["ge5_share"],
})
print(f"n≥{MIN_N} cell 佔比比較\n")
print((cmp * 100).round(1).astype(str).add("%").to_string())

relaxed_chosen = None
for name in ["村里 × 月（A 理論上限）", "村里 × 季（A 理論上限）",
             "鄉鎮市區 × 月", "鄉鎮市區 × 季"]:
    if R.loc[name, "ge5_share"] > PASS_SHARE:
        relaxed_chosen = name
        break
print(f"\n放寬建物型態後第一個過關的方案：{relaxed_chosen or '無'}")
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

fig, ax = plt.subplots(figsize=(11, 4.5))
vals = O.loc[ORDER, "ge5_share"] * 100
colors = ["#c44e52" if "村里" in s else "#4c72b0" for s in ORDER]
bars = ax.bar([scheme_short[s] for s in ORDER], vals, color=colors)
ax.axhline(PASS_SHARE * 100, color="k", ls="--", lw=1.2,
           label=LBL(f"決策門檻 {PASS_SHARE:.0%}", f"threshold {PASS_SHARE:.0%}"))
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}%", ha="center", fontsize=9)
ax.set_ylabel(LBL(f"n≥{MIN_N} 的 cell 佔比 (%)", f"share of cells with n>={MIN_N} (%)"))
ax.set_title(LBL("六都整體：各粒度方案是否達到 n≥5 門檻",
                 "All 6 cities: does each granularity clear the n>=5 bar?"))
ax.set_ylim(0, max(vals.max() * 1.2, 60))
ax.legend()
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(FIG_DIR / "ge5_share_by_scheme.png", dpi=150)
plt.show()
''')

code(r'''
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
for ax, freq in zip(axes, ("月", "季")):
    subset = [s for s in ORDER if f"× {freq}" in s]
    x = np.arange(len(CITY_ORDER))
    w = 0.8 / len(subset)
    for i, s in enumerate(subset):
        v = [by_city[(by_city["方案"] == s) & (by_city["範圍"] == c)]["ge5_share"].iloc[0] * 100
             for c in CITY_ORDER]
        ax.bar(x + i * w - 0.4 + w / 2, v, w, label=scheme_short[s])
    ax.axhline(PASS_SHARE * 100, color="k", ls="--", lw=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(CITY_ORDER if CJK else ["TPE", "NTP", "TYN", "TXG", "TNN", "KHH"])
    ax.set_title(LBL(f"時間粒度＝{freq}", f"time unit = {'month' if freq == '月' else 'quarter'}"))
    ax.legend(fontsize=8)
axes[0].set_ylabel(LBL(f"n≥{MIN_N} 的 cell 佔比 (%)", f"share of cells with n>={MIN_N} (%)"))
plt.suptitle(LBL("六都分別：密度達標比例", "By city: share of cells clearing the bar"))
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
decision = {
    "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    "spec": "taiwan-housets-mvp-plan_3.md §5.0",
    "window": {"start": str(DATE_START.date()), "end": str(DATE_END.date()),
               "n_months": N_PERIODS["月"], "n_quarters": N_PERIODS["季"]},
    "criteria": {"min_n": MIN_N, "pass_share": PASS_SHARE},
    "universe": {"cities": CITY_ORDER,
                 "n_districts": int(len(districts)),
                 "n_villages": int(len(villages))},
    "cleaning": {"building_types": list(BUILDING_MAIN),
                 "n_raw": int(len(raw)), "n_clean": int(len(lvr_m)),
                 "steps": clean_steps.to_dict("records")},
    "schemes": {r["方案"]: {k: (None if isinstance(v, float) and np.isnan(v) else v)
                            for k, v in r.items() if k not in ("方案", "範圍")}
                for r in overall},
    "by_city": {c: {r["方案"]: r["ge5_share"]
                    for r in by_city_rows if r["範圍"] == c} for c in CITY_ORDER},
    "ladder": [{k: v for k, v in item.items() if k != "細節"} | {"細節": item["細節"]}
               for item in ladder],
    "relaxed": {"building_types": list(BUILDING_RELAXED),
                "n_clean": int(len(lvr_relaxed)),
                "schemes": {r["方案"]: r["ge5_share"] for r in relaxed},
                "first_passing": relaxed_chosen},
    "decision": {
        "spatial_unit": chosen["方案"].split(" × ")[0] if chosen else None,
        "time_unit": chosen["方案"].split(" × ")[1] if chosen else None,
        "scheme": chosen["方案"] if chosen else None,
        "exact": bool(chosen["精確"]) if chosen else None,
        "needs_geocoding": (not chosen["精確"]) if chosen else None,
        "note": chosen["說明"] if chosen else "四種方案全部不過，見 relaxed 區塊",
    },
}

path = PROC / "density_decision.json"
path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"已寫出 {path}")
print(json.dumps(decision["decision"], ensure_ascii=False, indent=2))
''')

code(r'''
def pct(x) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.1%}"


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

city_tbl = (by_city.pivot(index="方案", columns="範圍", values="ge5_share")
                   .reindex(index=ORDER, columns=CITY_ORDER)
                   .apply(lambda col: col.map(lambda x: f"{x:.1%}")).reset_index())

ladder_tbl = pd.DataFrame([{
    "順位": i + 1, "方案": v["方案"], "n≥5 佔比": pct(v["ge5_share"]),
    "判定": v["判定"], "說明": v["說明"],
} for i, v in enumerate(ladder)])

relaxed_tbl = pd.DataFrame({
    "方案": O.index,
    "嚴格（僅電梯大樓）": [pct(x) for x in O["ge5_share"]],
    "放寬（+華廈+公寓）": [pct(R.loc[s, "ge5_share"]) for s in O.index],
})

report = f"""# NB0 密度分析報告

- 產生時間：{decision['generated_at']}
- 規格來源：`taiwan-housets-mvp-plan_3.md` §5.0
- 分析期間：{DATE_START:%Y-%m} ~ {DATE_END:%Y-%m}（{N_PERIODS['月']} 個月 / {N_PERIODS['季']} 季）
- 建物型態：{' + '.join(BUILDING_MAIN)}
- 決策門檻：n ≥ {MIN_N} 的 cell 佔比 > {PASS_SHARE:.0%}（分母含零交易 cell）

## 結論

**{decision['decision']['scheme'] or '四種方案全部不過關'}**

{decision['decision']['note']}

{'⚠️ 此結論尚需 geocoding 實測確認（實價登錄無村里欄位）。' if decision['decision'].get('needs_geocoding') else ''}

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

## 各縣市 n≥{MIN_N} 佔比

{md_table(city_tbl)}

## 決策階梯

{md_table(ladder_tbl)}

## 敏感度：放寬建物型態（決策樹第 4 分支）

放寬後樣本 {len(lvr_relaxed):,} 筆（{len(lvr_relaxed) / max(len(lvr_m), 1):.2f}× 於嚴格版）。
第一個過關的方案：**{relaxed_chosen or '無'}**

{md_table(relaxed_tbl)}

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
2. 若決策為村里粒度 → 需先補 geocoding（TGOS 門牌坐標 或 國土測繪中心村里界圖）才能定案
3. 若決策為鄉鎮市區粒度 → 直接進 **NB1**：上傳鄉鎮市區邊界為 GEE Asset，跑 Sentinel-2 遙測指數
4. `data/processed/lvr_clean.parquet` 已備妥，NB2 直接讀取，不需重跑清整
""")

# ============================== 組裝 ==============================
nb = {
    "cells": [
        {"cell_type": t, "metadata": {},
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

out = pathlib.Path(__file__).resolve().parent / "00_density_check.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"wrote {out}  cells={len(CELLS)}")
