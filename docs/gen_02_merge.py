# -*- coding: utf-8 -*-
"""產生 notebooks/02_data_merge.ipynb"""
import hashlib, json, pathlib

CELLS = []
def md(s):   CELLS.append(("markdown", s.strip("\n")))
def code(s): CELLS.append(("code", s.strip("\n")))

md(r"""
# NB2 — 資料整合

規格：`taiwan-housets-mvp-plan_3.md` 計畫 §5.2。**粒度從 NB0 的
`density_decision.json` 讀，不要硬編。**

**輸入**：`lvr_clean.parquet`（NB0）、`unit_remote_sensing.csv`（NB1）、
POI（OSM Overpass）、所得（財政部）、戶數（戶政司，NB0 已快取）、單元邊界（NB1 已快取）
**輸出**：`data/processed/modeling_table.parquet`

## 與計畫的三處差異，以及它們的**偏誤方向**

實驗要回答的是「影像特徵加上去有沒有增量」。所以**任何被省略的、真實有預測力的
非影像特徵，都會讓影像看起來更好**——偏誤方向對我們不利（會產生假陽性）。
下面三項都朝這個方向，必須明講，並在 NB3 用 robustness check 補。

| 計畫要的 | 實際做的 | 為什麼 | 偏誤方向 |
|---|---|---|---|
| 捷運「通車前後」二元特徵 | 只有靜態的站點數與距離 | OSM 車站的 `start_date` 覆蓋率只有 **11%** | **偏袒影像**——新站周邊的新建案正好是 ΔNDBI 會抓到的 |
| 土地使用分區的**面積**佔比 | 交易的 `都市土地使用分區` 組成 | 面積佔比要另一份 shapefile；交易組成是現成的代理 | **偏袒影像**——建成區面積佔比跟 NDBI 量的是同一件事 |
| 500m/1km 半徑內 POI 數 | 單元**內**的 POI 數與密度 ＋ 重心到最近設施距離 | 半徑圈是為村里設計的；鄉鎮市區平均 30 km²，半徑圈沒有意義 | 中性 |

**NB3 必須做的 robustness check**（因為上面前兩項）：
只留「窗口內沒有新捷運站通車」的單元重跑一次，看影像的增益是否還在。
若增益只存在於有新站的單元，那增益很可能是被省略的交通特徵。
""")

md(r"""
---
## 0. 環境與上游產出
""")

code(r"""
# Colab 執行時取消註解
# !pip install -q pandas numpy pyarrow geopandas shapely requests
""")

code(r'''
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.width", 180)
pd.set_option("display.max_columns", 80)

try:
    import google.colab  # noqa: F401
    from google.colab import drive

    drive.mount("/content/drive", force_remount=False)
    PROJECT = Path("/content/drive/MyDrive/TW_Housets")
except ImportError:
    _cwd = Path.cwd()
    PROJECT = _cwd.parent if _cwd.name == "notebooks" else _cwd

RAW, PROC = PROJECT / "data" / "raw", PROJECT / "data" / "processed"
POI_DIR = RAW / "poi"
POI_DIR.mkdir(parents=True, exist_ok=True)

# ---- 上游產出：缺一個就停，不要用預設值硬跑 ----
NEED = {"NB0 決策": PROC / "density_decision.json",
        "NB0 清整後實價登錄": PROC / "lvr_clean.parquet",
        "NB1 遙測": PROC / "unit_remote_sensing.csv",
        "NB1 中介資料": PROC / "rs_meta.json"}
_missing = {k: v for k, v in NEED.items() if not v.exists()}
if _missing:
    raise FileNotFoundError(
        "缺少上游產出：\n" + "\n".join(f"  {k}：{v}" for k, v in _missing.items())
        + "\n先把對應的 notebook 跑完。")

DEC = json.loads(NEED["NB0 決策"].read_text(encoding="utf-8"))
RSM = json.loads(NEED["NB1 中介資料"].read_text(encoding="utf-8"))
DECISION = DEC["decision"]
SPATIAL_UNIT, TIME_UNIT = DECISION["spatial_unit"], DECISION["time_unit"]
if DECISION["blocked_by_denominator_conflict"] or not DECISION["scheme"]:
    raise RuntimeError("NB0 的粒度決策未定案，先處理再跑 NB2。")
if RSM["spatial_unit"] != SPATIAL_UNIT:
    raise RuntimeError(f"NB1 用的空間單元（{RSM['spatial_unit']}）"
                       f"與 NB0 的決策（{SPATIAL_UNIT}）不一致。")
if not RSM["checks"]["passed"]:
    print("⚠️  NB1 的合理性檢查沒有全部通過——先看 rs_report.md 再決定要不要繼續。")

CITY_ORDER = ["臺北市", "新北市", "桃園市", "臺中市", "臺南市", "高雄市"]
DATE_START, DATE_END = pd.Timestamp(DEC["window"]["start"]), pd.Timestamp(DEC["window"]["end"])
print(f"粒度：{DECISION['scheme']}｜窗口 {DATE_START:%Y-%m} ~ {DATE_END:%Y-%m}")
print(f"影像頻率：{RSM['imagery_freq']}｜結構性重複：{RSM['structural_repeat_with_price']}")
print(f"NB1 提醒：{RSM['usable_window_note']}")
''')

md(r"""
---
## 1. Y：房價聚合

計畫 §5.2 步驟 1：按 NB0 決定的粒度聚合**中位單價**。中位數而非平均，因為實價登錄
的尾端很厚（豪宅、法拍殘留）。同時帶出 `n`（該 cell 的交易筆數）——NB0 已經證明
`n = 1` 的 cell 的「中位數」就是那一筆本身，沒有平滑效果。
""")

code(r'''
lvr = pd.read_parquet(NEED["NB0 清整後實價登錄"])
print(f"讀入 {len(lvr):,} 筆清整後交易，欄位 {len(lvr.columns)} 個")

lvr["交易日"] = pd.to_datetime(lvr["交易日"])
PERIOD_FREQ = "M" if TIME_UNIT == "月" else "Q"
lvr["period"] = lvr["交易日"].dt.to_period(PERIOD_FREQ)
lvr["quarter"] = lvr["交易日"].dt.to_period("Q")
UNIT_COL = "district" if SPATIAL_UNIT == "鄉鎮市區" else "village"
if UNIT_COL not in lvr.columns:
    raise RuntimeError(
        f"lvr_clean 沒有 {UNIT_COL} 欄。村里粒度需要地址 geocoding（計畫 §5.2），"
        "那是這本 notebook 還沒實作的部分。")
lvr = lvr.rename(columns={UNIT_COL: "unit"})

# 建物屬性（計畫 §5.3 特徵集 A）。屋齡用交易日減建築完成年月。
_bd = pd.to_numeric(lvr["建築完成年月"], errors="coerce")
_by = np.where(_bd > 1000, _bd // 10000 + 1911, np.nan)
lvr["屋齡"] = lvr["交易日"].dt.year - _by
lvr.loc[(lvr["屋齡"] < 0) | (lvr["屋齡"] > 100), "屋齡"] = np.nan
# **樓層是中文數字。** 實測 `移轉層次` 的值是「四層」「十一層」、`總樓層數` 是
# 「十五層」——用 `str.extract(r"(\d+)")` 抓阿拉伯數字會得到**全部 NaN**，
# 而 NaN 欄位在樹模型裡不會報錯、只是完全沒有資訊。
# （這個 bug 是靠「某個特徵 100% NaN」這個下游症狀才被發現的。）
_CN = {"零": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
       "六": 6, "七": 7, "八": 8, "九": 9}


def cn_floor(v) -> float:
    """「十五層」→ 15。地下層／全／見其他登記事項 等回 NaN。"""
    t = str(v).strip()
    if not t or t == "nan" or "地下" in t or "全" in t or "其他" in t:
        return np.nan
    t = t.replace("層", "").replace("　", "").strip()
    if t.isdigit():
        return float(t)
    total, cur = 0, 0
    for ch in t:
        if ch in _CN:
            cur = cur * 10 + _CN[ch] if cur and ch == "零" else _CN[ch]
        elif ch == "十":
            cur = (cur or 1) * 10
            total, cur = total + cur, 0
        elif ch == "百":
            cur = (cur or 1) * 100
            total, cur = total + cur, 0
        else:
            return np.nan
    return float(total + cur) or np.nan


_lv = lvr["移轉層次"].map(cn_floor)
_tt = lvr["總樓層數"].map(cn_floor)
print(f"樓層解析率：移轉層次 {_lv.notna().mean():.1%}、總樓層數 {_tt.notna().mean():.1%}")
if _lv.notna().mean() < 0.8 or _tt.notna().mean() < 0.8:
    print("⚠️  樓層解析率偏低——來源的寫法可能又變了，檢查 cn_floor。"
          "（這個欄位若全是 NaN，模型不會報錯，只是白佔一個特徵。）")
lvr["樓層比"] = _lv / _tt.replace(0, np.nan)
lvr.loc[(lvr["樓層比"] <= 0) | (lvr["樓層比"] > 1.2), "樓層比"] = np.nan
lvr["總樓層"] = _tt

agg = (lvr.groupby(["unit", "period"], observed=True)
          .agg(price_median=("單價元每坪", "median"),
               price_p25=("單價元每坪", lambda s: s.quantile(0.25)),
               price_p75=("單價元每坪", lambda s: s.quantile(0.75)),
               n_tx=("單價元每坪", "size"),
               area_median=("房屋面積坪", "median"),
               age_median=("屋齡", "median"),
               floor_ratio_median=("樓層比", "median"),
               total_floor_median=("總樓層", "median"),
               parking_unsplit_share=("車位價未拆", "mean"))
          .reset_index())
agg["quarter"] = agg["period"].dt.asfreq("Q")
agg["縣市"] = agg["unit"].str[:3]
print(f"\n聚合後 {len(agg):,} 個有交易的 cell"
      f"（單元 {agg.unit.nunique()} × 期 {agg.period.nunique()}）")
print(f"  n_tx 分布：median {agg.n_tx.median():.0f}、"
      f"n=1 佔 {(agg.n_tx == 1).mean():.1%}、n≥3 佔 {(agg.n_tx >= 3).mean():.1%}、"
      f"n≥5 佔 {(agg.n_tx >= 5).mean():.1%}")
print(agg.groupby("縣市")["price_median"].median().reindex(CITY_ORDER).round(0).to_string())
''')

code(r'''
# ---- 土地使用分區組成（代理，見開頭的偏誤說明）----
_zone = lvr["都市土地使用分區"].fillna("(空白)").astype(str).str.strip()
lvr["zone_grp"] = np.select(
    [_zone.str.contains("住"), _zone.str.contains("商"),
     _zone.str.contains("工"), _zone.str.contains("農")],
    ["住", "商", "工", "農"], default="其他")
# **只用訓練期的交易算組成。** 用全期算的話，這個「時間不變」的特徵裡就含有
# 測試期交易的資訊（實測單元內唯一值數 = 1，也就是整段測試期共用同一個值）。
# 邊界跟 NB3 的切分對齊；NB3 若改切分，這裡要一起改。
ZONE_TRAIN_END = pd.Period("2022Q4", freq="Q")
_ztrain = lvr[lvr["quarter"] <= ZONE_TRAIN_END]
print(f"土地使用分區組成只用 {ZONE_TRAIN_END} 之前的 {len(_ztrain):,} 筆"
      f"（全期 {len(lvr):,} 筆）——避免把測試期交易的組成洩漏進特徵")
zone = (_ztrain.pivot_table(index="unit", columns="zone_grp", values="編號",
                            aggfunc="size", fill_value=0))
zone = zone.div(zone.sum(axis=1).replace(0, np.nan), axis=0)
zone.columns = [f"zone_share_{c}" for c in zone.columns]
zone = zone.reset_index()
print("土地使用分區組成（交易加權，全六都平均）")
print(zone.set_index("unit").mean().round(3).to_string())
''')

md(r"""
---
## 2. 遙測 join

影像是**季**、房價可能是**月**。若是月，每季 3 個月共用同一組遙測值——這是計畫 §5.3
的結構性重複問題。這裡不做任何平滑或內插（那會把重複藏起來），而是照實 join 並
**驗證同一季內的遙測值真的完全相同**，然後把這個事實寫進產出的中介資料，讓 NB3
一定看得到。
""")

code(r'''
rs = pd.read_csv(NEED["NB1 遙測"])
rs["quarter"] = pd.PeriodIndex(rs["quarter"], freq="Q")
IDX_COLS = ["NDVI", "NDBI", "MNDWI", "dNDVI", "dNDBI"]
print(f"讀入遙測 {len(rs):,} 列（單元 {rs.unit.nunique()} × 季 {rs.quarter.nunique()}）")

VPR_FLOOR = 0.30            # 計畫 §5.2 步驟 8
rs["rs_usable"] = rs["valid_pixel_ratio"] >= VPR_FLOOR
print(f"valid_pixel_ratio ≥ {VPR_FLOOR} 的比例：{rs.rs_usable.mean():.1%}")

df = agg.merge(rs[["unit", "quarter", "valid_pixel_ratio", "n_solar_day",
                   "rs_usable"] + IDX_COLS],
               on=["unit", "quarter"], how="left", validate="many_to_one")
_no_rs = df["NDVI"].isna() & df["valid_pixel_ratio"].isna()
print(f"join 不到遙測的 cell：{int(_no_rs.sum()):,}（{_no_rs.mean():.2%}）")
if _no_rs.any():
    print("  那些 (unit, quarter)：",
          df.loc[_no_rs, ["unit", "quarter"]].drop_duplicates().head(8).to_dict("records"))
''')

md(r"""
---
## 3. POI（OpenStreetMap Overpass）

**一定要用 `nwr` 而不是 `node`。** 台灣的 OSM 把學校、公園、醫院、大型商場幾乎都畫成
多邊形（way / relation）。實測臺北市 bbox：只查 node 得到 **4 個**學校，
改 `nwr` 得到 **439 個**——差 100 倍，而且不會有任何錯誤訊息。
""")

code(r'''
import geopandas as gpd
from shapely.geometry import Point

BND = next(iter(sorted((RAW / "boundary").rglob("*.shp"))), None)
if BND is None:
    raise FileNotFoundError("找不到單元邊界 shapefile（NB1 §1 會下載並快取到 "
                            "data/raw/boundary/）。先跑 NB1。")
_g = gpd.read_file(BND).to_crs(4326)
_g = _g[_g["COUNTYNAME"].isin(CITY_ORDER)].copy()
_g["unit"] = _g["COUNTYNAME"] + _g["TOWNNAME"]
units = _g[["unit", "COUNTYNAME", "geometry"]].reset_index(drop=True)
units["area_km2"] = units.to_crs("EPSG:3826").area / 1e6
units["centroid"] = units.to_crs("EPSG:3826").geometry.centroid
print(f"單元邊界 {len(units)} 個")

# Overpass 會回 504 / 429（尤其是大 bbox 或公用實例忙碌時）。實測抓新北市就中過
# 504。多備幾個鏡像 + 指數退避；不做這件事的話 NB2 會在半路死掉，而前面幾個縣市
# 的快取又還在，重跑會很混亂。
OVERPASS_ENDPOINTS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
)
OVERPASS_TRIES = 3
POI_TAGS = {
    "station": 'nwr["railway"="station"]',
    "school": 'nwr["amenity"~"^(school|university)$"]',
    "hospital": 'nwr["amenity"~"^(hospital|clinic)$"]',
    "retail": 'nwr["shop"~"^(supermarket|mall|department_store)$"]',
    "convenience": 'nwr["shop"="convenience"]',
    "park": 'nwr["leisure"="park"]',
}


def fetch_poi(city: str, bbox: tuple) -> pd.DataFrame:
    """逐縣市抓一次並快取。bbox = (south, west, north, east)。"""
    cache = POI_DIR / f"{city}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    bb = ",".join(f"{v:.4f}" for v in bbox)
    body = "".join(f"  {sel}({bb});\n" for sel in POI_TAGS.values())
    q = f"[out:json][timeout:180];\n(\n{body});\nout tags center;"
    t0, els, last = time.time(), None, None
    for attempt in range(OVERPASS_TRIES):
        for ep in OVERPASS_ENDPOINTS:
            try:
                req = urllib.request.Request(
                    ep, data=urllib.parse.urlencode({"data": q}).encode(),
                    headers={"User-Agent": "TW_Housets/0.1 (research)"})
                with urllib.request.urlopen(req, timeout=300) as r:
                    els = json.loads(r.read())["elements"]
                break
            except Exception as exc:                     # noqa: BLE001
                last = f"{ep.split('/')[2]}: {type(exc).__name__} {exc}"
                print(f"    {city} ← {last}")
        if els is not None:
            break
        time.sleep(10 * (attempt + 1))
    if els is None:
        raise RuntimeError(
            f"{city} 的 POI 抓不到（試了 {OVERPASS_TRIES} 輪 × "
            f"{len(OVERPASS_ENDPOINTS)} 個端點）。最後一次：{last}\n"
            "    Overpass 是公用服務，忙碌時會回 504/429。稍後重跑本 cell 即可"
            "（已抓到的縣市有快取，不會重抓）。")
    rows = []
    for e in els:
        t = e.get("tags", {})
        lat = e.get("lat", (e.get("center") or {}).get("lat"))
        lon = e.get("lon", (e.get("center") or {}).get("lon"))
        if lat is None or lon is None:
            continue
        if t.get("railway") == "station":
            kind = "station"
        elif t.get("amenity") in ("school", "university"):
            kind = "school"
        elif t.get("amenity") in ("hospital", "clinic"):
            kind = "hospital"
        elif t.get("shop") in ("supermarket", "mall", "department_store"):
            kind = "retail"
        elif t.get("shop") == "convenience":
            kind = "convenience"
        elif t.get("leisure") == "park":
            kind = "park"
        else:
            continue
        rows.append({"kind": kind, "lat": lat, "lon": lon,
                     "start_date": t.get("start_date")})
    d = pd.DataFrame(rows)
    d.to_parquet(cache, index=False)
    print(f"  {city}：{len(els):,} 元素 → {len(d):,} 個 POI，{time.time() - t0:.0f}s")
    return d


poi_parts = []
for city in CITY_ORDER:
    sub = units[units["COUNTYNAME"] == city]
    minx, miny, maxx, maxy = sub.total_bounds
    pad = 0.02
    poi_parts.append(fetch_poi(city, (miny - pad, minx - pad, maxy + pad, maxx + pad))
                     .assign(縣市=city))
poi = pd.concat(poi_parts, ignore_index=True)
print(f"\nPOI 合計 {len(poi):,} 個")
print(poi.groupby("kind").size().sort_values(ascending=False).to_string())
_sd = poi[poi["kind"] == "station"]["start_date"].notna().mean()
print(f"\n車站有 start_date 的比例：{_sd:.0%}"
      f" → 通車前後特徵做不出來（見開頭的偏誤說明）")
''')

code(r'''
# ---- 單元內的 POI 數與密度、重心到最近設施的距離 ----
gp = gpd.GeoDataFrame(poi, geometry=[Point(xy) for xy in zip(poi.lon, poi.lat)],
                      crs=4326).to_crs("EPSG:3826")
gu = units[["unit", "area_km2", "geometry"]].to_crs("EPSG:3826")

joined = gpd.sjoin(gp, gu, how="left", predicate="within")
cnt = (joined.dropna(subset=["unit"])
             .pivot_table(index="unit", columns="kind", values="lat",
                          aggfunc="size", fill_value=0))
cnt.columns = [f"n_{c}" for c in cnt.columns]
cnt = cnt.reindex(units["unit"]).fillna(0)

feat = units[["unit", "area_km2"]].set_index("unit").join(cnt)
for c in [c for c in feat.columns if c.startswith("n_")]:
    feat[c.replace("n_", "dens_")] = feat[c] / feat["area_km2"]

# 重心到最近站／學校／公園的距離（公尺）
cent = gpd.GeoDataFrame(units[["unit"]], geometry=units["centroid"], crs="EPSG:3826")
for kind in ("station", "school", "park"):
    tgt = gp[gp["kind"] == kind]
    if len(tgt):
        near = gpd.sjoin_nearest(cent, tgt[["geometry"]], how="left",
                                 distance_col=f"dist_{kind}")
        feat[f"dist_{kind}"] = (near.groupby("unit")[f"dist_{kind}"].min()
                                    .reindex(feat.index))
    else:
        feat[f"dist_{kind}"] = np.nan
poi_feat = feat.reset_index()
print(f"POI 特徵 {poi_feat.shape[1] - 1} 個 × {len(poi_feat)} 單元")
print(poi_feat.set_index("unit").describe().T[["mean", "50%", "max"]].round(2).to_string())
''')

md(r"""
---
## 4. 社經：所得、戶數、密度

所得來源：財政部財政資訊中心「綜稅綜合所得總額全國各縣市鄉鎮村里統計分析表」。
URL 有規律：`https://www.fia.gov.tw/WEB/fia/ias/ias{ROC}/{ROC}_165-9.csv`。

實測**民國 101–112 都在、113 還沒發布**。所以 2024 年的所得用 2023 往後遞延，
並帶一個 `income_year_used` 欄位——**不要靜默遞延**，NB3 要能對它做敏感度測試。
""")

code(r'''
INCOME_URL = "https://www.fia.gov.tw/WEB/fia/ias/ias{y}/{y}_165-9.csv"
# **所得必須 lag。** 財政部的年度所得約落後兩年才發布（實測 2026-08 時民國 112 有、
# 113 還沒有）。直接用「同年」的所得等於預測 2023 年 1 月時就知道 2023 全年的所得
# ——那是前視資訊，而 M4 的名稱黑名單與「相關 > 0.98」都抓不到
# （實測 income_mean 與 Y 的相關只有 0.62）。
# 所以要抓的年份是「窗口年份各往前推 LAG 年」，不是窗口年份本身。
INCOME_LAG_YEARS = 2
years_needed = list(range(DATE_START.year - INCOME_LAG_YEARS,
                          DATE_END.year - INCOME_LAG_YEARS + 1))
print(f"所得 lag {INCOME_LAG_YEARS} 年 → 需要 {years_needed[0]}–{years_needed[-1]} 年的資料"
      f"（對應窗口 {DATE_START.year}–{DATE_END.year}）")


def fetch_income(west_year: int) -> pd.DataFrame:
    roc = west_year - 1911
    cache = RAW / f"income_{roc}.csv"
    if not cache.exists():
        url = INCOME_URL.format(y=roc)
        try:
            with urllib.request.urlopen(
                    urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}),
                    timeout=120) as r:
                cache.write_bytes(r.read())
        except Exception as exc:                     # noqa: BLE001
            print(f"  {west_year}（民國{roc}）抓不到：{type(exc).__name__}")
            return pd.DataFrame()
    d = pd.read_csv(cache, encoding="utf-8-sig", dtype=str)
    # 這個檔的第一欄名帶 BOM（實測 utf-8-sig 之後仍是 '\ufeff縣市別'），而且欄名
    # 可能帶空白。**用「包含」比對而不是等號**——寫死欄名的話會 KeyError，
    # 而錯誤訊息（KeyError: 'unit'）指的方向完全不對。
    d.columns = [str(c).replace("\ufeff", "").strip() for c in d.columns]
    # **欄名在不同年份不一樣。** 實測民國 106 的第一欄叫「鄉鎮市區」、民國 111 叫
    # 「縣市別」，但兩者的**值都是「臺北市松山區」這種縣市+區串好的字串**。
    # 所以第一欄一律當 unit（並比對值長得對不對），其餘用「包含」比對。
    _colmap = {}
    for c in d.columns:
        for want, key in (("納稅單位", "n_taxunit_raw"), ("綜合所得總額", "total"),
                          ("中位數", "median")):
            if want in c and key not in _colmap.values():
                _colmap[c] = key
                break
    _first = d.columns[0]
    if _first not in _colmap:
        _colmap[_first] = "unit"
    d = d.rename(columns=_colmap)
    if "unit" in d.columns:
        _looks = d["unit"].astype(str).str.match(r"^..[市縣].+[區鄉鎮市]$").mean()
        if _looks < 0.8:
            raise RuntimeError(
                f"所得檔 {cache.name} 的第一欄不像「縣市+鄉鎮市區」："
                f"{d['unit'].head(3).tolist()}（符合樣式的只有 {_looks:.0%}）。\n"
                f"    實際欄名 {list(d.columns)}——來源版面改了。")
    _need = {"unit", "n_taxunit_raw", "total", "median"}
    if not _need <= set(d.columns):
        raise RuntimeError(
            f"所得檔 {cache.name} 的欄名對不上：實際欄名 {list(d.columns)}\n"
            f"    缺 {sorted(_need - set(d.columns))}。來源改版了就更新上面的對照。")
    for c in ("n_taxunit_raw", "total", "median"):
        d[c] = pd.to_numeric(d[c].astype(str).str.replace(",", ""), errors="coerce")
    # 聚合到鄉鎮市區。income_mean 是**精確**的（總額 / 納稅單位）；
    # income_median_w 只是村里中位數的戶數加權平均，是近似值。
    d["_w"] = d["n_taxunit_raw"].fillna(0)
    d["_wm"] = d["_w"] * d["median"].fillna(0)
    g = (d.groupby("unit", as_index=False)
          .agg(_total=("total", "sum"), _w=("_w", "sum"), _wm=("_wm", "sum")))
    g["income_mean"] = g["_total"] / g["_w"].replace(0, np.nan)
    g["income_median_w"] = g["_wm"] / g["_w"].replace(0, np.nan)
    g["n_taxunit"] = g["_w"]
    g = g.drop(columns=["_total", "_w", "_wm"])
    g["year"] = west_year
    return g


inc_parts = [fetch_income(y) for y in years_needed]
inc = pd.concat([x for x in inc_parts if len(x)], ignore_index=True)
inc["income_year_used"] = inc["year"]      # 先設好，沒有遞延時也要存在
have_years = sorted(inc["year"].unique())
print(f"所得年份：{have_years}")
missing_years = [y for y in years_needed if y not in have_years]
if missing_years:
    latest = max(have_years)
    print(f"⚠️  {missing_years} 沒有資料 → 用 {latest} 年往後遞延"
          f"（會記在 income_year_used 欄，NB3 要對它做敏感度測試）")
    for y in missing_years:
        inc = pd.concat([inc, inc[inc["year"] == latest].assign(year=y)],
                        ignore_index=True)
        inc.loc[inc["year"] == y, "income_year_used"] = latest
inc["income_year_used"] = inc["income_year_used"].fillna(inc["year"]).astype(int)
print(f"所得表 {len(inc):,} 列（單元 × 年）"
      f"｜遞延的年份 {sorted(set(inc.loc[inc.income_year_used != inc.year, 'year'])) or '無'}")

inc_lagged = inc.copy()
inc_lagged["year"] = inc_lagged["year"] + INCOME_LAG_YEARS
inc_lagged = inc_lagged.rename(columns={"income_year_used": "income_source_year"})
print(f"所得往後推 {INCOME_LAG_YEARS} 年當特徵："
      f"預測第 Y 年用第 Y−{INCOME_LAG_YEARS} 年的所得（發布時點才拿得到）")

# ---- 戶數（戶政司，NB0 已快取）----
_vp = next(iter(sorted(RAW.glob("villages_*.parquet"))), None)
if _vp is None:
    raise FileNotFoundError("找不到 villages_*.parquet（NB0 會抓並快取）。先跑 NB0。")
hh = (pd.read_parquet(_vp).groupby("district", as_index=False)["戶數"].sum()
        .rename(columns={"district": "unit", "戶數": "n_household"}))
print(f"戶數表 {len(hh)} 個單元（靜態快照——時間變異這裡沒有，是已知限制）")
''')

md(r"""
---
## 5. 合併與篩選
""")

code(r'''
df["year"] = df["period"].dt.year
out = (df.merge(poi_feat, on="unit", how="left", validate="many_to_one")
         .merge(zone, on="unit", how="left", validate="many_to_one")
         .merge(hh, on="unit", how="left", validate="many_to_one")
         .merge(inc_lagged[["unit", "year", "income_mean", "income_median_w",
                            "income_source_year"]],
                on=["unit", "year"], how="left", validate="many_to_one"))
out["hh_density"] = out["n_household"] / out["area_km2"]

# 所得有一個**全體同步的水準斷點**（實測 2018→2019 中位數 −23.8%、92% 的單元同步
# 下跌，追到原始 CSV 確認斷點在來源本身），而且測試期的絕對值會落在訓練期範圍之外
# → 樹模型只能把它塞進末端葉。加一個**單元內相對值**，讓測試期的值落回訓練期範圍。
_um = out.groupby("unit")["income_mean"].transform("median")
out["income_rel"] = out["income_mean"] / _um.replace(0, np.nan)

# ---- 時間特徵（計畫 §5.2 步驟 7）----
out["t_index"] = (out["period"] - out["period"].min()).apply(lambda x: x.n)
out["q_num"] = out["quarter"].dt.quarter
out["q_sin"] = np.sin(2 * np.pi * out["q_num"] / 4)
out["q_cos"] = np.cos(2 * np.pi * out["q_num"] / 4)
if TIME_UNIT == "月":
    out["m_num"] = out["period"].dt.month
    out["m_sin"] = np.sin(2 * np.pi * out["m_num"] / 12)
    out["m_cos"] = np.cos(2 * np.pi * out["m_num"] / 12)

# ---- 篩選（計畫 §5.2 步驟 8、9）----
MIN_N_TX = 3
before = len(out)
out["keep"] = (out["n_tx"] >= MIN_N_TX) & out["rs_usable"].fillna(False)
print(f"篩選（n_tx ≥ {MIN_N_TX} 且 valid_pixel_ratio ≥ {VPR_FLOOR}）："
      f"{int(out.keep.sum()):,} / {before:,}（{out.keep.mean():.1%}）")
print(f"  只因 n_tx 被剔除：{int((~(out.n_tx >= MIN_N_TX)).sum()):,}")
print(f"  只因遙測不可用被剔除：{int((~out.rs_usable.fillna(False)).sum()):,}")
print(f"\n保留的期間：{out.loc[out.keep, 'period'].min()} ~ {out.loc[out.keep, 'period'].max()}")
print(out[out.keep].groupby("縣市").size().reindex(CITY_ORDER).to_string())
''')

md(r"""
---
## 6. 合理性檢查

跟 NB0／NB1 同一個立場：合併出錯時**不會有自然的失敗訊號**。join 少一批單元、
特徵欄位悄悄全是 NaN、影像值在同一季內不一致——模型照樣訓練得起來、MAPE 照樣算得出來。

| 層 | 檢查 | 抓什麼 | 會中止 |
|---|---|---|---|
| **M1** | 特徵沒有意外的 NaN | join 鍵對不上（單元名、年份） | ✅ |
| **M2** | 影像值在同一季內完全相同 | 結構性重複被誤處理（內插、平滑） | ✅ |
| **M3** | Y 的值域與城市排序 | 聚合單位錯、單價欄位錯 | ✅ |
| **M4** | 沒有目標洩漏 | 特徵裡混進由價格算出來的東西 | ✅ |
| **M5** | **跨來源相關** | 所得 vs 房價應正相關（獨立來源） | ✅ |
""")

code(r'''
MERGE_STRICT = True
mchecks: dict = {}


def merge_fail(title: str, detail: str) -> None:
    msg = f"{title}\n{detail}"
    if MERGE_STRICT:
        raise RuntimeError(msg + "\n\n確認可接受再把 MERGE_STRICT 設為 False。")
    print(f"{msg}\n\n（MERGE_STRICT = False，僅警告）\n")


k = out[out.keep].copy()
# **絕對不能進特徵集的東西**：
#   · n_tx / price_*：由 Y 那批交易算出來的
#   · n_solar_day / valid_pixel_ratio：**Sentinel-2 的產物**。放進 A 的話「無影像
#     基準」就用到了衛星資料（而且它是雲量與取像密度的代理，帶季節與地形資訊）。
#     實測把它從 A 拿掉，ΔMAPE 動 0.22 pp。只當品質篩選用。
NEVER_FEATURE = {"n_tx", "n_solar_day", "valid_pixel_ratio", "n_pixel",
                 "n_household", "n_taxunit", "keep", "year"}

# 特徵集分三層，因為「A 怎麼定義」造成的 ΔMAPE 分散實測比 2 pp 的成功門檻還大。
#   A0：**純外生**——完全不含由「當期那批交易」算出來的東西
#   A1：A0 + 當期交易的建物屬性組成（計畫 §5.3 字面要求的「建物屬性」，但那是同期資訊）
#   A2：A1 + 土地使用分區組成（已限定只用訓練期交易，見 §1）
_poi = [c for c in k.columns
        if c.startswith(("dens_", "dist_")) or
        (c.startswith("n_") and c not in NEVER_FEATURE)]
FEATS_A0 = ([c for c in _poi if c not in NEVER_FEATURE]
            + ["area_km2", "hh_density", "income_mean", "income_median_w",
               "income_rel", "t_index", "q_sin", "q_cos"]
            + (["m_sin", "m_cos"] if TIME_UNIT == "月" else []))
CONTEMP = ["area_median", "age_median", "floor_ratio_median", "total_floor_median"]
FEATS_A1 = FEATS_A0 + CONTEMP
FEATS_A2 = FEATS_A1 + [c for c in k.columns if c.startswith("zone_share_")]
IMG = ["NDVI", "NDBI", "MNDWI", "dNDVI", "dNDBI"]
_flt = lambda L: [c for c in dict.fromkeys(L)
                  if c in k.columns and c not in NEVER_FEATURE]
FEATS_A0, FEATS_A1, FEATS_A2 = _flt(FEATS_A0), _flt(FEATS_A1), _flt(FEATS_A2)
# 主特徵集 = A1（計畫 §5.3 的字面定義）；A0/A2 由 NB3 當敏感度層
FEATS_A, FEATS_B = FEATS_A1, _flt(FEATS_A1 + IMG)
print(f"特徵層：A0 純外生 {len(FEATS_A0)}｜A1 +同期建物屬性 {len(FEATS_A1)}"
      f"｜A2 +分區組成 {len(FEATS_A2)}｜影像 {len(_flt(IMG))}")

# ---- M1：特徵的 NaN ----
nan_share = k[FEATS_B].isna().mean().sort_values(ascending=False)
# dNDVI/dNDBI 的第一年本來就是 NaN（需要去年同季），那是預期的
EXPECTED_NAN = {"dNDVI", "dNDBI", "age_median", "floor_ratio_median"}
bad_nan = nan_share[(nan_share > 0.02) & (~nan_share.index.isin(EXPECTED_NAN))]
mchecks["M1"] = {"nan_share": {c: round(float(v), 4) for c, v in nan_share.items() if v > 0}}
print("特徵的 NaN 佔比（只列 > 0）")
print(nan_share[nan_share > 0].round(4).to_string() or "  （全部無 NaN）")
if len(bad_nan):
    merge_fail(f"❌ M1：{len(bad_nan)} 個特徵的 NaN 超過 2%：{list(bad_nan.index)}",
               "    多半是 join 鍵對不上（單元名用字、年份範圍）。\n"
               "    逐一檢查：那個特徵的來源表裡有沒有這些 unit / year。")
else:
    print("✅ M1：沒有意外的 NaN")
''')

code(r'''
# ---- M2：影像值在同一季內完全相同 ----
# 若房價是月粒度，同一 (unit, quarter) 的 3 個月必須共用同一組遙測值。
# 不相同就代表有人做了內插或平滑——那會把結構性重複藏起來，
# 讓 NB3 誤以為遙測特徵有 96 個獨立觀測（其實只有 32 個）。
if RSM["structural_repeat_with_price"]:
    g = k.groupby(["unit", "quarter"])[["NDVI", "NDBI", "MNDWI"]].nunique()
    bad = g[(g > 1).any(axis=1)]
    mchecks["M2"] = {"structural_repeat": True, "n_inconsistent": int(len(bad)),
                     "months_per_quarter": round(
                         float(k.groupby(["unit", "quarter"]).size().mean()), 2)}
    print(f"同一季內的月數：平均 {mchecks['M2']['months_per_quarter']}")
    if len(bad):
        merge_fail(f"❌ M2：{len(bad)} 個 (unit, quarter) 的遙測值在季內不一致",
                   f"    例：{bad.head(5).to_dict('index')}\n"
                   "    影像是季度合成，同一季內必須完全相同。不同就是被內插／平滑過了。")
    else:
        print("✅ M2：遙測值在每個季內完全相同（結構性重複如實保留）")
    print("   → NB3 的 paired t-test 必須以「單元 × 季」為 pairing 單位（計畫 §5.3）")
else:
    mchecks["M2"] = {"structural_repeat": False}
    print("✅ M2：房價與影像同為季度，1:1 對齊，沒有結構性重複")
''')

code(r'''
# ---- M3：Y 的值域與城市排序 ----
PRICE_MIN, PRICE_MAX = 50_000, 2_500_000        # 與 NB0 的清整規則一致
bad_price = k[(k["price_median"] < PRICE_MIN) | (k["price_median"] > PRICE_MAX)]
city_med = k.groupby("縣市")["price_median"].median().reindex(CITY_ORDER)
mchecks["M3"] = {"n_out_of_range": int(len(bad_price)),
                 "city_median": {c: round(float(v), 0) for c, v in city_med.items()}}
print("各縣市的 cell 中位單價（元/坪）")
print(city_med.round(0).to_string())
_rank_ok = city_med["臺北市"] == city_med.max()
if len(bad_price) or not _rank_ok:
    merge_fail(
        f"❌ M3：{len(bad_price)} 個 cell 的中位單價超出 [{PRICE_MIN:,}, {PRICE_MAX:,}]"
        f"，臺北市是否最高：{_rank_ok}",
        "    值域超出代表聚合用錯欄位（例如用了總價而非單價）。\n"
        "    臺北不是最高代表 unit → 縣市 的對應錯位。")
else:
    print(f"✅ M3：值域正常，且臺北市的中位單價最高（{city_med['臺北市']:,.0f} 元/坪）")
''')

code(r'''
# ---- M4：目標洩漏 ----
# 特徵集裡絕對不能出現由 price 算出來的東西。這裡用兩道：
#   1. 名稱黑名單（price_/n_tx 這類）
#   2. 與 Y 的相關係數 > 0.98 的欄位（幾乎等於 Y 本身）
LEAK_NAMES = [c for c in FEATS_B if c.startswith("price") or c in ("n_tx",)]
_corr = k[FEATS_B + ["price_median"]].corr()["price_median"].drop("price_median")
LEAK_CORR = [c for c, v in _corr.items() if abs(v) > 0.98]
mchecks["M4"] = {"leak_by_name": LEAK_NAMES, "leak_by_corr": LEAK_CORR,
                 "top_corr": {c: round(float(v), 3)
                              for c, v in _corr.abs().nlargest(6).items()}}
print("與 Y 相關性最高的特徵（絕對值）")
print(_corr.abs().nlargest(6).round(3).to_string())
# 第三道：**前視檢查**。名稱黑名單與相關門檻都抓不到「逐年特徵配逐月 Y」
# （同年 12 個月共用一個值，等於預測 1 月時就知道全年）。這裡直接量
# 「單元 × 年」內的唯一值數：若為 1 而 Y 的粒度是月，那個特徵就是年度常數。
YEARLY_OK = {"income_mean", "income_median_w", "income_rel"}   # 已 lag 過的
lookahead = []
if TIME_UNIT == "月":
    for c in FEATS_B:
        if k.groupby(["unit", "year"])[c].nunique(dropna=False).max() == 1 \
                and k.groupby("unit")[c].nunique(dropna=False).max() > 1:
            if c not in YEARLY_OK:
                lookahead.append(c)
mchecks["M4"]["yearly_constant_not_lagged"] = lookahead
print(f"年度常數但未 lag 的特徵：{lookahead or '（無）'}")
if LEAK_NAMES or LEAK_CORR or lookahead:
    merge_fail(
        f"❌ M4：疑似目標洩漏 名稱={LEAK_NAMES} 相關>0.98={LEAK_CORR} "
        f"年度常數未 lag={lookahead}",
        "    特徵集不能含由價格算出來的欄位，也不能含「預測時點還拿不到」的欄位。\n"
        "    逐年特徵要 lag 到發布時點真正可得的年份（見 §4 的 INCOME_LAG_YEARS）。")
else:
    print("✅ M4：沒有目標洩漏（最高相關 "
          f"{_corr.abs().max():.3f}，來自 {_corr.abs().idxmax()}）；"
          "逐年特徵都已 lag")
''')

code(r'''
# ---- M5：跨來源相關（所得 vs 房價）----
# 所得來自財政部、房價來自實價登錄，兩個獨立來源。相關性消失或反向代表
# unit 對位錯了——而各單元的房價與所得本身看起來都完全正常。
u = (k.groupby("unit")
      .agg(price=("price_median", "median"), income=("income_mean", "median"))
      .dropna())
rho = float(u.corr(method="spearman").iloc[0, 1])
mchecks["M5"] = {"n_units": len(u), "spearman_income_price": round(rho, 3)}
print(f"所得 vs 房價的 Spearman 相關（{len(u)} 個單元）：{rho:+.3f}")
if not (rho > 0.4):
    merge_fail(f"❌ M5：所得與房價的相關只有 {rho:+.3f}（應 > +0.4）",
               "    兩者是獨立來源，相關性消失代表 unit 對位錯位。\n"
               "    抽查：臺北市大安區的所得與房價應該都在前幾名。")
else:
    print("✅ M5：所得與房價正相關，unit 對位可信")
''')

md(r"""
---
## 7. 產出
""")

code(r'''
# **要匯出所有層的特徵，不只 FEATS_B。** 只匯出 FEATS_B 的話，A2 層的
# zone_share_* 不會進 parquet，而 NB3 的 TIERS 又是從 meta 讀名字再
# `if c in tbl.columns` 過濾——結果 A2 會靜默退化成跟 A1 一樣，
# 「三層敏感度」少了一層而沒有任何訊號。
_ALL_FEATS = list(dict.fromkeys(FEATS_A0 + FEATS_A1 + FEATS_A2 + _flt(IMG)))
KEEP_COLS = (["unit", "縣市", "period", "quarter", "year", "keep",
              "price_median", "price_p25", "price_p75", "n_tx",
              "valid_pixel_ratio", "n_solar_day", "income_source_year"]
             + [c for c in _ALL_FEATS if c != "t_index"] + ["t_index"])
KEEP_COLS = list(dict.fromkeys([c for c in KEEP_COLS if c in out.columns]))
tbl = out[KEEP_COLS].copy()
tbl["period"] = tbl["period"].astype(str)
tbl["quarter"] = tbl["quarter"].astype(str)
tbl.to_parquet(PROC / "modeling_table.parquet", index=False)
print(f"已寫出 {PROC / 'modeling_table.parquet'}"
      f"（{len(tbl):,} 列 × {tbl.shape[1]} 欄，其中 keep=True {int(tbl.keep.sum()):,} 列）")

mchecks["passed"] = bool(
    not [c for c in mchecks["M1"]["nan_share"]
         if mchecks["M1"]["nan_share"][c] > 0.02 and c not in EXPECTED_NAN]
    and mchecks["M2"].get("n_inconsistent", 0) == 0
    and mchecks["M3"]["n_out_of_range"] == 0
    and not mchecks["M4"]["leak_by_name"] and not mchecks["M4"]["leak_by_corr"]
    and mchecks["M5"]["spearman_income_price"] > 0.4)

meta = {
    "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    "scheme": DECISION["scheme"], "spatial_unit": SPATIAL_UNIT, "time_unit": TIME_UNIT,
    "n_rows": int(len(tbl)), "n_keep": int(tbl.keep.sum()),
    "n_units": int(tbl.unit.nunique()), "n_periods": int(tbl.period.nunique()),
    "features_A": FEATS_A, "features_B": FEATS_B,
    "feature_tiers": {"A0": FEATS_A0, "A1": FEATS_A1, "A2": FEATS_A2,
                      "img": _flt(IMG)},
    "never_feature": sorted(NEVER_FEATURE),
    "income_lag_years": INCOME_LAG_YEARS,
    "target": "price_median",
    "filters": {"min_n_tx": MIN_N_TX, "min_valid_pixel_ratio": VPR_FLOOR},
    "structural_repeat_with_price": RSM["structural_repeat_with_price"],
    "checks": mchecks,
    "known_biases": [
        "捷運通車前後特徵缺失（OSM start_date 覆蓋率 11%）→ **偏袒影像**："
        "新站周邊的新建案正好是 ΔNDBI 會抓到的。NB3 必須做「只留窗口內沒有新站的單元」"
        "的 robustness check",
        "土地使用分區用交易組成當代理、不是面積佔比 → **偏袒影像**："
        "建成區面積佔比與 NDBI 量的是同一件事",
        "戶數是靜態快照（無時間變異），是慢變控制變數",
        "特徵集 A 的定義本身會讓 ΔMAPE 大幅移動（實測跨合理定義的分散比 2 pp 的"
        "成功門檻還大）→ 分成 A0（純外生）／A1（+同期建物屬性，計畫字面定義）／"
        "A2（+分區組成）三層，NB3 要三層都報，結論只在符號一致時才成立",
        "A1／A2 含「當期那批交易」算出來的組成（坪數、屋齡、樓層中位數）——"
        "那與 Y 出自同一個 groupby，對「預測」的主張而言是同期前視。"
        "主結果用 A1（計畫字面要求）但必須同時報 A0",
        "n_solar_day / valid_pixel_ratio 是 Sentinel-2 的產物，已從特徵集移除"
        "（只當品質篩選）——放進 A 的話「無影像基準」就用到了衛星資料",
        f"2024 年的所得用 {max(have_years)} 年遞延（財政部尚未發布）"
        "→ NB3 要對 income_year_used 做敏感度測試",
        "NB1 提醒：2017 各季影像不可用、帶差分特徵的窗口從 2019Q1 起——"
        "NB3 的時間切分要照這個調整，不要照計畫的 2017Q1",
    ],
}
(PROC / "merge_meta.json").write_text(
    json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"已寫出 {PROC / 'merge_meta.json'}")
print(f"\n特徵集 A：{len(FEATS_A)} 個｜特徵集 B：{len(FEATS_B)} 個"
      f"（多了 {sorted(set(FEATS_B) - set(FEATS_A))}）")
print(f"checks.passed = {mchecks['passed']}")
''')

# ============================== 組裝 ==============================
def cell_id(kind: str, source: str, seen: dict) -> str:
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
    "nbformat": 4, "nbformat_minor": 5,
}
try:
    import nbformat

    n_changes, nb = nbformat.validator.normalize(nb, version=4, version_minor=5)
    nbformat.validate(nb)
    print(f"nbformat: valid (normalize made {n_changes} fixes)")
except ImportError:
    print("WARNING: nbformat not installed — skipping validation")

out = (pathlib.Path(__file__).resolve().parent.parent
       / "notebooks" / "02_data_merge.ipynb")
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"wrote {out}  cells={len(CELLS)}")
