# -*- coding: utf-8 -*-
"""產生 notebooks/01_imagery.ipynb"""
import hashlib, json, pathlib

CELLS = []
def md(s):   CELLS.append(("markdown", s.strip("\n")))
def code(s): CELLS.append(("code", s.strip("\n")))

# ============================== 標題 ==============================
md(r"""
# NB1 — 遙測指數（Sentinel-2 → 分析單元 × 季）

規格：`taiwan-housets-mvp-plan_3.md` 計畫 §5.1。**粒度一律從 NB0 的
`density_decision.json` 讀，不要硬編。**

## 與計畫的兩處差異（都有實測依據）

**1. 不用 Google Earth Engine。** 計畫 §12 說第一天先註冊 GEE、等審核 1–3 天。
不需要——Earth Search STAC（`earth-search.aws.element84.com`）與 AWS 上的
`sentinel-cogs` 都可**匿名**存取，工具鏈是 `pystac-client` + `odc-stac` + `rasterio`，
一樣 $0、不必等審核。GEE 仍然是可選路徑（NB4 導出大 mosaic 時比較方便）。

**2. 影像窗口從 2018Q1 起，不是 2017Q1。** 實測（以臺北市大安區為幾何條件逐季數 scene）：

| 年 | Q1 | Q2 | Q3 | Q4 |（括號為雲量 < 40% 的張數）|
|---|---|---|---|---|---|
| 2017 | 4 (2) | **1 (0)** | 7 (5) | 7 (1) | ← 幾乎無法用 |
| 2018 | 19 (5) | 20 (7) | 22 (6) | 12 (5) | |
| 2019+ | 18–40 | 18–37 | 18–36 | 19–40 | |

`sentinel-2-l1c` 的 2017 一樣稀疏、`sentinel-2-c1-l2a` 的 2017 是 0——所以這不是
L2A 產製的問題，是 ESA 當年對東亞的取像計畫本來就疏（S2B 2017 年中才服役）。
**2017Q2 連一張雲量 < 40% 的都沒有。**

本 notebook 仍然把 2017 一起抽出來（成本很低，因為它本來就沒幾張），但會在涵蓋率
檢查裡明確標記哪幾季不可靠。ΔNDVI/ΔNDBI 需要去年同季，所以**帶差分特徵的建模窗口
實際上從 2019Q1 開始**——NB3 的時間切分要照這個調整，不要照計畫的 2017Q1。

## 這本 notebook 的產出

| 檔案 | 用途 |
|---|---|
| `data/processed/unit_remote_sensing.csv` | 分析單元 × 季 × 指數（NB2 讀這個） |
| `data/processed/rs_report.md` | 人看的抽取報告 |
| `data/raw/rs_cache/*.parquet` | 逐（縣市 × 季）快取，中斷可續跑 |
""")

# ============================== §0 環境 ==============================
md(r"""
---
## 0. 環境設定
""")

code(r"""
# Colab 執行時取消註解
# !pip install -q pystac-client odc-stac rioxarray geopandas shapely dask pyarrow
""")

code(r'''
import io
import json
import os
import re
import time
import zipfile
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 60)

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
RS_CACHE, BND_DIR = RAW / "rs_cache", RAW / "boundary"
FIG_DIR = PROC / "figs"
for _d in (RAW, PROC, RS_CACHE, BND_DIR, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

print(f"IN_COLAB = {IN_COLAB}")
print(f"PROJECT  = {PROJECT}")
''')

code(r'''
# ---- 粒度：一律從 NB0 的決策讀，不要硬編 ----
DEC_PATH = PROC / "density_decision.json"
if not DEC_PATH.exists():
    raise FileNotFoundError(
        f"找不到 {DEC_PATH}。先跑 NB0（00_density_check.ipynb）取得粒度決策。\n"
        "NB1 的分析單元、期數、乃至於影像解析度的選擇都取決於它。")

_dec = json.loads(DEC_PATH.read_text(encoding="utf-8"))
DECISION = _dec["decision"]
SPATIAL_UNIT = DECISION["spatial_unit"]      # "鄉鎮市區" 或 "村里"
TIME_UNIT = DECISION["time_unit"]            # "月" 或 "季"

print(f"NB0 的決策：{DECISION['scheme']}"
      f"（精確判定={DECISION['exact']}，需要 geocoding={DECISION['needs_geocoding']}）")
print(f"  理由：{DECISION['note']}")
if DECISION["blocked_by_denominator_conflict"]:
    raise RuntimeError("NB0 的分母口徑衝突尚未拍板（decision.scheme 是 null），"
                       "先回去處理再跑 NB1。")
if SPATIAL_UNIT not in ("鄉鎮市區", "村里"):
    raise ValueError(f"未預期的空間粒度：{SPATIAL_UNIT}")

# **影像一律是季度**（計畫 §2：季度合成，降低台灣高雲覆蓋的影響）。
# 若房價粒度是「月」，每季 3 個月共用同一組遙測值 → 計畫 §5.3 的結構性重複問題會發生，
# NB3 的 paired t-test 必須以「分析單元 × 季」為 pairing 單位。這裡先把它記進產出。
IMAGERY_FREQ = "季"
STRUCTURAL_REPEAT = (TIME_UNIT == "月")
print(f"影像頻率固定為「{IMAGERY_FREQ}」；房價粒度是「{TIME_UNIT}」"
      f" → 結構性重複={STRUCTURAL_REPEAT}")
''')

code(r'''
# ============================ 抽取設定 ============================
CITY_ORDER = ["臺北市", "新北市", "桃園市", "臺中市", "臺南市", "高雄市"]

# 影像窗口。刻意比 NB0 的分析窗口（2017Q1–2024Q4）一樣寬——2017 幾乎沒有可用影像，
# 但把它抽出來成本極低，而且「明確標記為不可靠」比「悄悄不存在」好。
Q_START, Q_END = "2017Q1", "2024Q4"
QUARTERS = [str(q) for q in pd.period_range(Q_START, Q_END, freq="Q")]

STAC_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"
BANDS = ("red", "nir", "swir16", "green", "scl")
# SCL 雲遮罩（計畫 §5.1）：保留 4 植被 / 5 裸地 / 6 水體；
# 排除 1 飽和 / 3 雲影 / 8 中機率雲 / 9 高機率雲 / 10 卷雲 / 11 冰雪；
# 2 暗區域 與 7 未分類 先排除，若有效像素覆蓋率不足再放寬（計畫建議）。
SCL_KEEP = (4, 5, 6)

# 解析度。**按縣市載入、再對單元做 zonal mean**，不要逐單元載：
# 實測逐單元載一個 (區 × 季) 要 22.6 秒 → 158 區 × 32 季 ≈ 28 小時，不可行；
# 按縣市載則臺北 9 秒、高雄 100–480 秒。
# **200m 是量過之後的選擇，不是省事。** 實測（高雄市 2019Q1，最重的縣市）：
#   100m → 1104×905 格、478 秒、區域平均 NDVI 0.534
#   200m →  553×453 格、103 秒、區域平均 NDVI 0.530
# 區域平均只差 0.004（降解析度幾乎不動空間平均，因為那本來就是在做平均），
# 但快 4.6 倍。100m 跑全量實測要 5 小時以上（單一 (縣市×季) 最壞 25 分鐘），200m 約 1 小時。
# 代價：最小的單元（高雄鹽埕 1.4 km²）只剩約 35 個像素。所以產出帶 `n_pixel`，
# 下游可以據此標記或降權——不要靜默地讓 35 個像素的平均跟 3000 個像素的平均等價。
# 若粒度是村里（均面積 0.3 km²）必須降到 20–30m，成本會漲一個量級。
RESOLUTION = 200 if SPATIAL_UNIT == "鄉鎮市區" else 20
MIN_PIXEL_WARN = 50            # 單元像素數低於此值時在涵蓋率檢查裡列出來
CRS = "EPSG:32651"          # UTM 51N；六都都在 120–122E

# 雲量門檻自適應：先嚴、solar day 不足才放寬。固定用嚴的會讓 2018 那種本來就疏的季空掉。
CLOUD_STEPS = (35, 60, 101)
MIN_DAYS_TARGET = 3

MAX_WORKERS = 3             # 平行的 (縣市 × 季) 數；網路瓶頸，太多反而互相拖累
DASK_THREADS = 4

print(f"空間單元 = {SPATIAL_UNIT}｜解析度 = {RESOLUTION} m｜"
      f"季數 = {len(QUARTERS)}（{Q_START} ~ {Q_END}）")
''')

# ============================== §1 邊界 ==============================
md(r"""
---
## 1. 分析單元邊界

來源：政府資料開放平臺 dataset **7441**「鄉鎮市區界線(TWD97經緯度)」（國土測繪中心）。
匿名可下載，但有三個坑：

1. **URL 含中文**，要 `urllib.parse.quote` 過 path 才抓得到
2. **zip 內檔名是 cp950**，`unzip` 會亂碼並中斷 → 用 `zipfile` 自己解碼
3. **高雄市旗津區的 geometry 含東沙／南沙群島** → 不先跟台灣本島 bbox 取交集，
   那個縣市的 bbox 會延伸到北緯 10 度、影像抽取直接爆掉

單元數會跟 NB0 的母體對照——對不上就中止，因為那代表兩本 notebook 的分析單元不是同一組。
""")

code(r'''
BOUNDARY_URLS = {
    "鄉鎮市區": "https://www.tgos.tw/tgos/VirtualDir/Product/"
                "3fe61d4a-ca23-4f45-8aca-4a536f40f290/鄉(鎮、市、區)界線1140318.zip",
}
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"}


def fetch_boundary(kind: str) -> Path:
    """下載並解壓邊界 zip，回傳 .shp 的路徑。已存在就直接用。"""
    if kind not in BOUNDARY_URLS:
        raise NotImplementedError(
            f"還沒接 {kind} 的邊界圖。村里用的是同機關的「村里界圖(TWD97經緯度)」，"
            "接法一樣（中文 URL + cp950 檔名）。")
    out = BND_DIR / kind
    shps = sorted(out.glob("*.shp"))
    if shps:
        return max(shps, key=lambda p: p.stat().st_size)

    url = BOUNDARY_URLS[kind]
    sp = urllib.parse.urlsplit(url)
    safe = urllib.parse.urlunsplit(
        (sp.scheme, sp.netloc, urllib.parse.quote(sp.path), sp.query, sp.fragment))
    print(f"下載邊界圖：{url.rsplit('/', 1)[-1]}")
    resp = requests.get(safe, headers=UA, timeout=180)
    resp.raise_for_status()

    out.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(io.BytesIO(resp.content))
    for info in z.infolist():
        if info.is_dir():
            continue
        # zip 的檔名旗標若沒設 UTF-8，Python 會用 cp437 解——要還原位元組再用 cp950 解
        raw = (info.filename.encode("utf-8") if info.flag_bits & 0x800
               else info.filename.encode("cp437"))
        name = None
        for enc in ("cp950", "big5", "utf-8"):
            try:
                name = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        (out / Path(name or info.filename).name).write_bytes(z.read(info))
    shps = sorted(out.glob("*.shp"))
    if not shps:
        raise RuntimeError(f"{out} 裡沒有 .shp，來源的打包方式可能改了")
    return max(shps, key=lambda p: p.stat().st_size)
''')

code(r'''
import geopandas as gpd
from shapely.geometry import box

# 台灣本島 bbox。旗津區的 geometry 含東沙（北緯 20.7）與南沙（北緯 10.4），
# 不取交集的話該縣市的 convex hull 會橫跨 15 個緯度。
TW_BBOX = box(119.3, 21.7, 122.2, 25.4)

shp = fetch_boundary(SPATIAL_UNIT)
_g = gpd.read_file(shp).to_crs(4326)
print(f"邊界圖：{shp.name}（全國 {len(_g)} 筆）")

_g = _g[_g["COUNTYNAME"].isin(CITY_ORDER)].copy()
_before = _g.geometry.total_bounds
_g["geometry"] = _g.geometry.intersection(TW_BBOX)
_g = _g[~_g.geometry.is_empty].copy()
_g["unit"] = _g["COUNTYNAME"] + _g["TOWNNAME"]
units = _g[["COUNTYNAME", "TOWNNAME", "unit", "TOWNID", "geometry"]].reset_index(drop=True)

print(f"六都單元：{len(units)} 個")
print(f"  取交集前的 bounds：{np.round(_before, 3)}")
print(f"  取交集後的 bounds：{np.round(units.geometry.total_bounds, 3)}"
      f"  ← 沒有壓到 22°N 以下才對（東沙／南沙已排除）")
print(units.groupby("COUNTYNAME").size().reindex(CITY_ORDER).to_string())

# 面積（用等面積投影算，不要用經緯度）
units["area_km2"] = units.to_crs("EPSG:3826").area / 1e6
print(f"\n單元面積 km²：min {units.area_km2.min():.2f}"
      f"（{units.loc[units.area_km2.idxmin(), 'unit']}）、"
      f"median {units.area_km2.median():.1f}、max {units.area_km2.max():.0f}")
_px = units.area_km2 * 1e6 / (RESOLUTION ** 2)
print(f"{RESOLUTION}m 下每個單元的像素數：min {_px.min():.0f}、median {_px.median():.0f}")
if _px.min() < 30:
    print(f"⚠️  最小的單元只有 {_px.min():.0f} 個像素，空間平均會很不穩——"
          f"考慮把 RESOLUTION 調小")
''')

code(r'''
# ---- 與 NB0 的母體對照：對不上就中止 ----
# 兩本 notebook 若用了不同的單元集合，NB2 join 出來的大表會靜默少掉一批單元，
# 而模型照樣訓練得起來、指標照樣算得出來。
_nb0_units = set(_dec["denominator"]["full"].get("districts") or [])
if not _nb0_units:
    # NB0 舊版沒有存單元清單，退而比數量
    _n0 = _dec["denominator"]["full"]["n_districts" if SPATIAL_UNIT == "鄉鎮市區"
                                      else "n_villages"]
    if len(units) != _n0:
        raise RuntimeError(
            f"單元數對不上：邊界圖 {len(units)} 個、NB0 母體 {_n0} 個。\n"
            "多半是邊界圖的年份與 NB0 的村里／鄉鎮清單年份不同（改制、合併）。\n"
            "計畫 §10 風險 9 要求以 2020 年邊界為基準，兩邊要用同一套。")
    print(f"✅ 單元數與 NB0 母體一致（{len(units)} 個）")
else:
    miss = _nb0_units - set(units["unit"])
    extra = set(units["unit"]) - _nb0_units
    if miss or extra:
        raise RuntimeError(
            f"單元集合對不上 NB0 母體：NB0 有而邊界圖沒有 {sorted(miss)[:10]}"
            f"（共 {len(miss)}）、邊界圖有而 NB0 沒有 {sorted(extra)[:10]}"
            f"（共 {len(extra)}）")
    print(f"✅ 單元集合與 NB0 母體逐一相符（{len(units)} 個）")
''')

# ============================== §2 抽取 ==============================
md(r"""
---
## 2. Sentinel-2 抽取

流程：**每（縣市 × 季）載一次影像 → 算指數 → 對該縣市所有單元做 zonal mean**。

### 反射率換算：一個會靜默毀掉整個實驗的地方

Sentinel-2 的 processing baseline 04.00（2022-01-25 起）給 L2A 加了
`BOA_ADD_OFFSET = -1000`，反射率變成 `DN × 1e-4 − 0.1`。
**NDVI 是比值但 offset 是加法，不會抵銷**——沒處理的話 2022 之後會出現一個假的階梯，
長得跟「真的都市化」一模一樣，而且方向剛好會讓「加了影像特徵後 MAPE 變好」變得不可信。

實測結論（Earth Search 的 `sentinel-2-l2a`）：**這個 collection 已經幫你套好了。**
post-2021 的 item 帶 `raster:bands.offset = -0.1`，但同時
`earthsearch:boa_offset_applied = True`——那個 offset 是 provenance metadata，
不是要你再套一次的指令。

**證據不要只看 metadata，要看像素。** 同一片山林（臺北文山區）同一季（Q1）的 NIR
原始 DN 中位數，2019→2024 是 **2318 / 2379 / 2308 / 2516 / 2467 / 2475**——沒有 1000
的階梯。若再自己套一次 offset，NDVI 會爆掉（實測 2022Q1 = **−1442**，因為分母趨近 0）。

所以下面的 `reflectance_offset()` 讀**兩個**欄位，只在 `offset != 0` **且**
`boa_offset_applied` 為 False 時才自己補。這樣未來若出現未統一口徑的 item 也不會靜默壞掉。
""")

code(r'''
# **匿名讀 AWS 公開 bucket 必須先設好，而且要在 import rasterio 之前。**
# Earth Search 有部分 asset href 是 s3://，GDAL 對那些路徑會去找 ~/.aws/credentials，
# 找不到就丟 RasterioIOError。症狀是**間歇性**失敗（同一季有的縣市過、有的不過），
# 很容易被當成網路不穩而漏掉——實測 192 個工作裡有 4 個這樣掛掉。
os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
os.environ.setdefault("AWS_DEFAULT_REGION", "us-west-2")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")

import pystac_client
import odc.stac
import xarray as xr
import dask
from rasterio.features import rasterize

odc.stac.configure_rio(cloud_defaults=True, aws={"aws_unsigned": True})
dask.config.set(scheduler="threads", num_workers=DASK_THREADS)

SCALE = 1e-4
EPS_DENOM = 0.02        # 分母（nir+red 等）小於這個值的像素不算——比值會爆到幾百


def reflectance_offset(item) -> float:
    """回傳「我們還需要自己補」的 offset。已經套進像素的回 0。"""
    rb = (item.assets["nir"].extra_fields.get("raster:bands") or [{}])[0]
    off = float(rb.get("offset") or 0.0)
    applied = bool(item.properties.get("earthsearch:boa_offset_applied", False))
    return 0.0 if applied else off


def search_items(cat, geom, q: pd.Period) -> tuple:
    """自適應雲量門檻。回傳 (items, 實際用到的門檻)。"""
    items = []
    for cc in CLOUD_STEPS:
        query = None if cc > 100 else {"eo:cloud_cover": {"lt": cc}}
        items = list(cat.search(collections=[COLLECTION], intersects=geom,
                                datetime=f"{q.start_time.date()}/{q.end_time.date()}",
                                query=query).items())
        if len(items) >= MIN_DAYS_TARGET:
            return items, cc
    return items, CLOUD_STEPS[-1]


def quarter_composite(items, geom) -> xr.Dataset:
    """SCL 遮罩後的逐季中位數合成。按「需要自己補的 offset」分組載入再併起來。"""
    groups: dict = {}
    for it in items:
        groups.setdefault(reflectance_offset(it), []).append(it)
    parts, n_days = [], 0
    for off, its in groups.items():
        ds = odc.stac.load(its, bands=BANDS, geopolygon=geom, resolution=RESOLUTION,
                           crs=CRS, chunks={"x": 2048, "y": 2048}, groupby="solar_day",
                           dtype="uint16", nodata=0)
        m = ds.scl.isin(SCL_KEEP)
        # DN == 0 是 nodata。不排掉的話 off != 0 時會變成一個假的負反射率。
        parts.append(xr.Dataset({
            b: (ds[b].where(m & (ds[b] > 0)).astype("float32") * SCALE + off)
            for b in BANDS if b != "scl"}))
        n_days += int(ds.sizes["time"])
    allp = xr.concat(parts, dim="time").sortby("time") if len(parts) > 1 else parts[0]
    out = allp.median("time").compute()
    out.attrs["n_solar_days"] = n_days
    out.attrs["offsets_applied"] = sorted(groups)
    return out


def _ratio(a, b):
    """(a-b)/(a+b)，分母過小的像素設成 NaN（否則比值會爆到幾百）。"""
    den = a + b
    return ((a - b) / den).where(np.abs(den) > EPS_DENOM)


def indices(out: xr.Dataset) -> dict:
    """NDVI（有效解析度 10m）／NDBI、MNDWI（用 B11，原生 20m，經重採樣對齊）。"""
    r, n, s, g = out.red, out.nir, out.swir16, out.green
    return {"NDVI": _ratio(n, r).values,
            "NDBI": _ratio(s, n).values,
            "MNDWI": _ratio(g, s).values}
''')

code(r'''
def zonal(sub: gpd.GeoDataFrame, out: xr.Dataset, idx: dict) -> pd.DataFrame:
    """把單元多邊形燒到影像格網上，逐單元取空間平均與有效像素佔比。"""
    tr = out.odc.geobox.transform
    shape = (out.sizes["y"], out.sizes["x"])
    u = sub.to_crs(CRS)
    # 0 保留給「不屬於任何單元」，所以單元編號從 1 起。
    # all_touched=True：小單元在 100m 格網下可能一個像素中心都沒落在裡面。
    # 代價是邊界那一圈像素會混進鄰區的地表覆蓋——但**每一季用的是同一組幾何**，
    # 所以那個偏誤對每個單元是常數，會被模型的單元層級特徵吸收掉，
    # 不會污染「同一單元的時間變化」（那才是本實驗要的訊號）。
    lab = rasterize(((geom, i + 1) for i, geom in enumerate(u.geometry)),
                    out_shape=shape, transform=tr, fill=0, dtype="int32",
                    all_touched=True)
    rows = []
    for i, (_, r) in enumerate(u.iterrows(), start=1):
        sel = lab == i
        n_px = int(sel.sum())
        rec = {"unit": r["unit"], "n_pixel": n_px}
        if not n_px:
            rec.update({k: np.nan for k in idx})
            rec["valid_pixel_ratio"] = 0.0
        else:
            fin = np.isfinite(idx["NDVI"][sel])
            rec["valid_pixel_ratio"] = float(fin.mean())
            for k, arr in idx.items():
                v = arr[sel]
                v = v[np.isfinite(v)]
                rec[k] = float(v.mean()) if len(v) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def city_quarter(city: str, qs: str) -> pd.DataFrame:
    """一個 (縣市 × 季)。已有快取就直接讀。"""
    cache = RS_CACHE / f"{city}_{qs}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    cat = pystac_client.Client.open(STAC_URL)
    sub = units[units["COUNTYNAME"] == city]
    hull = sub.geometry.union_all().convex_hull
    q = pd.Period(qs, freq="Q")
    t0 = time.time()
    items, cc = search_items(cat, hull, q)
    if not items:
        d = pd.DataFrame({"unit": sub["unit"].to_numpy(), "n_pixel": 0,
                          "valid_pixel_ratio": 0.0,
                          "NDVI": np.nan, "NDBI": np.nan, "MNDWI": np.nan})
        d["n_scene"], d["n_solar_day"], d["offset_applied"] = 0, 0, ""
    else:
        out = quarter_composite(items, hull)
        d = zonal(sub, out, indices(out))
        d["n_scene"], d["n_solar_day"] = len(items), out.attrs["n_solar_days"]
        d["offset_applied"] = ",".join(f"{o:g}" for o in out.attrs["offsets_applied"])
    d["cloud_lt"], d["縣市"], d["quarter"] = cc, city, qs
    d["secs"] = round(time.time() - t0, 1)
    d.to_parquet(cache, index=False)
    return d
''')

code(r'''
# ---- 全量抽取（可中斷續跑）----
# 每個 (縣市 × 季) 一個 parquet；重跑只會補缺的。
jobs = [(c, q) for q in QUARTERS for c in CITY_ORDER]
todo = [j for j in jobs if not (RS_CACHE / f"{j[0]}_{j[1]}.parquet").exists()]
print(f"共 {len(jobs)} 個（縣市 × 季），待抽 {len(todo)} 個"
      f"（已快取 {len(jobs) - len(todo)} 個）")

if todo:
    t0, done, fails = time.time(), 0, []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(city_quarter, c, q): (c, q) for c, q in todo}
        for f in as_completed(futs):
            city, qs = futs[f]
            done += 1
            try:
                d = f.result()
                msg = f"n={len(d)} days={int(d.n_solar_day.iloc[0])}"
            except Exception as exc:                      # noqa: BLE001
                msg = f"FAIL {type(exc).__name__}: {exc}"
                fails.append((city, qs, msg))
            print(f"[{done}/{len(todo)}] {city} {qs}  {msg}"
                  f"  {(time.time() - t0) / 60:.1f}m", flush=True)
    if fails:
        print(f"\n⚠️  {len(fails)} 個失敗，重跑本 cell 會只補這些：")
        for c, q, m in fails[:10]:
            print(f"    {c} {q}  {m}")

rs = pd.concat([pd.read_parquet(p) for p in sorted(RS_CACHE.glob("*.parquet"))],
               ignore_index=True)
rs["quarter"] = rs["quarter"].astype(str)
print(f"\n合併後：{len(rs):,} 列（單元 {rs.unit.nunique()} × 季 {rs.quarter.nunique()}）")
''')

# ============================== §3 涵蓋率 ==============================
md(r"""
---
## 3. 涵蓋率與合理性檢查

跟 NB0 的 §3.2 同一個立場：**遙測特徵出錯時不會有自然的失敗訊號**。雲遮罩把某幾季刮掉、
反射率口徑改變、單元邊界對不上——這些都會讓 NDVI 照樣算得出來、圖照樣畫得出來、
模型照樣訓練得起來，只是數字是錯的。所以在交給 NB2 之前先擋幾層。

| 層 | 檢查 | 抓什麼 | 會中止 |
|---|---|---|---|
| **I1** | （單元 × 季）矩陣完整 | 某些單元或季整批沒抽到 | ✅ |
| **I2** | 有效像素佔比 | 雲遮罩把資料刮掉（絕對下限 ＋ 逐季趨勢） | ✅ |
| **I3** | 每季的 solar day 數 | 合成只用了 1–2 天 → 中位數沒有抗離群能力 | ⚠️ 標記 |
| **I4** | 指數值域 | 分母保護失效、反射率口徑錯 | ✅ |
| **I5** | **已知日期測試點** | processing baseline 轉換（2022-01-25）造成的假階梯 | ✅ |
| **I6** | 季節性方向 | 合成邏輯壞掉（台灣 NDVI 夏 > 冬） | ✅ |
| **I7** | **跨來源空間相關** | 單元對位錯（NDVI 應與戶數密度負相關） | ✅ |

**I5 與 I7 是這一節唯二有「外部參照」的檢查**，其他都是拿資料跟自己比——
跟 NB0 學到的一樣，那類檢查對「基準被污染」結構性無免疫。
I5 的外部參照是**已知的日期**（ESA 公布的 baseline 轉換日）；
I7 的外部參照是**另一個資料源**（NB0 從戶政司抓的戶數）。
""")

code(r'''
COVERAGE_STRICT = True          # False = 只警告不中止
VPR_FLOOR = 0.30                # 計畫 §5.2 步驟 8：valid_pixel_ratio < 0.3 的 cell 不可靠
VPR_QUARTER_MIN = 0.50          # 逐季平均有效像素佔比的下限
MIN_SOLAR_DAYS = 3              # 低於此值的季標記為不可靠（不中止）
BASELINE_SPLIT = "2022Q1"       # ESA processing baseline 04.00 生效（2022-01-25）
BASELINE_TOL = 0.10             # 穩定地物在轉換前後的 NDVI 相對差上限
STABLE_FRAC = 0.20              # 取 NDVI 最高的這個比例當「穩定地物」（山林區）


class ImageryError(RuntimeError):
    """遙測抽取的合理性檢查失敗。數字會錯但看起來正常，故預設中止。"""


def imagery_fail(title: str, detail: str) -> None:
    msg = f"{title}\n{detail}"
    if COVERAGE_STRICT:
        raise ImageryError(msg + "\n\n若已確認可以接受，把 COVERAGE_STRICT 設為 False 再重跑。")
    print(f"{msg}\n\n（COVERAGE_STRICT = False，僅警告不中止）\n")


checks: dict = {}
''')

code(r'''
# ---- I1：（單元 × 季）矩陣完整 ----
want = pd.MultiIndex.from_product([sorted(units["unit"]), QUARTERS],
                                  names=["unit", "quarter"])
have = pd.MultiIndex.from_frame(rs[["unit", "quarter"]].drop_duplicates()
                                  .sort_values(["unit", "quarter"]))
missing_cells = want.difference(have)
extra_cells = have.difference(want)
checks["I1"] = {"n_expected": len(want), "n_have": len(have),
                "n_missing": len(missing_cells), "n_extra": len(extra_cells)}
if len(missing_cells) or len(extra_cells):
    imagery_fail(
        f"❌ I1：（單元 × 季）矩陣不完整"
        f"（缺 {len(missing_cells)}、多 {len(extra_cells)}）",
        "\n".join([f"    缺：{list(missing_cells[:10])}"] +
                  ([f"    多：{list(extra_cells[:10])}"] if len(extra_cells) else []) +
                  ["", "缺格子多半是某個 (縣市 × 季) 抽取失敗 → 重跑 §2 的抽取 cell；",
                   "多格子代表 units 與快取不一致（邊界圖換版了）→ 清掉 rs_cache 重抽。"]))
else:
    print(f"✅ I1：{len(want):,} 個（單元 × 季）全部到齊")
''')

code(r'''
# ---- I2：有效像素佔比 ----
rs["low_vpr"] = rs["valid_pixel_ratio"] < VPR_FLOOR
by_q = (rs.groupby("quarter")
          .agg(vpr_mean=("valid_pixel_ratio", "mean"),
               vpr_min=("valid_pixel_ratio", "min"),
               n_low=("low_vpr", "sum"),
               days_min=("n_solar_day", "min"))
          .reindex(QUARTERS))
print("逐季的有效像素佔比與 solar day 數\n")
print(by_q.round(3).to_string())

bad_q = by_q[by_q["vpr_mean"] < VPR_QUARTER_MIN]
checks["I2"] = {"floor": VPR_FLOOR, "quarter_min": VPR_QUARTER_MIN,
                "n_low_cells": int(rs["low_vpr"].sum()),
                "low_cell_share": round(float(rs["low_vpr"].mean()), 4),
                "bad_quarters": list(bad_q.index)}
print(f"\nvalid_pixel_ratio < {VPR_FLOOR} 的 cell：{int(rs['low_vpr'].sum()):,} 個"
      f"（{rs['low_vpr'].mean():.1%}）")
if len(bad_q):
    imagery_fail(
        f"❌ I2：{len(bad_q)} 個季的平均有效像素佔比低於 {VPR_QUARTER_MIN}",
        "\n".join([f"    {q}：平均 {r.vpr_mean:.2f}、最小 {r.vpr_min:.2f}、"
                   f"solar day 最少 {r.days_min:.0f}" for q, r in bad_q.iterrows()] +
                  ["", "那幾季的雲遮罩把大部分像素刮掉了。可以：",
                   "  · 放寬 SCL_KEEP（加入 2 暗區域 / 7 未分類，計畫 §5.1 允許）",
                   "  · 引入 Sentinel-1 SAR 當備援（SAR 不受雲影響，計畫 §4.2）",
                   "  · 或接受並在 NB2 用 valid_pixel_ratio 篩掉那些 cell"]))
else:
    print(f"✅ I2：所有季的平均有效像素佔比都 ≥ {VPR_QUARTER_MIN}")
''')

code(r'''
# ---- 小單元提示（不中止）----
_small = (rs.groupby("unit")["n_pixel"].max()
            .loc[lambda s: s < MIN_PIXEL_WARN].sort_values())
checks["small_units"] = {"threshold": MIN_PIXEL_WARN,
                         "units": {k: int(v) for k, v in _small.items()}}
if len(_small):
    print(f"⚠️  {len(_small)} 個單元的像素數少於 {MIN_PIXEL_WARN}"
          f"（{RESOLUTION}m 格網下面積太小），空間平均會比較不穩：")
    print("    " + "、".join(f"{k}({v})" for k, v in _small.items()))
    print("    產出的 n_pixel 欄可以讓 NB2/NB3 標記或降權這些單元。")
else:
    print(f"✅ 所有單元的像素數都 ≥ {MIN_PIXEL_WARN}")

# ---- I3：solar day 數（標記，不中止）----
thin = by_q[by_q["days_min"] < MIN_SOLAR_DAYS]
checks["I3"] = {"min_days": MIN_SOLAR_DAYS, "thin_quarters": list(thin.index)}
if len(thin):
    print(f"⚠️  I3：{len(thin)} 個季的 solar day 數少於 {MIN_SOLAR_DAYS}"
          f"，中位數合成沒有抗離群能力：\n    {list(thin.index)}")
    print("    這是預期的——2017 年 ESA 對東亞的取像計畫本來就疏（見本 notebook 開頭）。")
    print("    NB3 的建模窗口請避開這些季；產出的 csv 有 n_solar_day 欄可以篩。")
else:
    print(f"✅ I3：所有季的 solar day 數都 ≥ {MIN_SOLAR_DAYS}")
''')

code(r'''
# ---- I4：指數值域（絕對檢查）----
IDX_COLS = ["NDVI", "NDBI", "MNDWI"]
oob = {c: int(((rs[c] < -1) | (rs[c] > 1)).sum()) for c in IDX_COLS}
nan_share = {c: round(float(rs[c].isna().mean()), 4) for c in IDX_COLS}
checks["I4"] = {"out_of_range": oob, "nan_share": nan_share}
print("指數值域\n")
print(rs[IDX_COLS].describe().round(3).to_string())
print(f"\n超出 [-1, 1] 的數量：{oob}｜NaN 佔比：{nan_share}")
if any(oob.values()):
    imagery_fail(
        f"❌ I4：有指數超出 [-1, 1]：{oob}",
        "    正規化差值指數在數學上不可能超出這個範圍——除非分母趨近 0 而保護失效。\n"
        f"    檢查 EPS_DENOM（目前 {EPS_DENOM}）與 reflectance_offset() 的處理；\n"
        "    若 offset 被套了兩次，NDVI 會爆到幾百（實測 -1442）。")
else:
    print("✅ I4：所有指數都在 [-1, 1] 內")
''')

code(r'''
# ---- I5：已知日期測試點（processing baseline 轉換）----
# 這是本節唯一有「絕對外部參照」的檢查：ESA 公布 baseline 04.00 於 2022-01-25 生效，
# 那個日期是已知的，跟我們的資料無關。穩定地物（山林）的 NDVI 不該在那裡出現階梯。
# 若反射率口徑處理錯了（offset 沒套或套了兩次），這裡一定會中標。
_mean_ndvi = rs.groupby("unit")["NDVI"].mean()
stable = set(_mean_ndvi.nlargest(max(3, int(len(_mean_ndvi) * STABLE_FRAC))).index)
_st = rs[rs["unit"].isin(stable) & rs["NDVI"].notna()].copy()
_st["q"] = pd.PeriodIndex(_st["quarter"], freq="Q")
_split = pd.Period(BASELINE_SPLIT, freq="Q")
# 同季比同季，避免季節性混進來
_piv = (_st.assign(season=_st["q"].quarter,
                   era=np.where(_st["q"] < _split, "before", "after"))
           .groupby(["season", "era"])["NDVI"].median().unstack("era"))
_piv = _piv.dropna()
_piv["相對差"] = (_piv["after"] - _piv["before"]) / _piv["before"].abs()
print(f"穩定地物（NDVI 前 {STABLE_FRAC:.0%} 的 {len(stable)} 個單元）"
      f"在 baseline 轉換（{BASELINE_SPLIT}）前後的 NDVI 中位數\n")
print(_piv.round(4).to_string())
worst = float(_piv["相對差"].abs().max()) if len(_piv) else np.nan
checks["I5"] = {"split": BASELINE_SPLIT, "tol": BASELINE_TOL,
                "n_stable_units": len(stable),
                "worst_rel_diff": None if pd.isna(worst) else round(worst, 4)}
print(f"\n最大相對差 {worst:.1%}（門檻 {BASELINE_TOL:.0%}）")
if pd.isna(worst) or worst > BASELINE_TOL:
    imagery_fail(
        f"❌ I5：穩定地物的 NDVI 在 {BASELINE_SPLIT} 前後差了 {worst:.1%}",
        "    ESA 的 processing baseline 04.00 於 2022-01-25 生效，加了\n"
        "    BOA_ADD_OFFSET = -1000。NDVI 是比值但 offset 是加法，不會抵銷。\n"
        "    看 rs['offset_applied'] 這一欄：\n"
        "      · 全是空字串 → 來源已統一口徑，那這個階梯是別的原因（雲遮罩偏差？）\n"
        "      · 有 -0.1 → 我們自己補了 offset。確認 earthsearch:boa_offset_applied\n"
        "        真的是 False，否則就是套了兩次。\n"
        "    這個階梯長得跟真的都市化一模一樣，不修掉的話整個 A/B 實驗都不可信。")
else:
    print(f"✅ I5：穩定地物在 baseline 轉換前後沒有階梯（最大 {worst:.1%}）")
''')

code(r'''
# ---- I6：季節性方向（絕對的領域先驗）----
# 台灣的植被 NDVI 夏（Q2/Q3）應該高於冬（Q1/Q4）。方向錯或完全沒有季節性，
# 代表合成邏輯壞了（例如時間軸被壓掉、或每季都抓到同一批影像）。
_s = rs.dropna(subset=["NDVI"]).copy()
_s["season"] = pd.PeriodIndex(_s["quarter"], freq="Q").quarter
season_ndvi = _s.groupby("season")["NDVI"].median()
summer = float(season_ndvi.reindex([2, 3]).mean())
winter = float(season_ndvi.reindex([1, 4]).mean())
checks["I6"] = {"by_season": {int(k): round(float(v), 4) for k, v in season_ndvi.items()},
                "summer_minus_winter": round(summer - winter, 4)}
print(f"NDVI 中位數 by 季：{season_ndvi.round(4).to_dict()}")
print(f"夏（Q2/Q3）{summer:.4f} − 冬（Q1/Q4）{winter:.4f} = {summer - winter:+.4f}")
if not (summer > winter):
    imagery_fail(
        f"❌ I6：夏季 NDVI（{summer:.4f}）沒有高於冬季（{winter:.4f}）",
        "    台灣的植被季節性方向是確定的。方向反了或完全沒有差異，代表：\n"
        "      · groupby(\"solar_day\") 之後時間軸被壓掉（每季只剩一個切片）\n"
        "      · 或雲遮罩在夏季（雲多）刮掉的正好是植被最盛的像素\n"
        "    後者要看 I2 的逐季 valid_pixel_ratio 是不是夏季特別低。")
else:
    print("✅ I6：季節性方向正確（夏 > 冬）")
''')

code(r'''
# ---- I7：跨來源空間相關（單元對位的獨立驗證）----
# 這一層的參照來自**另一個資料源**：NB0 從戶政司抓的戶數。
# 戶數密度高 = 都市化程度高 → NDVI 應該負相關、NDBI 正相關。
# 若單元對位錯了（rasterize 的 transform 錯、或 unit 名稱錯位），相關性會消失或反向——
# 而各單元的 NDVI 本身看起來完全正常。
# 戶數：直接讀 NB0 已經抓好並快取的村里表（戶政司 ODRP019）。
# 沒有快取就自己抓一次——這一層的價值就在於參照來自影像之外，不能因為缺快取就跳過。
_vpath = next(iter(sorted(RAW.glob("villages_*.parquet"))), None)
_uu = units[["unit", "area_km2"]].copy()
if _vpath is not None:
    _v = pd.read_parquet(_vpath)
    print(f"戶數來源：{_vpath.name}（NB0 抓的戶政司資料，{len(_v):,} 個村里）")
else:
    print("找不到 NB0 的村里快取，改用彙總表推估——I7 會退化，建議先跑 NB0")
    _v = pd.DataFrame(columns=["district", "戶數"])
if len(_v) and {"district", "戶數"} <= set(_v.columns):
    _hh = (_v.groupby("district", as_index=False)["戶數"].sum()
             .rename(columns={"district": "unit"}))
else:
    _hh = pd.DataFrame(columns=["unit", "戶數"])

if len(_hh):
    m = (rs.groupby("unit")[IDX_COLS].mean().reset_index()
           .merge(_uu, on="unit").merge(_hh, on="unit"))
    m["hh_density"] = m["戶數"] / m["area_km2"]
    rho = {c: float(m[[c, "hh_density"]].corr(method="spearman").iloc[0, 1])
           for c in IDX_COLS}
    checks["I7"] = {"n_units": len(m), "spearman_vs_hh_density":
                    {k: round(v, 3) for k, v in rho.items()}}
    print(f"與戶數密度的 Spearman 相關（{len(m)} 個單元）：")
    for c in IDX_COLS:
        print(f"  {c:<6} {rho[c]:+.3f}")
    if not (rho["NDVI"] < -0.3 and rho["NDBI"] > 0.3):
        imagery_fail(
            f"❌ I7：與戶數密度的相關性不符預期"
            f"（NDVI {rho['NDVI']:+.3f} 應 < -0.3、NDBI {rho['NDBI']:+.3f} 應 > +0.3）",
            "    戶數密度是**獨立來源**（戶政司），與影像無關。相關性消失或反向代表\n"
            "    單元對位出錯——最可能是 zonal() 的 transform 或 unit 名稱錯位。\n"
            "    先抽查幾個單元：都市核心（如臺北市大同區）的 NDVI 應該明顯低於\n"
            "    山區（如臺北市北投區）。")
    else:
        print("✅ I7：NDVI 與戶數密度負相關、NDBI 正相關，單元對位可信")
else:
    checks["I7"] = {"skipped": "找不到戶數資料（NB0 的 villages_*.parquet）"}
    print("⚠️  I7 跳過：找不到戶數資料，無法做跨來源驗證。"
          "這是 NB1 唯二有外部參照的檢查之一，別就這樣放過去——先跑 NB0。")
''')

code(r'''
checks["passed"] = bool(
    checks["I1"]["n_missing"] == 0 and checks["I1"]["n_extra"] == 0
    and not checks["I2"]["bad_quarters"]
    and not any(checks["I4"]["out_of_range"].values())
    and (checks["I5"].get("worst_rel_diff") is not None
         and checks["I5"]["worst_rel_diff"] <= BASELINE_TOL)
    and checks["I6"]["summer_minus_winter"] > 0
    and ("skipped" in checks["I7"]
         or (checks["I7"]["spearman_vs_hh_density"]["NDVI"] < -0.3
             and checks["I7"]["spearman_vs_hh_density"]["NDBI"] > 0.3)))
checks["limitations"] = [
    "I5 與 I7 之外的每一層都是拿資料跟自己比，對「基準被污染」無免疫力",
    "I5 只看得到 baseline 轉換那一個日期的階梯；別的日期發生的口徑變更看不到",
    "I7 是全期平均的空間相關，對「某幾季對位錯」失明（那需要逐季做）",
    "雲遮罩的偏差不是隨機的：雲多的季節被刮掉的像素比例更高，"
    "而那些季節正好是植被最盛的時候 → 指數的時間序列有系統性偏差，"
    "I2 只擋得住嚴重的情況",
    f"solar day < {MIN_SOLAR_DAYS} 的季只標記不中止（2017 全年都是）——"
    "NB3 要自己避開",
    "NDBI/MNDWI 用 B11（原生 20m），有效空間解析度是 20m 不是 10m",
]
print(f"\nchecks.passed = {checks['passed']}")
''')

# ============================== §4 差分 ==============================
md(r"""
---
## 4. 年際差分特徵

`ΔNDVI = NDVI(Q) − NDVI(Q−4)`、`ΔNDBI` 同理（計畫 §5.1 步驟 5）。

差分需要去年同季，所以**第一年沒有差分**。加上 2017 的影像本來就不可用，
帶差分特徵的可用窗口實際上是 **2019Q1 起**。
""")

code(r'''
rs["q"] = pd.PeriodIndex(rs["quarter"], freq="Q")
rs = rs.sort_values(["unit", "q"]).reset_index(drop=True)

for c in ("NDVI", "NDBI"):
    rs[f"d{c}"] = rs.groupby("unit")[c].diff(4)
    # diff(4) 只在「前面第 4 列真的是去年同季」時才對——缺季會讓它錯位
    prev_q = rs.groupby("unit")["q"].shift(4)
    ok = (rs["q"] - prev_q).apply(lambda x: getattr(x, "n", None) == 4)
    rs.loc[~ok, f"d{c}"] = np.nan

print("差分特徵的可用性（非 NaN 佔比）")
print(rs.groupby("quarter")[["dNDVI", "dNDBI"]].apply(lambda g: g.notna().mean())
        .round(3).to_string())
''')

# ============================== §5 產出 ==============================
md(r"""
---
## 5. 產出
""")

code(r'''
OUT_COLS = ["unit", "quarter", "縣市", "NDVI", "NDBI", "MNDWI", "dNDVI", "dNDBI",
            "valid_pixel_ratio", "n_pixel", "n_scene", "n_solar_day", "cloud_lt",
            "offset_applied"]
out = rs[OUT_COLS].copy()
out.to_csv(PROC / "unit_remote_sensing.csv", index=False, encoding="utf-8-sig")
print(f"已寫出 {PROC / 'unit_remote_sensing.csv'}（{len(out):,} 列）")

meta = {
    "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    "spatial_unit": SPATIAL_UNIT, "n_units": int(units.shape[0]),
    "imagery_freq": IMAGERY_FREQ, "structural_repeat_with_price": STRUCTURAL_REPEAT,
    "window": [Q_START, Q_END], "n_quarters": len(QUARTERS),
    "resolution_m": RESOLUTION, "crs": CRS,
    "collection": COLLECTION, "stac": STAC_URL,
    "scl_keep": list(SCL_KEEP), "cloud_steps": list(CLOUD_STEPS),
    "usable_window_note": "2017 各季的 solar day 數過少（ESA 當年對東亞取像計畫疏），"
                          "帶差分特徵的建模窗口建議從 2019Q1 起",
    "checks": checks,
}
(PROC / "rs_meta.json").write_text(
    json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"已寫出 {PROC / 'rs_meta.json'}")
''')

code(r'''
def _r(x) -> str:
    return "—" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{x:.3f}"


report = f"""# NB1 遙測指數抽取報告

產生時間：{meta['generated_at']}

分析單元：**{SPATIAL_UNIT}**（{len(units)} 個，粒度取自 NB0 的 `density_decision.json`）
影像：`{COLLECTION}`（Sentinel-2 L2A，Earth Search STAC，匿名存取，**不需要 GEE**）
解析度 {RESOLUTION} m｜CRS {CRS}｜窗口 {Q_START} ~ {Q_END}（{len(QUARTERS)} 季）

## 合理性檢查

| 層 | 結果 |
|---|---|
| I1（單元 × 季 完整） | 期望 {checks['I1']['n_expected']:,}、實得 {checks['I1']['n_have']:,}、缺 {checks['I1']['n_missing']} |
| I2（有效像素） | < {VPR_FLOOR} 的 cell {checks['I2']['n_low_cells']:,} 個（{checks['I2']['low_cell_share']:.1%}）｜不合格季 {len(checks['I2']['bad_quarters'])} 個 |
| I3（solar day） | 少於 {MIN_SOLAR_DAYS} 天的季 {len(checks['I3']['thin_quarters'])} 個：{checks['I3']['thin_quarters']} |
| I4（值域） | 超出 [-1,1]：{checks['I4']['out_of_range']} |
| I5（baseline 已知日期） | 穩定地物前後最大相對差 {_r(checks['I5'].get('worst_rel_diff'))}（門檻 {BASELINE_TOL}） |
| I6（季節性） | 夏 − 冬 = {checks['I6']['summer_minus_winter']:+.4f} |
| I7（跨來源空間相關） | {checks['I7'].get('spearman_vs_hh_density', checks['I7'].get('skipped'))} |
| **結論** | **{'✅ 通過' if checks['passed'] else '⚠️ 未通過'}** |

**I5 與 I7 是唯二有外部參照的檢查。** I5 的參照是 ESA 公布的 processing baseline
轉換日（2022-01-25，與我們的資料無關）；I7 的參照是戶政司的戶數（另一個資料源）。
其餘各層都是拿資料跟自己比，對「基準被污染」結構性無免疫——這一點跟 NB0 一樣。

### 抓不到什麼

{chr(10).join('- ' + x for x in checks['limitations'])}

## 給 NB2 / NB3 的提醒

- 房價粒度是「{TIME_UNIT}」、影像是「{IMAGERY_FREQ}」
  → 結構性重複 = **{STRUCTURAL_REPEAT}**。若為 True，NB3 的 paired t-test
  必須以「分析單元 × 季」為 pairing 單位（計畫 §5.3），不是 × 月。
- **2017 各季不可用**（solar day 過少），差分特徵從 2019Q1 起才齊全。
  計畫 §5.3 寫的 train 2017Q1–2022Q4 要改成 **2019Q1–2022Q4**。
- NB2 要用 `valid_pixel_ratio` 篩掉不可靠的 cell（計畫 §5.2 步驟 8：< 0.3）。
- NDBI / MNDWI 的有效解析度是 **20m**（B11 原生），不是 10m——報告中要如實說明。
"""
(PROC / "rs_report.md").write_text(report, encoding="utf-8")
print(f"已寫出 {PROC / 'rs_report.md'}")
print(report[:1200])
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
    "nbformat": 4,
    "nbformat_minor": 5,
}

try:
    import nbformat

    n_changes, nb = nbformat.validator.normalize(nb, version=4, version_minor=5)
    nbformat.validate(nb)
    print(f"nbformat: valid (normalize made {n_changes} fixes)")
except ImportError:
    print("WARNING: nbformat not installed — skipping validation")

out = (pathlib.Path(__file__).resolve().parent.parent
       / "notebooks" / "01_imagery.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"wrote {out}  cells={len(CELLS)}")
