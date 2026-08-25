# -*- coding: utf-8 -*-
"""產生 notebooks/03_modeling.ipynb"""
import hashlib, json, pathlib

CELLS = []
def md(s):   CELLS.append(("markdown", s.strip("\n")))
def code(s): CELLS.append(("code", s.strip("\n")))

md(r"""
# NB3 — A vs B：影像特徵有沒有增量

規格：`taiwan-housets-mvp-plan_3.md` 計畫 §5.3、§1。**這本 notebook 就是專案要的答案。**

## 假設與成功標準（計畫 §1，逐字）

> 假設：Sentinel-2 遙測指數（NDVI、NDBI）加入 XGBoost 後，房價預測 MAPE 可降低。

1. 六都測試集整體 MAPE 改善 **≥ 2 個絕對百分點**
2. 至少 **3 個都市區**達到統計顯著改善（paired t-test, p < 0.05；
   pairing 單位為「分析單元 × 季」，**非 × 月**）
3. SHAP 分析中，遙測特徵至少有 **1 項**進入前 10 重要特徵

## 這本 notebook 最重要的設計決定：**先假設答案是「沒有增量」**

一個「MAPE 改善了 2.3 個百分點」的數字，有太多種方式可以是假的：

| 假陽性來源 | 這裡怎麼擋 |
|---|---|
| 單一次 train/test 的隨機性 | **多個 seed**（`N_SEEDS`），報分布不報單一數字 |
| 「多了 5 個特徵」本身就會讓樹模型變好 | **負對照 N1**：把影像特徵換成同維度的**隨機雜訊** |
| 影像特徵其實只是單元的固定效果（不是時間訊號） | **負對照 N2**：在單元內把影像特徵**時間打散** |
| 季度影像 × 月度房價的結構性重複壓低 p-value | pairing 用「單元 × 季」；**N3** 直接用季粒度重跑 |
| 被省略的交通特徵讓影像代它領功 | **N4**：排除窗口內有新站通車的單元（NB2 已記錄這個偏誤） |
| 所得遞延造成的偽訊號 | **N5**：排除所得被遞延的年份 |

**負對照 N1／N2 是這裡最關鍵的兩項。** 如果隨機雜訊也能讓 MAPE 改善 2 個百分點，
那第 1 條成功標準本身就沒有意義；如果打散後改善還在，那影像帶的是空間固定效果，
不是計畫假設的「都市變遷訊號」。
""")

md(r"""
---
## 0. 環境與上游產出
""")

code(r"""
# Colab 執行時取消註解
# !pip install -q xgboost lightgbm scikit-learn shap pyarrow
""")

code(r'''
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
pd.set_option("display.width", 190)
pd.set_option("display.max_columns", 80)

try:
    import google.colab  # noqa: F401
    from google.colab import drive

    drive.mount("/content/drive", force_remount=False)
    PROJECT = Path("/content/drive/MyDrive/TW_Housets")
except ImportError:
    _cwd = Path.cwd()
    PROJECT = _cwd.parent if _cwd.name == "notebooks" else _cwd

PROC = PROJECT / "data" / "processed"
NEED = {"NB2 大表": PROC / "modeling_table.parquet",
        "NB2 中介資料": PROC / "merge_meta.json"}
_miss = {k: v for k, v in NEED.items() if not v.exists()}
if _miss:
    raise FileNotFoundError("缺少上游產出：\n"
                            + "\n".join(f"  {k}：{v}" for k, v in _miss.items()))

MM = json.loads(NEED["NB2 中介資料"].read_text(encoding="utf-8"))
if not MM["checks"]["passed"]:
    print("⚠️  NB2 的合理性檢查沒有全部通過——先看 merge_meta.json 的 checks 再繼續。")
print(f"粒度：{MM['scheme']}｜結構性重複：{MM['structural_repeat_with_price']}")
print("NB2 記錄的已知偏誤：")
for b in MM["known_biases"]:
    print(f"  · {b}")

tbl = pd.read_parquet(NEED["NB2 大表"])
tbl["period"] = pd.PeriodIndex(tbl["period"], freq="M" if MM["time_unit"] == "月" else "Q")
tbl["quarter"] = pd.PeriodIndex(tbl["quarter"], freq="Q")
tbl = tbl[tbl["keep"]].copy()
print(f"\n大表：{len(tbl):,} 列（單元 {tbl.unit.nunique()}、期 {tbl.period.nunique()}）")

FEATS_A = [c for c in MM["features_A"] if c in tbl.columns]
IMG_FEATS = [c for c in MM["features_B"] if c not in MM["features_A"] and c in tbl.columns]
FEATS_B = FEATS_A + IMG_FEATS
TARGET = MM["target"]
print(f"特徵集 A：{len(FEATS_A)} 個｜影像特徵：{IMG_FEATS}")
''')

md(r"""
---
## 1. 時間切分

計畫 §5.3 寫 train 2017Q1–2022Q4。**那個起點用不了**——NB1 實測 2017 年台灣沒有
可用的 Sentinel-2（2017Q2 連一張雲量 < 40% 的都沒有），而年際差分還要再往後一年。
所以起點改成 **2019Q1**，並且**下限由資料自己決定**：取「影像差分特徵齊全」的最早一季。
""")

code(r'''
# 差分特徵齊全的最早一季（不硬編，讓資料決定）
_ok = tbl.dropna(subset=[c for c in IMG_FEATS if c.startswith("d")])
TRAIN_START = _ok["quarter"].min() if len(_ok) else tbl["quarter"].min()
TRAIN_END, VAL_END, TEST_END = (pd.Period("2022Q4", "Q"), pd.Period("2023Q2", "Q"),
                                pd.Period("2024Q4", "Q"))
print(f"差分特徵齊全的最早一季：{TRAIN_START}（計畫寫 2017Q1，實際用不了）")

def split_of(q):
    if q < TRAIN_START:
        return "unusable"
    if q <= TRAIN_END:
        return "train"
    if q <= VAL_END:
        return "val"
    if q <= TEST_END:
        return "test"
    return "future"

tbl["split"] = tbl["quarter"].map(split_of)
print(tbl.groupby("split").agg(列數=("unit", "size"), 單元=("unit", "nunique"),
                               季=("quarter", "nunique")).to_string())
data = tbl[tbl["split"].isin(["train", "val", "test"])].copy()
# 影像差分在最早一年是 NaN；樹模型吃得下 NaN，但要確認不是整欄都缺
_nan = data[IMG_FEATS].isna().mean()
print(f"\n影像特徵的 NaN 佔比：{_nan.round(3).to_dict()}")
if (_nan > 0.5).any():
    raise RuntimeError(f"有影像特徵超過一半是 NaN：{_nan[_nan > 0.5].to_dict()}——"
                       "切分起點或 NB1 的抽取有問題。")
''')

md(r"""
---
## 2. 模型設定

**A 與 B 用完全相同的超參數與相同的 seed。** 不對 B 額外調參——否則比較的是
「調參的成果」而不是「影像特徵的增量」。多個 seed 是為了量出 run 之間的變異：
一個 2 個百分點的差距如果小於 seed 之間的標準差，那它就不是結果。
""")

code(r'''
import xgboost as xgb

N_SEEDS = 10
PARAMS = dict(n_estimators=600, max_depth=6, learning_rate=0.05,
              subsample=0.8, colsample_bytree=0.8,
              min_child_weight=5, reg_lambda=1.0,
              objective="reg:squarederror", tree_method="hist")
EARLY_STOP = 50


def mape(y, p):
    return float(np.mean(np.abs((y - p) / y)) * 100)


def mdape(y, p):
    return float(np.median(np.abs((y - p) / y)) * 100)


def fit_predict(feats, seed: int, d=None, log_target: bool = True):
    """回傳 (test 的預測, 用到的模型)。log 目標讓誤差在比例上對稱（房價是右偏的）。"""
    d = data if d is None else d
    tr, va, te = (d[d.split == s] for s in ("train", "val", "test"))
    ytr = np.log(tr[TARGET]) if log_target else tr[TARGET]
    yva = np.log(va[TARGET]) if log_target else va[TARGET]
    m = xgb.XGBRegressor(**PARAMS, random_state=seed,
                         early_stopping_rounds=EARLY_STOP)
    m.fit(tr[feats], ytr, eval_set=[(va[feats], yva)], verbose=False)
    p = m.predict(te[feats])
    return (np.exp(p) if log_target else p), m


def run_ab(feats_a, feats_b, d=None, n_seeds=N_SEEDS, tag=""):
    """同一組 seed 跑 A 與 B，回傳逐 seed 的指標與逐列誤差。"""
    d = data if d is None else d
    te = d[d.split == "test"]
    rows, errs = [], []
    for s in range(n_seeds):
        pa, _ = fit_predict(feats_a, s, d)
        pb, _ = fit_predict(feats_b, s, d)
        y = te[TARGET].to_numpy()
        rows.append({"seed": s, "MAPE_A": mape(y, pa), "MAPE_B": mape(y, pb),
                     "MdAPE_A": mdape(y, pa), "MdAPE_B": mdape(y, pb)})
        errs.append(pd.DataFrame({
            "unit": te["unit"].to_numpy(), "縣市": te["縣市"].to_numpy(),
            "quarter": te["quarter"].astype(str).to_numpy(), "seed": s,
            "ape_A": np.abs((y - pa) / y) * 100, "ape_B": np.abs((y - pb) / y) * 100}))
    r = pd.DataFrame(rows)
    r["ΔMAPE"] = r["MAPE_A"] - r["MAPE_B"]          # 正值 = B 比較好
    if tag:
        print(f"[{tag}] MAPE_A {r.MAPE_A.mean():.2f}±{r.MAPE_A.std():.2f}"
              f"｜MAPE_B {r.MAPE_B.mean():.2f}±{r.MAPE_B.std():.2f}"
              f"｜ΔMAPE {r['ΔMAPE'].mean():+.2f}±{r['ΔMAPE'].std():.2f} pp")
    return r, pd.concat(errs, ignore_index=True)
''')

md(r"""
---
## 3. 主結果：A vs B
""")

code(r'''
res, errs = run_ab(FEATS_A, FEATS_B, tag="主結果")
print()
print(res.round(3).to_string(index=False))
print(f"\n測試集：{int((data.split == 'test').sum()):,} 列"
      f"（{data[data.split == 'test'].unit.nunique()} 單元 × "
      f"{data[data.split == 'test'].quarter.nunique()} 季）")

dm, ds = res["ΔMAPE"].mean(), res["ΔMAPE"].std()
print(f"\nΔMAPE（A − B，正值代表影像有幫助）：{dm:+.3f} pp，"
      f"seed 間標準差 {ds:.3f} pp")
if abs(dm) < ds:
    print("  ⚠️  平均差距小於 seed 間的標準差——這個差距**不能當成結果**。")
crit1 = bool(dm >= 2.0)
print(f"成功標準 1（ΔMAPE ≥ 2 pp）：{'✅ 達成' if crit1 else '❌ 未達成'}")
''')

code(r'''
# ---- 逐縣市 ----
by_city = (errs.groupby(["縣市", "seed"])[["ape_A", "ape_B"]].mean()
               .groupby("縣市").mean())
by_city["ΔMAPE"] = by_city["ape_A"] - by_city["ape_B"]
CITY_ORDER = ["臺北市", "新北市", "桃園市", "臺中市", "臺南市", "高雄市"]
print("逐縣市 MAPE（跨 seed 平均）")
print(by_city.reindex([c for c in CITY_ORDER if c in by_city.index])
             .round(2).to_string())
''')

md(r"""
---
## 4. 統計檢定：pairing 單位是「單元 × 季」

計畫 §5.3 說得很清楚：若房價是月粒度，同季 3 個月共用同一組遙測值，
以「單元 × 月」為 pair 會讓 3 個 pair 不獨立、**p-value 被人為壓低**。
所以先把同季各月的誤差平均，再做 paired t-test。

下面同時報「× 月」的 p-value 作為對照——不是要用它，是要**看出它有多樂觀**。
""")

code(r'''
def paired_test(e: pd.DataFrame, by: list) -> dict:
    """先在 by 這個粒度上把誤差平均（跨 seed 也平均），再做 paired t-test。"""
    g = e.groupby(by)[["ape_A", "ape_B"]].mean()
    d = (g["ape_A"] - g["ape_B"]).dropna()
    if len(d) < 3:
        return {"n_pair": len(d), "mean_diff": np.nan, "t": np.nan, "p": np.nan}
    t, p = stats.ttest_rel(g["ape_A"], g["ape_B"])
    w = stats.wilcoxon(g["ape_A"], g["ape_B"])[1] if len(d) >= 10 else np.nan
    return {"n_pair": int(len(d)), "mean_diff": float(d.mean()),
            "t": float(t), "p": float(p), "wilcoxon_p": float(w)}


overall_q = paired_test(errs, ["unit", "quarter"])
overall_m = paired_test(errs, ["unit", "quarter", "seed"])
print("整體 paired t-test")
print(f"  pairing = 單元 × 季（計畫指定）：n={overall_q['n_pair']}, "
      f"mean diff {overall_q['mean_diff']:+.3f} pp, p={overall_q['p']:.4g}, "
      f"Wilcoxon p={overall_q['wilcoxon_p']:.4g}")
print(f"  pairing = 單元 × 季 × seed（**錯的**，seed 之間不獨立）："
      f"n={overall_m['n_pair']}, p={overall_m['p']:.4g}")
print("  ↑ 兩者的 n 差 10 倍、p 差好幾個數量級。這就是為什麼 pairing 單位要固定。")

rows = []
for city, g in errs.groupby("縣市"):
    r = paired_test(g, ["unit", "quarter"])
    r["縣市"] = city
    r["顯著改善"] = bool(r["p"] < 0.05 and r["mean_diff"] > 0)
    rows.append(r)
city_test = pd.DataFrame(rows).set_index("縣市").reindex(
    [c for c in CITY_ORDER if c in {r["縣市"] for r in rows}])
print("\n逐縣市 paired t-test（pairing = 單元 × 季）")
print(city_test.round(4).to_string())
n_sig = int(city_test["顯著改善"].sum())
crit2 = bool(n_sig >= 3)
print(f"\n成功標準 2（≥ 3 個縣市顯著改善）：{n_sig} 個 → "
      f"{'✅ 達成' if crit2 else '❌ 未達成'}")
''')

md(r"""
---
## 5. SHAP：只解讀符號與排名

計畫 §5.3 說明得很清楚：季度影像 × 月度房價的結構性重複會讓 SHAP 的**量級**偏誤，
而偏誤方向不確定（樣本重複加重分裂權重 → 偏高；季內無變異 → 偏低）。
所以**只看影像特徵有沒有進前 10、以及方向**，不解讀數值大小。
""")

code(r'''
import shap

_, model_b = fit_predict(FEATS_B, 0)
te = data[data.split == "test"]
expl = shap.TreeExplainer(model_b)
sv = expl.shap_values(te[FEATS_B])
imp = (pd.DataFrame({"feature": FEATS_B,
                     "mean_abs_shap": np.abs(sv).mean(axis=0)})
       .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))
imp["rank"] = imp.index + 1
imp["是影像特徵"] = imp["feature"].isin(IMG_FEATS)
print("SHAP 重要性排名（前 15）")
print(imp.head(15).to_string(index=False))

img_ranks = imp.loc[imp["是影像特徵"], ["feature", "rank"]]
crit3 = bool((img_ranks["rank"] <= 10).any())
print(f"\n影像特徵的排名：{img_ranks.set_index('feature')['rank'].to_dict()}")
print(f"成功標準 3（≥ 1 項影像特徵進前 10）：{'✅ 達成' if crit3 else '❌ 未達成'}")

# 方向：SHAP 與特徵值的相關符號
sign = {}
for f in IMG_FEATS:
    j = FEATS_B.index(f)
    x = te[f].to_numpy()
    ok = np.isfinite(x)
    sign[f] = (round(float(np.corrcoef(x[ok], sv[ok, j])[0, 1]), 3)
               if ok.sum() > 10 else None)
print(f"影像特徵的作用方向（SHAP 與特徵值的相關）：{sign}")
''')

md(r"""
---
## 6. 負對照與 robustness

**這一節比第 3 節重要。** 第 3 節只告訴我們「有沒有差」，這一節告訴我們
「那個差是不是影像帶來的」。
""")

code(r'''
checks = {}

# ---- N1：隨機雜訊對照 ----
# 把影像特徵換成同維度的隨機雜訊。樹模型多拿到幾個特徵本身就可能讓 MAPE 稍好
# （更多分裂機會）。若雜訊也能改善接近 2 pp，那成功標準 1 本身就沒有意義。
rng = np.random.default_rng(0)
d_noise = data.copy()
noise_cols = []
for i, f in enumerate(IMG_FEATS):
    c = f"noise_{i}"
    d_noise[c] = rng.standard_normal(len(d_noise))
    noise_cols.append(c)
res_n1, _ = run_ab(FEATS_A, FEATS_A + noise_cols, d=d_noise, n_seeds=5, tag="N1 隨機雜訊")
checks["N1_noise"] = {"dMAPE_mean": round(float(res_n1["ΔMAPE"].mean()), 3),
                      "dMAPE_sd": round(float(res_n1["ΔMAPE"].std()), 3)}
''')

code(r'''
# ---- N2：影像特徵在單元內時間打散 ----
# 保留每個單元的影像分布（也就是保留空間固定效果），只破壞時間對應。
# 若改善還在，那影像帶的是「哪個區」而不是「這個區在變」——後者才是計畫的假設。
d_shuf = data.copy()
rs2 = np.random.default_rng(1)
for f in IMG_FEATS:
    d_shuf[f] = (d_shuf.groupby("unit")[f]
                       .transform(lambda s: rs2.permutation(s.to_numpy())))
res_n2, _ = run_ab(FEATS_A, FEATS_B, d=d_shuf, n_seeds=5, tag="N2 單元內時間打散")
checks["N2_shuffled"] = {"dMAPE_mean": round(float(res_n2["ΔMAPE"].mean()), 3),
                         "dMAPE_sd": round(float(res_n2["ΔMAPE"].std()), 3)}
''')

code(r'''
# ---- N3：季粒度重跑（消除結構性重複）----
if MM["structural_repeat_with_price"]:
    q = (data.groupby(["unit", "縣市", "quarter", "split"], observed=True)
             .agg({TARGET: "median", **{c: "mean" for c in FEATS_B if c in data.columns}})
             .reset_index())
    q["quarter"] = pd.PeriodIndex(q["quarter"].astype(str), freq="Q")
    res_n3, errs_n3 = run_ab(FEATS_A, FEATS_B, d=q, n_seeds=5, tag="N3 季粒度")
    t_n3 = paired_test(errs_n3, ["unit", "quarter"])
    checks["N3_quarterly"] = {"dMAPE_mean": round(float(res_n3["ΔMAPE"].mean()), 3),
                              "dMAPE_sd": round(float(res_n3["ΔMAPE"].std()), 3),
                              "p": None if pd.isna(t_n3["p"]) else round(t_n3["p"], 5)}
    print(f"  季粒度的 paired t-test：n={t_n3['n_pair']}, p={t_n3['p']:.4g}")
else:
    checks["N3_quarterly"] = {"skipped": "房價本來就是季粒度"}
    print("N3 跳過：房價本來就是季粒度，沒有結構性重複")
''')

code(r'''
# ---- N4：排除窗口內有新捷運站的單元 ----
# NB2 記錄的偏誤：OSM 車站的 start_date 覆蓋率只有 11%，所以「通車前後」特徵做不出來。
# 這裡用**能辨識的那部分**把有新站的單元排除掉——不完整，但方向明確：
# 若影像的增益只存在於有新站的單元，那增益很可能是被省略的交通特徵。
POI_DIR = PROJECT / "data" / "raw" / "poi"
new_station_units = set()
if POI_DIR.exists():
    import geopandas as gpd
    from shapely.geometry import Point

    _bnd = next(iter(sorted((PROJECT / "data" / "raw" / "boundary").rglob("*.shp"))), None)
    _poi = pd.concat([pd.read_parquet(p) for p in sorted(POI_DIR.glob("*.parquet"))],
                     ignore_index=True) if any(POI_DIR.glob("*.parquet")) else None
    if _bnd is not None and _poi is not None:
        st = _poi[(_poi["kind"] == "station") & _poi["start_date"].notna()].copy()
        st["y"] = pd.to_numeric(st["start_date"].str.slice(0, 4), errors="coerce")
        st = st[(st["y"] >= data["quarter"].min().year) & (st["y"] <= TEST_END.year)]
        if len(st):
            g = gpd.read_file(_bnd).to_crs(4326)
            g["unit"] = g["COUNTYNAME"] + g["TOWNNAME"]
            gs = gpd.GeoDataFrame(st, geometry=[Point(xy) for xy in zip(st.lon, st.lat)],
                                  crs=4326)
            j = gpd.sjoin(gs, g[["unit", "geometry"]], how="left", predicate="within")
            new_station_units = set(j["unit"].dropna())
print(f"窗口內有新站（OSM 有 start_date 的部分）的單元：{len(new_station_units)} 個")
if new_station_units:
    d_n4 = data[~data["unit"].isin(new_station_units)]
    res_n4, _ = run_ab(FEATS_A, FEATS_B, d=d_n4, n_seeds=5, tag="N4 排除新站單元")
    checks["N4_no_new_station"] = {
        "n_units_excluded": len(new_station_units),
        "dMAPE_mean": round(float(res_n4["ΔMAPE"].mean()), 3),
        "dMAPE_sd": round(float(res_n4["ΔMAPE"].std()), 3)}
else:
    checks["N4_no_new_station"] = {"skipped": "OSM 沒有可用的 start_date"}
    print("N4 跳過：OSM 沒有可用的 start_date（這一項的偏誤因此沒有被排除，"
          "要寫在結論的限制裡）")
''')

code(r'''
# ---- N5：排除所得被遞延的年份 ----
if "income_year_used" in data.columns and (data["income_year_used"] != data["year"]).any():
    d_n5 = data[data["income_year_used"] == data["year"]]
    if (d_n5["split"] == "test").sum() > 50:
        res_n5, _ = run_ab(FEATS_A, FEATS_B, d=d_n5, n_seeds=5, tag="N5 排除所得遞延年")
        checks["N5_income_actual"] = {
            "dMAPE_mean": round(float(res_n5["ΔMAPE"].mean()), 3),
            "dMAPE_sd": round(float(res_n5["ΔMAPE"].std()), 3)}
    else:
        checks["N5_income_actual"] = {"skipped": "排除後測試集太小"}
        print("N5 跳過：排除遞延年之後測試集太小")
else:
    checks["N5_income_actual"] = {"skipped": "沒有遞延"}
    print("N5 跳過：所得沒有被遞延")
''')

md(r"""
---
## 7. 結論
""")

code(r'''
NOISE_TOL = 0.5      # 隨機雜訊帶來的 ΔMAPE 超過這個值，就代表「多特徵」本身有效果
main_d = float(res["ΔMAPE"].mean())
main_sd = float(res["ΔMAPE"].std())
noise_d = checks["N1_noise"]["dMAPE_mean"]
shuf_d = checks["N2_shuffled"]["dMAPE_mean"]

# 扣掉「多特徵」的效果之後，影像真正的增量
net_d = main_d - noise_d
# 扣掉「空間固定效果」之後，影像的**時間**訊號
time_d = main_d - shuf_d

verdict = {
    "criteria": {
        "1_mape_ge_2pp": {"target": 2.0, "actual": round(main_d, 3),
                          "sd_across_seeds": round(main_sd, 3), "met": crit1},
        "2_cities_significant": {"target": 3, "actual": n_sig, "met": crit2},
        "3_shap_top10": {"met": crit3,
                         "img_ranks": img_ranks.set_index("feature")["rank"].to_dict()},
    },
    "negative_controls": checks,
    "derived": {
        "net_of_noise_pp": round(net_d, 3),
        "time_signal_pp": round(time_d, 3),
        "exceeds_seed_noise": bool(abs(main_d) > main_sd),
    },
}
all_met = crit1 and crit2 and crit3
controls_ok = bool(noise_d < NOISE_TOL and shuf_d < main_d - NOISE_TOL)
verdict["overall"] = ("SUPPORTED" if all_met and controls_ok else
                      "NOT_SUPPORTED" if not any((crit1, crit2, crit3)) else
                      "PARTIAL")
verdict["controls_ok"] = controls_ok

bar = "=" * 78
print(bar)
print("  假設：Sentinel-2 遙測指數加入 XGBoost 後，房價預測 MAPE 可降低")
print(bar)
print(f"  MAPE_A  {res.MAPE_A.mean():6.2f} ± {res.MAPE_A.std():.2f}")
print(f"  MAPE_B  {res.MAPE_B.mean():6.2f} ± {res.MAPE_B.std():.2f}")
print(f"  ΔMAPE   {main_d:+6.3f} ± {main_sd:.3f} pp"
      f"   {'（大於 seed 雜訊）' if abs(main_d) > main_sd else '（**小於 seed 雜訊**）'}")
print()
print(f"  成功標準 1（ΔMAPE ≥ 2 pp）        ：{'✅' if crit1 else '❌'}")
print(f"  成功標準 2（≥ 3 縣市顯著）        ：{'✅' if crit2 else '❌'}  實得 {n_sig}")
print(f"  成功標準 3（影像進 SHAP 前 10）   ：{'✅' if crit3 else '❌'}")
print()
print(f"  負對照 N1 隨機雜訊  ΔMAPE {noise_d:+.3f} pp"
      f"   → 扣掉後影像的淨增量 {net_d:+.3f} pp")
print(f"  負對照 N2 時間打散  ΔMAPE {shuf_d:+.3f} pp"
      f"   → 其中屬於**時間訊號**的 {time_d:+.3f} pp")
print(f"  負對照是否過關：{'✅' if controls_ok else '❌ —— 主結果不可信'}")
print()
print(f"  結論：**{verdict['overall']}**")
print(bar)

(PROC / "modeling_result.json").write_text(
    json.dumps({"generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "scheme": MM["scheme"], "n_seeds": N_SEEDS,
                "split": {"train_start": str(TRAIN_START), "train_end": str(TRAIN_END),
                          "val_end": str(VAL_END), "test_end": str(TEST_END)},
                "n_test_rows": int((data.split == "test").sum()),
                "features_A": FEATS_A, "features_img": IMG_FEATS,
                "mape": {"A_mean": round(float(res.MAPE_A.mean()), 3),
                         "A_sd": round(float(res.MAPE_A.std()), 3),
                         "B_mean": round(float(res.MAPE_B.mean()), 3),
                         "B_sd": round(float(res.MAPE_B.std()), 3)},
                "by_city": by_city.round(3).to_dict("index"),
                "city_test": city_test.round(5).to_dict("index"),
                "overall_paired_test_quarter": {
                    k: (None if pd.isna(v) else v) for k, v in overall_q.items()},
                "verdict": verdict,
                "inherited_biases": MM["known_biases"]},
               ensure_ascii=False, indent=1), encoding="utf-8")
print(f"已寫出 {PROC / 'modeling_result.json'}")
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
    print("WARNING: nbformat not installed")

out = (pathlib.Path(__file__).resolve().parent.parent
       / "notebooks" / "03_modeling.ipynb")
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"wrote {out}  cells={len(CELLS)}")
