"""TaiwanTS MVP demo dashboard（計畫 §6）

尚未實作。等 NB3 的模型結果與 NB4 的 VLM 輸出齊備後再建。

規劃功能：
1. 六都地圖（Folium），分析單元著色 = 房價漲幅
2. 點擊分析單元 → 房價時序、NDVI/NDBI 疊圖、預測 vs 實際、SHAP waterfall
3. VLM 變遷總覽熱力圖
4. Highlight：3-5 個最佳案例（2017 vs 2024 影像並排 + 變遷描述 + 漲幅）

本機執行： streamlit run app/streamlit_app.py
"""

import streamlit as st

st.set_page_config(page_title="TaiwanTS MVP", layout="wide")
st.title("TaiwanTS — 衛星影像 × 台灣房價")
st.info("尚未實作。請先依序完成 NB0 → NB4。")
