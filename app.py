import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from utils import fuzzy_map_columns, detect_anomalies
from profit_engine import simulate_strategy

st.set_page_config(page_title="亚马逊AI运营策略回测引擎", layout="wide")
st.title("亚马逊AI运营策略回测与智能建议系统")

st.sidebar.header("数据上传")
uploaded_file = st.sidebar.file_uploader("上传产品历史表现数据（支持Excel/xlsx）", type=["xlsx"])

if uploaded_file:
    df_raw = pd.read_excel(uploaded_file)
    mapping = fuzzy_map_columns(df_raw.columns)
    df = df_raw.rename(columns=mapping)
    st.write("**字段映射后的数据预览：**")
    st.dataframe(df.head())

    # 检查异常
    anomalies = detect_anomalies(df)
    if not anomalies.empty:
        st.warning("检测到数据中可能存在异常波动（如促销、断货），建议检查这些行。")
        st.dataframe(anomalies)

    st.subheader("策略模拟与建议")
    # 策略一：销量优先
    with st.expander("销量优先策略（加速动销）"):
        result_vol = simulate_strategy(df, mode='volume')
        st.json(result_vol['params'])
        st.markdown(result_vol['explanation'])
        st.write(f"预计14天销量: {int(result_vol['pred_sales'])}, 预计利润: {result_vol['pred_profit']:.2f}")
        st.subheader("SHAP变量权重")
        shap_fig = result_vol['shap'].visualize()
        st.pyplot(shap_fig)

    # 策略二：利润最大化
    with st.expander("利润最大化策略"):
        result_profit = simulate_strategy(df, mode='profit')
        st.json(result_profit['params'])
        st.markdown(result_profit['explanation'])
        st.write(f"预计14天销量: {int(result_profit['pred_sales'])}, 预计利润: {result_profit['pred_profit']:.2f}")
        st.subheader("SHAP变量权重")
        shap_fig = result_profit['shap'].visualize()
        st.pyplot(shap_fig)

    # 历史/预测对比图
    st.subheader("利润与销量趋势")
    fig, ax = plt.subplots()
    df['利润预测'] = df['profit']
    pred_days = list(range(len(df), len(df)+14))
    profits = list(df['profit']) + [result_profit['pred_profit']] * 14
    ax.plot(list(range(len(profits))), profits, label="利润预测")
    ax.legend()
    st.pyplot(fig)
