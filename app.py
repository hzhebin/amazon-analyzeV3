
import streamlit as st
import pandas as pd
from optimizer import StrategyOptimizer

st.set_page_config(page_title="亚马逊顶级运营AI策略系统 v4.0", layout="wide")
st.title("🧠 亚马逊顶级运营AI策略系统 v4.0")

def load_first_dataframe(uploaded_file):
    df_or_dict = pd.read_excel(uploaded_file, sheet_name=None)
    if isinstance(df_or_dict, dict):
        for df in df_or_dict.values():
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df
        raise ValueError("没有非空的Sheet")
    return df_or_dict

uploaded_file = st.file_uploader("📤 上传产品表现表格 (Excel)", type=["xlsx"])
inventory_qty = st.number_input("现有库存（件）", value=100, step=10)
target_days = st.number_input("目标动销天数", value=30, step=1)

if uploaded_file and inventory_qty and target_days:
    try:
        df = load_first_dataframe(uploaded_file)
        st.success("✅ 文件读取成功，准备分析 ...")

        optimizer = StrategyOptimizer(df, inventory_qty, target_days)
        sales_strategy, profit_strategy = optimizer.optimize_strategies()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📦 销量优先策略")
            for k, v in sales_strategy.items():
                st.write(f"**{k}**: {v}")

        with col2:
            st.subheader("🚀 利润最大化策略")
            for k, v in profit_strategy.items():
                st.write(f"**{k}**: {v}")

    except Exception as e:
        st.error(f"❌ 模型运行失败，请检查数据表字段或联系技术支持。\n详细错误: {e}")
