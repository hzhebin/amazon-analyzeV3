import streamlit as st
import pandas as pd
from optimizer import StrategyOptimizer

st.set_page_config(page_title="亚马逊智能策略系统", layout="wide")
st.title("📦 亚马逊销量/利润双策略建议系统")

uploaded_file = st.file_uploader("📤 上传产品表现表格 (Excel)", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # 字段映射
    rename_map = {
        '售价(总价)': '售价',
        'Sessions-Total': 'Sessions'
    }
    df.rename(columns=rename_map, inplace=True)

    optimizer = StrategyOptimizer()
    try:
        optimizer.train(df)
        st.success("✅ 模型训练完成，生成策略中...")

        strategy1 = optimizer.optimize_sales()
        strategy2 = optimizer.optimize_profit()

        st.subheader("🚀 📦 销量优先策略")
        st.json(strategy1)

        st.subheader("🚀 💰 利润最大化策略")
        st.json(strategy2)

    except Exception as e:
        st.error(f"❌ 模型训练失败: {e}")
else:
    st.info("请上传历史表现 Excel 表开始回测策略")
