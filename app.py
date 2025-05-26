
import streamlit as st
import pandas as pd
from optimizer import OperatorStrategyEngine

st.set_page_config(page_title="亚马逊顶尖运营AI策略引擎", layout="wide")
st.title("🧠 亚马逊顶级运营AI策略系统 v4.0")

uploaded_file = st.file_uploader("📤 上传产品表现表格 (Excel)", type=["xlsx"])
inventory_qty = st.number_input("现有库存（件）", value=100, step=10)
target_days = st.number_input("目标动销天数", value=30, step=1)

if uploaded_file and inventory_qty and target_days:
    df = pd.read_excel(uploaded_file)
    engine = OperatorStrategyEngine(inventory_qty, target_days)
    engine.train(df)
    st.success("✅ 智能AI模型已训练完毕，开始生成策略...")

    with st.spinner("策略生成中..."):
        res = engine.optimize()

    st.subheader("🚀 智能策略建议卡")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 推荐组合")
        for k, v in res["建议参数"].items():
            st.write(f"**{k}**: {v:.2f}" if isinstance(v, float) else f"**{k}**: {v}")
        st.write("#### 预计指标")
        st.metric("14天销量", f"{res['预测14天销量']:.1f} 单")
        st.metric("14天利润", f"${res['预测14天利润']:.2f}")
        st.metric("动销周转天数", f"{res['预测库存周转天数']:.1f} 天")
        st.metric("目标动销天数", f"{res['目标周转天数']} 天")
        st.metric("广告ACoAS", f"{res['广告指标']['ACoAS']*100:.2f}%")
        st.metric("广告ASoAS", f"{res['广告指标']['ASoAS']*100:.2f}%")
        st.metric("广告CPM", f"${res['广告指标']['CPM']:.2f}")
        st.metric("广告CPO", f"${res['广告指标']['CPO']:.2f}")

    with col2:
        st.markdown("#### 策略逻辑 & 解释")
        st.info(res["策略解释"])
        st.warning(res["落地建议"])

else:
    st.info("请上传Excel并输入库存、动销天数。")
