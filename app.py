import streamlit as st
import pandas as pd
from xgb_model import XGBModel

st.set_page_config(page_title="亚马逊智能利润优化系统", layout="wide")
st.title("📈 亚马逊利润最大化分析工具")

uploaded_file = st.file_uploader("📤 上传产品表现表格 (Excel)", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    model = XGBModel()
    try:
        model.train(df)
        st.success("✅ 模型训练成功，可进行分析与解释")

        with st.expander("📊 SHAP 变量解释分析"):
            if st.button("显示变量影响力图 (Beeswarm)"):
                model.shap_summary_plot()
            if st.button("显示变量影响力图 (Bar)"):
                model.shap_bar_plot()

        st.subheader("🧠 策略建议模拟")
        row = df.iloc[-1]  # 用最后一条数据做参考点
        current = {k: row[k] for k in model.features}

        with st.form("模拟器"):
            new_price = st.number_input("模拟售价", value=float(current["售价"]))
            new_cpc = st.number_input("模拟CPC", value=float(current["CPC"]))
            new_ad = st.number_input("模拟广告花费", value=float(current["广告花费"]))
            new_acos = st.number_input("模拟ACOS", value=float(current["ACOS"]))
            new_sessions = st.number_input("模拟流量", value=float(current["Sessions"]))
            new_cvr = st.number_input("模拟CVR", value=float(current["CVR"]))
            submitted = st.form_submit_button("提交模拟")
            if submitted:
                vars_input = {
                    "售价": new_price,
                    "CPC": new_cpc,
                    "广告花费": new_ad,
                    "ACOS": new_acos,
                    "Sessions": new_sessions,
                    "CVR": new_cvr
                }
                pred_orders = model.predict_sales(vars_input)
                pred_profit = model.predict_profit(vars_input)
                st.metric("📦 预测订单数", f"{pred_orders:.1f} 单")
                st.metric("💰 预测利润", f"${pred_profit:,.2f}")

    except Exception as e:
        st.error(f"❌ 模型训练失败: {e}")
else:
    st.info("请上传产品数据 Excel 文件以开始分析")
