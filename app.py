import streamlit as st

st.set_page_config(page_title="📊 Amazon Profit Optimizer", layout="wide")
st.title("🧠 Amazon Pro-Op AI Suite")
st.markdown("请上传你的运营数据（广告/价格/库存），系统将为你生成策略建议与回测报告")

uploaded_file = st.file_uploader("上传CSV或Excel文件", type=["csv", "xlsx"])
if uploaded_file:
    st.success("✅ 文件上传成功")
    st.write("📈 模拟展示图表：")
    st.line_chart([1, 3, 2, 4])
