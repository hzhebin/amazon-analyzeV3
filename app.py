import streamlit as st

# ⚠️ 第一个 Streamlit 调用必须是 set_page_config
st.set_page_config(page_title="Amazon Pro-Op AI", layout="wide")

st.title("📊 Amazon Profit Optimization AI Suite")
st.markdown("上传广告/价格/库存表现表格，生成回测结果与优化建议")

uploaded_file = st.file_uploader("上传产品表现表", type=["csv", "xlsx"])
if uploaded_file:
    st.success("✅ 文件上传成功")
    st.write("📈 示例图表如下：")
    st.line_chart([1, 3, 2, 4])
