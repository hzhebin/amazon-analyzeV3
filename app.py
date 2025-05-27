# Streamlit 主入口文件
# 启动后展示上传控件、参数选择、回测按钮和建议卡片
import streamlit as st

st.set_page_config(page_title="Amazon Pro-Op AI", layout="wide")

st.title("📊 Amazon Profit Optimization AI Suite")
st.markdown("上传广告/价格/库存表现表格，生成回测结果与优化建议")

uploaded_file = st.file_uploader("上传产品表现表", type=["csv", "xlsx"])

if uploaded_file:
    st.success("文件已上传 ✅")
    # 模拟调用后端模块
    st.write("正在分析数据...")
    st.line_chart([1, 3, 2, 4])  # 测试可视化
if __name__ == "__main__":
    import streamlit as st
    st.set_page_config(page_title="Amazon Pro-Op AI", layout="wide")
    from streamlit.web import cli as stcli
    import sys
    sys.argv = ["streamlit", "run", "app.py", "--server.port=8000", "--server.address=0.0.0.0"]
    sys.exit(stcli.main())
