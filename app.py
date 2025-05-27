# Streamlit 主入口文件
# 启动后展示上传控件、参数选择、回测按钮和建议卡片
if __name__ == "__main__":
    import streamlit as st
    st.set_page_config(page_title="Amazon Pro-Op AI", layout="wide")
    from streamlit.web import cli as stcli
    import sys
    sys.argv = ["streamlit", "run", "app.py", "--server.port=8000", "--server.address=0.0.0.0"]
    sys.exit(stcli.main())
