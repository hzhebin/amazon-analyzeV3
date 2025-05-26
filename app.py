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

        strategy_sales = optimizer.optimize_sales()
        strategy_profit = optimizer.optimize_profit()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🚀 📦 销量优先策略")
            st.metric("建议售价", f"${strategy_sales['建议参数']['售价']:.2f}")
            st.metric("建议 CPC", f"${strategy_sales['建议参数']['CPC']:.2f}")
            st.metric("广告花费", f"${strategy_sales['建议参数']['广告花费']:.2f}")
            st.metric("Sessions", f"{strategy_sales['建议参数']['Sessions']:.1f}")
            st.metric("CVR", f"{strategy_sales['建议参数']['CVR']*100:.2f}%")
            st.metric("14天预计销量", f"{strategy_sales['预计14天销量']} 单")
            st.metric("14天预计利润", f"${strategy_sales['预计14天利润']:.2f}")
            st.info(f"📌 执行建议：{strategy_sales['执行建议']}")
            st.markdown("#### 🤖 策略说明")
            st.markdown(strategy_sales["解释说明"])

        with col2:
            st.markdown("### 🚀 💰 利润最大化策略")
            st.metric("建议售价", f"${strategy_profit['建议参数']['售价']:.2f}")
            st.metric("建议 CPC", f"${strategy_profit['建议参数']['CPC']:.2f}")
            st.metric("广告花费", f"${strategy_profit['建议参数']['广告花费']:.2f}")
            st.metric("Sessions", f"{strategy_profit['建议参数']['Sessions']:.1f}")
            st.metric("CVR", f"{strategy_profit['建议参数']['CVR']*100:.2f}%")
            st.metric("14天预计销量", f"{strategy_profit['预计14天销量']} 单")
            st.metric("14天预计利润", f"${strategy_profit['预计14天利润']:.2f}")
            st.info(f"📌 执行建议：{strategy_profit['执行建议']}")
            st.markdown("#### 🤖 策略说明")
            st.markdown(strategy_profit["解释说明"])

    except Exception as e:
        st.error(f"❌ 模型训练失败: {e}")
else:
    st.info("请上传历史表现 Excel 表开始回测策略")
