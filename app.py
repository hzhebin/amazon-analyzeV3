# app.py - Streamlit App for Profit Optimization

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="亚马逊利润优化分析工具", layout="wide")
st.title("📈 亚马逊利润最大化分析工具")

uploaded_file = st.file_uploader("上传产品表现Excel文件", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    required_cols = ['时间', 'ASIN', '售价(总价)', '价格', 'FBA-可售', '订单量', '点击', 'CPC']
    if not all(col in df.columns for col in required_cols):
        st.error("缺少必要字段，请确认模板格式。")
    else:
        st.success("数据载入成功")

        unit_cost = st.number_input("请输入平均单位成本（USD）", value=8.0)

        df['售价'] = df['售价(总价)']
        df.loc[df['售价'] == 0, '售价'] = df['价格']

        df['广告成本'] = df['点击'] * df['CPC']
        df['总收入'] = df['订单量'] * df['售价']
        df['产品成本'] = df['订单量'] * unit_cost
        df['利润'] = df['总收入'] - df['广告成本'] - df['产品成本']
        df['利润率'] = np.where(df['总收入'] > 0, df['利润'] / df['总收入'], 0)

        df['库存可售周'] = np.where(df['订单量'] > 0, df['FBA-可售'] / df['订单量'], np.nan)
        def tag_risk(weeks):
            if pd.isna(weeks): return '未知'
            elif weeks < 3: return '⚠️低'
            elif weeks < 6: return '🟡中'
            else: return '🟢高'
        df['库存风险'] = df['库存可售周'].apply(tag_risk)

        elasticity = st.slider("销量价格弹性系数（估算）", -3.0, 0.0, -1.0, step=0.1)
        df['售价+10%'] = df['售价'] * 1.1
        df['售价-10%'] = df['售价'] * 0.9
        df['销量+10%'] = (df['订单量'] * (1 + elasticity * (-0.1))).round(1)
        df['销量-10%'] = (df['订单量'] * (1 + elasticity * 0.1)).round(1)
        df['利润+10%'] = (df['售价+10%'] * df['销量+10%'] - df['广告成本'] - df['销量+10%'] * unit_cost).round(2)
        df['利润-10%'] = (df['售价-10%'] * df['销量-10%'] - df['广告成本'] - df['销量-10%'] * unit_cost).round(2)

        df['建议售价'] = np.where(df['利润+10%'] > df['利润'], df['售价+10%'],
                              np.where(df['利润-10%'] > df['利润'], df['售价-10%'], df['售价']))
        df['建议操作'] = np.where(df['库存风险'] == '⚠️低', '暂停广告/控价',
                              np.where(df['利润+10%'] > df['利润'], '建议涨价',
                                       np.where(df['利润-10%'] > df['利润'], '建议降价', '保持')))

        df['当前ACOS'] = np.where(df['总收入'] > 0, df['广告成本'] / df['总收入'], np.nan)
        df['目标ACOS'] = np.where(df['利润率'] < 0.15, df['当前ACOS'] * 0.8,
                               np.where(df['利润率'] > 0.3, df['当前ACOS'] * 1.1, df['当前ACOS']))

        chart_data = df[['ASIN', '利润', '利润+10%', '利润-10%']].dropna()
        melted = chart_data.melt(id_vars='ASIN', var_name='方案', value_name='模拟利润')
        fig = px.bar(melted, x='ASIN', y='模拟利润', color='方案', barmode='group',
                     title='💡 当前 vs 涨/降价利润对比', height=500)
        st.plotly_chart(fig, use_container_width=True)

        show_cols = ['时间', 'ASIN', '售价', '订单量', 'CPC', '广告成本', '利润', '利润率',
                     'FBA-可售', '库存可售周', '库存风险',
                     '建议售价', '建议操作', '当前ACOS', '目标ACOS',
                     '利润+10%', '利润-10%']
        st.dataframe(df[show_cols].sort_values(by='利润', ascending=False), use_container_width=True)

        to_download = df[show_cols]
        csv = to_download.to_csv(index=False).encode('utf-8-sig')
        st.download_button("下载优化建议 CSV", csv, file_name="利润优化建议.csv")
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    st._is_running_with_streamlit = True  # 避免非CLI启动问题
    os.system(f"streamlit run app.py --server.port {port} --server.enableCORS false")
