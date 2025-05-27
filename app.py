import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from profit_engine import ProfitEngine
from optimizer import StrategyOptimizer
from xgb_model import XGBPredictor

# 页面配置
st.set_page_config(
    page_title="亚马逊AI运营策略系统",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .strategy-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: white;
    }
    .profit-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: white;
    }
    .metric-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("🚀 亚马逊AI运营策略系统")
    st.markdown("### 基于机器学习的智能定价与广告优化平台")
    
    # 侧边栏配置
    with st.sidebar:
        st.header("⚙️ 系统配置")
        
        # 目标库存周转天数
        target_inventory_days = st.slider("目标库存周转天数", 15, 90, 30)
        
        # 风险偏好
        risk_preference = st.selectbox(
            "风险偏好",
            ["保守型", "平衡型", "激进型"]
        )
        
        # 优化目标权重
        st.subheader("优化目标权重")
        profit_weight = st.slider("利润权重", 0.0, 1.0, 0.6)
        sales_weight = st.slider("销量权重", 0.0, 1.0, 0.4)
        
        # 约束条件
        st.subheader("约束条件")
        max_acos = st.slider("最大ACOS限制 (%)", 10, 100, 35)
        min_profit_margin = st.slider("最小利润率 (%)", 5, 50, 15)
    
    # 主界面
    tab1, tab2, tab3, tab4 = st.tabs(["📤 数据上传", "🧮 模型训练", "🎯 策略生成", "📊 结果分析"])
    
    with tab1:
        data_upload_section()
    
    with tab2:
        if 'uploaded_data' in st.session_state:
            model_training_section()
        else:
            st.warning("请先上传数据文件")
    
    with tab3:
        if 'model_trained' in st.session_state and st.session_state.model_trained:
            strategy_generation_section(target_inventory_days, risk_preference, profit_weight, sales_weight, max_acos, min_profit_margin)
        else:
            st.warning("请先完成模型训练")
    
    with tab4:
        if 'strategies_generated' in st.session_state and st.session_state.strategies_generated:
            results_analysis_section()
        else:
            st.warning("请先生成策略建议")

def data_upload_section():
    st.header("📤 数据文件上传")
    
    uploaded_file = st.file_uploader(
        "上传产品表现数据文件 (.xlsx)",
        type=['xlsx'],
        help="支持包含销量、价格、广告等数据的Excel文件"
    )
    
    if uploaded_file is not None:
        try:
            # 读取数据
            df = pd.read_excel(uploaded_file)
            st.success(f"✅ 文件上传成功！数据形状: {df.shape}")
            
            # 数据预览
            st.subheader("📋 数据预览")
            st.dataframe(df.head(10))
            
            # 字段映射
            st.subheader("🔗 字段映射配置")
            
            columns = df.columns.tolist()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                date_col = st.selectbox("日期字段", columns, 
                    index=find_best_match(columns, ['日期', 'date', '时间', 'time']))
                price_col = st.selectbox("售价字段", columns,
                    index=find_best_match(columns, ['售价', '价格', 'price', '总价']))
                sales_col = st.selectbox("销量字段", columns,
                    index=find_best_match(columns, ['销量', 'sales', '订单', 'units']))
                
            with col2:
                sessions_col = st.selectbox("Sessions字段", columns,
                    index=find_best_match(columns, ['sessions', 'pv', '访问']))
                cpc_col = st.selectbox("CPC字段", columns,
                    index=find_best_match(columns, ['cpc', '点击成本']))
                ad_spend_col = st.selectbox("广告花费字段", columns,
                    index=find_best_match(columns, ['广告花费', 'ad_spend', '花费']))
                
            with col3:
                acos_col = st.selectbox("ACOS字段", columns,
                    index=find_best_match(columns, ['acos', '广告成本占比']))
                cvr_col = st.selectbox("转化率字段", columns,
                    index=find_best_match(columns, ['cvr', '转化率', 'conversion']))
                inventory_col = st.selectbox("库存字段", columns,
                    index=find_best_match(columns, ['库存', 'inventory', '可售', 'fba']))
            
            # 数据处理和存储
            if st.button("🔄 处理数据并开始分析", type="primary"):
                processed_data = process_uploaded_data(
                    df, date_col, price_col, sales_col, sessions_col, 
                    cpc_col, ad_spend_col, acos_col, cvr_col, inventory_col
                )
                
                if processed_data is not None:
                    st.session_state.uploaded_data = processed_data
                    st.session_state.field_mapping = {
                        'date': date_col, 'price': price_col, 'sales': sales_col,
                        'sessions': sessions_col, 'cpc': cpc_col, 'ad_spend': ad_spend_col,
                        'acos': acos_col, 'cvr': cvr_col, 'inventory': inventory_col
                    }
                    st.success("✅ 数据处理完成！请切换到模型训练标签页。")
                    st.rerun()
                
        except Exception as e:
            st.error(f"❌ 文件读取失败: {str(e)}")

def find_best_match(columns, keywords):
    """字段模糊匹配"""
    for i, col in enumerate(columns):
        col_lower = col.lower()
        for keyword in keywords:
            if keyword.lower() in col_lower:
                return i
    return 0

def process_uploaded_data(df, date_col, price_col, sales_col, sessions_col, 
                         cpc_col, ad_spend_col, acos_col, cvr_col, inventory_col):
    """处理上传的数据"""
    try:
        # 重命名列
        processed_df = df.rename(columns={
            date_col: 'date',
            price_col: 'price', 
            sales_col: 'sales',
            sessions_col: 'sessions',
            cpc_col: 'cpc',
            ad_spend_col: 'ad_spend',
            acos_col: 'acos',
            cvr_col: 'cvr',
            inventory_col: 'inventory'
        })
        
        # 数据类型转换
        processed_df['date'] = pd.to_datetime(processed_df['date'])
        numeric_cols = ['price', 'sales', 'sessions', 'cpc', 'ad_spend', 'acos', 'cvr', 'inventory']
        
        for col in numeric_cols:
            if col in processed_df.columns:
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
        
        # 计算衍生指标
        processed_df['profit'] = processed_df['sales'] * processed_df['price'] - processed_df['ad_spend']
        processed_df['profit_margin'] = processed_df['profit'] / (processed_df['sales'] * processed_df['price']) * 100
        processed_df['inventory_days'] = processed_df['inventory'] / processed_df['sales'].rolling(7).mean()
        
        # 删除缺失值过多的行
        processed_df = processed_df.dropna(subset=['price', 'sales', 'sessions'])
        
        return processed_df
        
    except Exception as e:
        st.error(f"数据处理失败: {str(e)}")
        return None

def model_training_section():
    st.header("🧮 机器学习模型训练")
    
    data = st.session_state.uploaded_data
    
    # 数据统计信息
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📅 数据天数", len(data))
    with col2:
        st.metric("💰 平均售价", f"¥{data['price'].mean():.2f}")
    with col3:
        st.metric("📦 平均销量", f"{data['sales'].mean():.1f}")
    with col4:
        st.metric("📊 平均ACOS", f"{data['acos'].mean():.1f}%")
    
    # 数据质量检查
    st.subheader("🔍 数据质量检查")
    
    missing_data = data.isnull().sum()
    if missing_data.sum() > 0:
        st.warning("⚠️ 发现缺失数据")
        st.dataframe(missing_data[missing_data > 0])
    else:
        st.success("✅ 数据完整，无缺失值")
    
    # 异常值检测
    outliers = detect_outliers(data)
    if len(outliers) > 0:
        st.warning(f"⚠️ 发现 {len(outliers)} 个异常数据点")
        if st.checkbox("显示异常值详情"):
            st.dataframe(data.iloc[outliers])
    
    # 模型训练
    if st.button("🚀 开始训练模型", type="primary"):
        with st.spinner("正在训练XGBoost模型..."):
            try:
                # 初始化模型
                predictor = XGBPredictor()
                
                # 准备训练数据
                features = ['price', 'sessions', 'cpc', 'ad_spend', 'acos', 'inventory_days']
                targets = ['sales', 'profit']
                
                X = data[features].fillna(data[features].mean())
                y_sales = data['sales'].fillna(0)
                y_profit = data['profit'].fillna(0)
                
                # 训练模型
                sales_score = predictor.train_sales_model(X, y_sales)
                profit_score = predictor.train_profit_model(X, y_profit)
                
                # 存储模型
                st.session_state.predictor = predictor
                st.session_state.model_trained = True
                st.session_state.training_data = data
                
                st.success(f"✅ 模型训练完成！")
                st.info(f"销量预测准确度: {sales_score:.3f} | 利润预测准确度: {profit_score:.3f}")
                
                # SHAP分析
                show_shap_analysis(predictor, X)
                
            except Exception as e:
                st.error(f"❌ 模型训练失败: {str(e)}")

def detect_outliers(data):
    """检测异常值"""
    outliers = []
    numeric_cols = ['price', 'sales', 'sessions', 'cpc', 'ad_spend']
    
    for col in numeric_cols:
        if col in data.columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            col_outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)].index
            outliers.extend(col_outliers)
    
    return list(set(outliers))

def show_shap_analysis(predictor, X):
    """显示SHAP分析结果"""
    st.subheader("🔍 模型解释性分析 (SHAP)")
    
    try:
        import shap
        
        # 销量模型SHAP
        explainer_sales = shap.TreeExplainer(predictor.sales_model)
        shap_values_sales = explainer_sales.shap_values(X.head(100))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**销量影响因子**")
            feature_importance = np.abs(shap_values_sales).mean(0)
            importance_df = pd.DataFrame({
                'Feature': X.columns,
                'Importance': feature_importance
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**利润影响因子**")
            explainer_profit = shap.TreeExplainer(predictor.profit_model)
            shap_values_profit = explainer_profit.shap_values(X.head(100))
            
            feature_importance_profit = np.abs(shap_values_profit).mean(0)
            importance_df_profit = pd.DataFrame({
                'Feature': X.columns,
                'Importance': feature_importance_profit
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(importance_df_profit, x='Importance', y='Feature', orientation='h')
            st.plotly_chart(fig, use_container_width=True)
            
    except ImportError:
        st.info("💡 安装 shap 库可获得更详细的模型解释")

def strategy_generation_section(target_inventory_days, risk_preference, profit_weight, sales_weight, max_acos, min_profit_margin):
    st.header("🎯 AI策略生成")
    
    # 当前状态概览
    data = st.session_state.training_data
    latest_data = data.iloc[-1]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("当前售价", f"¥{latest_data['price']:.2f}")
    with col2:
        st.metric("当前ACOS", f"{latest_data['acos']:.1f}%")
    with col3:
        st.metric("库存天数", f"{latest_data['inventory_days']:.1f}天")
    with col4:
        st.metric("当前利润率", f"{latest_data['profit_margin']:.1f}%")
    
    # 策略生成按钮
    if st.button("🚀 生成优化策略", type="primary"):
        with st.spinner("AI正在分析数据并生成策略建议..."):
            try:
                # 初始化优化器
                optimizer = StrategyOptimizer(st.session_state.predictor)
                
                # 设置优化参数
                optimization_params = {
                    'target_inventory_days': target_inventory_days,
                    'risk_preference': risk_preference,
                    'profit_weight': profit_weight,
                    'sales_weight': sales_weight,
                    'max_acos': max_acos,
                    'min_profit_margin': min_profit_margin
                }
                
                # 生成策略
                current_state = {
                    'price': latest_data['price'],
                    'cpc': latest_data['cpc'],
                    'ad_spend': latest_data['ad_spend'],
                    'acos': latest_data['acos'],
                    'sessions': latest_data['sessions'],
                    'inventory_days': latest_data['inventory_days']
                }
                
                strategies = optimizer.generate_strategies(current_state, optimization_params)
                
                # 存储策略结果
                st.session_state.strategies = strategies
                st.session_state.strategies_generated = True
                st.session_state.current_state = current_state
                
                # 显示策略卡片
                display_strategy_cards(strategies)
                
            except Exception as e:
                st.error(f"❌ 策略生成失败: {str(e)}")

def display_strategy_cards(strategies):
    """显示策略建议卡片"""
    st.subheader("💡 AI策略建议")
    
    for i, (strategy_name, strategy_data) in enumerate(strategies.items()):
        
        if i == 0:  # 销量优先策略
            card_class = "strategy-card"
            icon = "📈"
        else:  # 利润优先策略
            card_class = "profit-card"
            icon = "💰"
        
        with st.container():
            st.markdown(f"""
            <div class="{card_class}">
                <h3>{icon} {strategy_name}</h3>
                <p><strong>建议售价:</strong> ¥{strategy_data['price']:.2f}</p>
                <p><strong>建议CPC:</strong> ¥{strategy_data['cpc']:.2f}</p>
                <p><strong>建议广告预算:</strong> ¥{strategy_data['ad_spend']:.2f}</p>
                <p><strong>预期ACOS:</strong> {strategy_data['acos']:.1f}%</p>
                <p><strong>预测销量:</strong> {strategy_data['predicted_sales']:.1f} 单</p>
                <p><strong>预测利润:</strong> ¥{strategy_data['predicted_profit']:.2f}</p>
                <p><strong>适用场景:</strong> {strategy_data['scenario']}</p>
                <p><strong>执行建议:</strong> {strategy_data['execution_advice']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # 详细指标对比
        col1, col2, col3, col4 = st.columns(4)
        
        current_state = st.session_state.current_state
        
        with col1:
            price_change = ((strategy_data['price'] - current_state['price']) / current_state['price']) * 100
            st.metric(
                "售价变化", 
                f"¥{strategy_data['price']:.2f}",
                f"{price_change:+.1f}%"
            )
        
        with col2:
            cpc_change = ((strategy_data['cpc'] - current_state['cpc']) / current_state['cpc']) * 100
            st.metric(
                "CPC变化",
                f"¥{strategy_data['cpc']:.2f}",
                f"{cpc_change:+.1f}%"
            )
        
        with col3:
            st.metric(
                "预测销量",
                f"{strategy_data['predicted_sales']:.1f}",
                "单/14天"
            )
        
        with col4:
            st.metric(
                "预测利润",
                f"¥{strategy_data['predicted_profit']:.2f}",
                "14天总计"
            )
        
        st.divider()

def results_analysis_section():
    st.header("📊 策略分析与对比")
    
    strategies = st.session_state.strategies
    current_state = st.session_state.current_state
    
    # 策略对比表
    st.subheader("📋 策略对比总览")
    
    comparison_data = []
    for name, strategy in strategies.items():
        comparison_data.append({
            '策略名称': name,
            '建议售价': f"¥{strategy['price']:.2f}",
            '建议CPC': f"¥{strategy['cpc']:.2f}", 
            '广告预算': f"¥{strategy['ad_spend']:.2f}",
            '预期ACOS': f"{strategy['acos']:.1f}%",
            '预测销量': f"{strategy['predicted_sales']:.1f}",
            '预测利润': f"¥{strategy['predicted_profit']:.2f}",
            '投资回报': f"{strategy.get('roi', 0):.1f}%"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    
    # 可视化分析
    st.subheader("📈 可视化分析")
    
    tab1, tab2, tab3 = st.tabs(["💰 利润对比", "📦 销量对比", "🎯 风险收益分析"])
    
    with tab1:
        # 利润对比图
        profits = [strategy['predicted_profit'] for strategy in strategies.values()]
        names = list(strategies.keys())
        
        fig = go.Figure(data=[
            go.Bar(x=names, y=profits, marker_color=['#FF6B6B', '#4ECDC4'])
        ])
        fig.update_layout(
            title="14天预测利润对比",
            xaxis_title="策略",
            yaxis_title="利润 (¥)",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # 销量对比图
        sales = [strategy['predicted_sales'] for strategy in strategies.values()]
        
        fig = go.Figure(data=[
            go.Bar(x=names, y=sales, marker_color=['#FF9F43', '#10AC84'])
        ])
        fig.update_layout(
            title="14天预测销量对比",
            xaxis_title="策略",
            yaxis_title="销量 (单)",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # 风险收益散点图
        acos_values = [strategy['acos'] for strategy in strategies.values()]
        
        fig = go.Figure()
        
        for i, (name, strategy) in enumerate(strategies.items()):
            fig.add_trace(go.Scatter(
                x=[strategy['acos']],
                y=[strategy['predicted_profit']],
                mode='markers+text',
                name=name,
                text=[name],
                textposition="top center",
                marker=dict(size=15, color=['#FF6B6B', '#4ECDC4'][i])
            ))
        
        fig.update_layout(
            title="风险-收益分析",
            xaxis_title="ACOS (%)",
            yaxis_title="预测利润 (¥)",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # 执行建议
    st.subheader("🎯 执行建议")
    
    best_profit_strategy = max(strategies.items(), key=lambda x: x[1]['predicted_profit'])
    best_sales_strategy = max(strategies.items(), key=lambda x: x[1]['predicted_sales'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success(f"**最优利润策略:** {best_profit_strategy[0]}")
        st.write(f"预期利润: ¥{best_profit_strategy[1]['predicted_profit']:.2f}")
        st.write(f"执行建议: {best_profit_strategy[1]['execution_advice']}")
    
    with col2:
        st.info(f"**最优销量策略:** {best_sales_strategy[0]}")
        st.write(f"预期销量: {best_sales_strategy[1]['predicted_sales']:.1f} 单")
        st.write(f"执行建议: {best_sales_strategy[1]['execution_advice']}")

if __name__ == "__main__":
    # 初始化session state
    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = None
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'strategies_generated' not in st.session_state:
        st.session_state.strategies_generated = False
    
    main()
