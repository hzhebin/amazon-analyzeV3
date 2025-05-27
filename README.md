# Amazon Profit AI 回测与策略建议系统

## 简介
本系统为亚马逊卖家量身定制，自动对历史运营数据回测，并输出专业运营决策建议。

## 用法
1. `pip install -r requirements.txt`
2. `streamlit run app.py`
3. 上传产品表现.xlsx文件，系统自动输出两套策略建议（销量优先/利润最大化）

## 部署
支持 Railway / Hugging Face / Streamlit Cloud。已包含 `Procfile`。

## 注意
- 数据字段支持中英文混合/模糊识别
- 有异常检测、变量解释、预测趋势
- 支持多变量联动优化，逻辑均为落地实战型
