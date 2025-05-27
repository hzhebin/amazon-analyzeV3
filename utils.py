import re
import pandas as pd

FIELD_MAP = {
    '售价': 'price',
    '总价': 'price',
    '价格': 'price',
    'FBA可售': 'fba_qty',
    '可用库存': 'stock_qty',
    '库存': 'stock_qty',
    '销量': 'sales_qty',
    '销量(件)': 'sales_qty',
    '销售量': 'sales_qty',
    '订单销量': 'sales_qty',
    '利润额': 'profit',
    '订单利润': 'profit',
    '利润': 'profit',
    '结算利润': 'profit',
    '利润值': 'profit',
    'Sessions': 'sessions',
    'Sessions-Total': 'sessions',
    '浏览量': 'sessions',
    'PV-Total': 'pv',
    '广告花费': 'ad_spend',
    'CPC': 'cpc',
    'ACoS': 'acos',
    'ACoAS': 'acoas',
    'CVR': 'cvr',
    'CPM': 'cpm',
    'CPO': 'cpo',
    '结算利润率': 'profit_margin',
    '利润率': 'profit_margin',
    '日期': 'date',
    'ASIN': 'asin',
    'SKU': 'sku',
    '品名': 'title',
    '评论数': 'review_count',
    '可售天数': 'days_in_stock',
}

def fuzzy_map_columns(cols):
    result = {}
    used = set()
    for col in cols:
        mapped = None
        for k, v in FIELD_MAP.items():
            if k in col or v in col.lower():
                mapped = v
                break
        if not mapped:
            mapped = col.lower()
        suffix = 1
        new_mapped = mapped
        while new_mapped in used:
            suffix += 1
            new_mapped = f"{mapped}_{suffix}"
        used.add(new_mapped)
        result[col] = new_mapped
    return result

def detect_anomalies(df):
    if 'sales_qty' not in df.columns or 'profit' not in df.columns:
        return pd.DataFrame()
    sales = df['sales_qty']
    profit = df['profit']
    anomaly = (sales.pct_change().abs() > 2) | (profit.pct_change().abs() > 2)
    return df[anomaly]
