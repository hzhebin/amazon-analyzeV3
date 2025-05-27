import re

FIELD_MAP = {
    '售价': 'price',
    '总价': 'price',
    '价格': 'price',
    'FBA可售': 'fba_qty',
    '可用库存': 'stock_qty',
    '库存': 'stock_qty',
    '销量': 'sales_qty',
    '销量(件)': 'sales_qty',
    '利润额': 'profit',
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
    for col in cols:
        for k, v in FIELD_MAP.items():
            if k in col or v in col.lower():
                result[col] = v
                break
        else:
            result[col] = col.lower()
    return result

def detect_anomalies(df):
    sales = df['sales_qty']
    profit = df['profit']
    anomaly = (sales.pct_change().abs() > 2) | (profit.pct_change().abs() > 2)
    return df[anomaly]
