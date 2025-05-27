import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class ProfitEngine:
    """
    亚马逊运营利润引擎核心类
    融合真实运营逻辑，实现利润最大化与库存周转平衡
    """
    
    def __init__(self):
        self.historical_data = None
        self.market_elasticity = {}
        self.inventory_pressure_threshold = {
            'high': 60,  # 高库存压力阈值(天)
            'medium': 30,  # 中等库存压力阈值(天)
            'low': 15    # 低库存压力阈值(天)
        }
        
    def load_historical_data(self, data: pd.DataFrame):
        """加载历史数据并计算市场弹性"""
        self.historical_data = data.copy()
        self._calculate_market_elasticity()
        self._detect_seasonality()
        
    def _calculate_market_elasticity(self):
        """计算价格和广告的市场弹性"""
        if self.historical_data is None:
            return
            
        data = self.historical_data
        
        # 价格弹性计算
        price_changes = data['price'].pct_change()
        sales_changes = data['sales'].pct_change()
        
        # 过滤异常值
        valid_mask = (abs(price_changes) < 0.5) & (abs(sales_changes) < 2.0)
        price_changes = price_changes[valid_mask]
        sales_changes = sales_changes[valid_mask]
        
        if len(price_changes) > 10:
            price_elasticity = np.corrcoef(price_changes, sales_changes)[0, 1]
            self.market_elasticity['price'] = price_elasticity if not np.isnan(price_elasticity) else -0.8
        else:
            self.market_elasticity['price'] = -0.8  # 默认价格弹性
            
        # 广告弹性计算
        ad_changes = data['ad_spend'].pct_change()
        sessions_changes = data['sessions'].pct_change()
        
        valid_mask = (abs(ad_changes) < 1.0) & (abs(sessions_changes) < 2.0)
        ad_changes = ad_changes[valid_mask]
        sessions_changes = sessions_changes[valid_mask]
        
        if len(ad_changes) > 10:
            ad_elasticity = np.corrcoef(ad_changes, sessions_changes)[0, 1]
            self.market_elasticity['advertising'] = ad_elasticity if not np.isnan(ad_elasticity) else 0.6
        else:
            self.market_elasticity['advertising'] = 0.6  # 默认广告弹性
            
    def _detect_seasonality(self):
        """检测季节性模式"""
        if self.historical_data is None or len(self.historical_data) < 30:
            self.seasonality_factor = 1.0
            return
            
        # 简单的季节性检测 - 基于销量的周期性变化
        sales = self.historical_data['sales'].values
        
        # 7天周期性检测
        weekly_pattern = []
        for i in range(7):
            day
