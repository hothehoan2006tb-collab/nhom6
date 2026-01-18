"""
Data Quality Validators - Kiểm tra chất lượng dữ liệu thủ công
"""
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List


class DataQualityValidator:
    """
    Validate chất lượng dữ liệu - Logic thủ công
    """
    
    @staticmethod
    def validate_data_ranges(df: pd.DataFrame) -> Dict:
        """
        Kiểm tra birth/death rate trong khoảng hợp lý (0-100‰)
        
        Returns:
            {
                "valid": bool,
                "errors": [{column, year, value, reason}],
                "warnings": [...],
                "score": 0-100
            }
        """
        errors = []
        warnings = []
        
        for col in ['birth_rate', 'death_rate']:
            if col not in df.columns:
                continue
            
            series = pd.to_numeric(df[col], errors='coerce')
            
            # Lỗi: Ngoài khoảng 0-100‰
            invalid_mask = ~series.between(0, 100, inclusive='both')
            invalid = df[invalid_mask]
            
            for _, row in invalid.iterrows():
                if pd.notna(row[col]):
                    errors.append({
                        "column": col,
                        "year": int(row['year']),
                        "value": float(row[col]),
                        "reason": f"{col} phải trong khoảng 0-100‰"
                    })
            
            # Warning: Vượt ngưỡng bất thường (>50‰ cho birth, >20‰ cho death)
            threshold = 50 if col == 'birth_rate' else 20
            extreme_mask = series > threshold
            extreme = df[extreme_mask]
            
            for _, row in extreme.iterrows():
                if pd.notna(row[col]):
                    warnings.append({
                        "column": col,
                        "year": int(row['year']),
                        "value": float(row[col]),
                        "reason": f"{col} vượt ngưỡng bất thường (>{threshold}‰)"
                    })
        
        score = max(0, 100 - len(errors)*10 - len(warnings)*5)
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "score": score,
            "total_issues": len(errors) + len(warnings)
        }
    
    @staticmethod
    def detect_anomalies(df: pd.DataFrame, column: str, method: str = 'iqr') -> Dict:
        """
        Phát hiện outliers bằng IQR hoặc Z-score
        
        Args:
            method: 'iqr' hoặc 'zscore'
        """
        series = pd.to_numeric(df[column], errors='coerce').dropna()
        outliers = []
        
        if len(series) < 3:
            return {
                "outliers": [],
                "method": method,
                "count": 0,
                "message": "Insufficient data for outlier detection"
            }
        
        if method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            
            mask = (series < lower) | (series > upper)
            outlier_indices = series[mask].index
            
            for idx in outlier_indices:
                outliers.append({
                    "year": int(df.loc[idx, 'year']),
                    "value": float(df.loc[idx, column]),
                    "method": "IQR",
                    "bounds": f"{lower:.2f} to {upper:.2f}",
                    "reason": f"Nằm ngoài khoảng IQR"
                })
            
            threshold = f"{lower:.2f} to {upper:.2f}"
        
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(series))
            threshold_val = 3
            mask = z_scores > threshold_val
            outlier_indices = series[mask].index
            
            for idx in outlier_indices:
                outliers.append({
                    "year": int(df.loc[idx, 'year']),
                    "value": float(df.loc[idx, column]),
                    "z_score": float(z_scores[series.index.get_loc(idx)]),
                    "method": "Z-score",
                    "reason": f"|Z| > {threshold_val}"
                })
            
            threshold = threshold_val
        
        return {
            "outliers": outliers,
            "method": method,
            "threshold": threshold,
            "count": len(outliers)
        }
    
    @staticmethod
    def check_data_completeness(df: pd.DataFrame) -> Dict:
        """
        Đánh giá % dữ liệu thiếu
        """
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isnull().sum().sum()
        completeness_pct = (1 - missing_cells / total_cells) * 100 if total_cells > 0 else 0
        
        missing_by_col = df.isnull().sum().to_dict()
        
        # Chi tiết các năm thiếu data
        missing_details = []
        for col in df.columns:
            if col == 'year':
                continue
            missing_years = df[df[col].isnull()]['year'].tolist()
            if missing_years:
                missing_details.append({
                    "column": col,
                    "missing_years": [int(y) for y in missing_years],
                    "count": len(missing_years)
                })
        
        return {
            "completeness_pct": round(completeness_pct, 2),
            "missing_by_column": missing_by_col,
            "total_missing": int(missing_cells),
            "missing_details": missing_details,
            "score": round(completeness_pct, 0)
        }
