"""
Statistical Processor - Core Logic của Đồ Án
Xử lý TOÀN BỘ thống kê bằng code thủ công - KHÔNG dùng AI
"""
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from typing import Dict, List, Optional


def _to_python_type(val):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(val, (np.integer, np.floating)):
        return float(val) if isinstance(val, np.floating) else int(val)
    elif isinstance(val, np.bool_):
        return bool(val)
    elif isinstance(val, np.ndarray):
        return val.tolist()
    elif val is None or val is np.nan:
        return None
    return val


class StatisticalProcessor:
    """
    Xử lý thống kê thủ công - Core logic thể hiện kỹ năng khoa học dữ liệu
    """
    
    @staticmethod
    def process_country_statistics(df: pd.DataFrame, country_name: str) -> Dict:
        """
        Xử lý TOÀN BỘ thống kê cho 1 quốc gia - HÀM CHÍNH
        
        Args:
            df: DataFrame với columns [year, birth_rate, death_rate, population]
            country_name: Tên quốc gia
        
        Returns:
            Dict chứa tất cả phân tích thống kê
        """
        result = {
            "country": country_name,
            "data_period": {},
            "birth_rate_analysis": {},
            "death_rate_analysis": {},
            "population_analysis": {},
            "demographic_indicators": {},
            "trend_analysis": {},
            "predictions": {}
        }
        
        # Data period
        result["data_period"] = {
            "start_year": int(df['year'].min()),
            "end_year": int(df['year'].max()),
            "total_years": len(df),
            "data_completeness_pct": _to_python_type(round((1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100, 2))
        }
        
        # === BIRTH RATE ANALYSIS ===
        birth = pd.to_numeric(df['birth_rate'], errors='coerce').dropna()
        if len(birth) > 0:
            result["birth_rate_analysis"] = {
                "mean": _to_python_type(round(birth.mean(), 2)),
                "median": _to_python_type(round(birth.median(), 2)),
                "std": _to_python_type(round(birth.std(), 2)),
                "min": _to_python_type(round(birth.min(), 2)),
                "max": _to_python_type(round(birth.max(), 2)),
                "range": _to_python_type(round(birth.max() - birth.min(), 2)),
                "cv_percent": _to_python_type(round((birth.std() / birth.mean() * 100), 2)) if birth.mean() > 0 else None,
                "first_value": _to_python_type(round(birth.iloc[0], 2)),
                "last_value": _to_python_type(round(birth.iloc[-1], 2)),
                "total_change": _to_python_type(round(birth.iloc[-1] - birth.iloc[0], 2)),
                "percent_change": _to_python_type(round((birth.iloc[-1] - birth.iloc[0]) / birth.iloc[0] * 100, 2)) if birth.iloc[0] > 0 else None,
                # Quartiles
                "q1": _to_python_type(round(birth.quantile(0.25), 2)),
                "q3": _to_python_type(round(birth.quantile(0.75), 2)),
                "iqr": _to_python_type(round(birth.quantile(0.75) - birth.quantile(0.25), 2))
            }
        
        # === DEATH RATE ANALYSIS ===
        death = pd.to_numeric(df['death_rate'], errors='coerce').dropna()
        if len(death) > 0:
            result["death_rate_analysis"] = {
                "mean": _to_python_type(round(death.mean(), 2)),
                "median": _to_python_type(round(death.median(), 2)),
                "std": _to_python_type(round(death.std(), 2)),
                "min": _to_python_type(round(death.min(), 2)),
                "max": _to_python_type(round(death.max(), 2)),
                "total_change": _to_python_type(round(death.iloc[-1] - death.iloc[0], 2)),
                "percent_change": _to_python_type(round((death.iloc[-1] - death.iloc[0]) / death.iloc[0] * 100, 2)) if death.iloc[0] > 0 else None
            }
        
        # === POPULATION ANALYSIS ===
        pop = pd.to_numeric(df['population'], errors='coerce').dropna()
        if len(pop) > 0:
            result["population_analysis"] = {
                "current": int(pop.iloc[-1]),
                "initial": int(pop.iloc[0]),
                "mean": int(pop.mean()),
                "total_change": int(pop.iloc[-1] - pop.iloc[0]),
                "percent_change": _to_python_type(round((pop.iloc[-1] - pop.iloc[0]) / pop.iloc[0] * 100, 2)) if pop.iloc[0] > 0 else None,
                "avg_annual_change": int((pop.iloc[-1] - pop.iloc[0]) / len(pop)) if len(pop) > 0 else None
            }
        
        # === DEMOGRAPHIC INDICATORS ===
        if len(birth) > 0 and len(death) > 0:
            natural_increase = birth.mean() - death.mean()
            
            # Tính confidence interval cho natural increase
            birth_ci = StatisticalProcessor._calculate_confidence_interval(birth, confidence=0.95)
            death_ci = StatisticalProcessor._calculate_confidence_interval(death, confidence=0.95)
            
            result["demographic_indicators"] = {
                "natural_increase_rate": _to_python_type(round(natural_increase, 2)),
                "crude_growth_rate_percent": _to_python_type(round(natural_increase / 10, 2)),
                "birth_death_ratio": _to_python_type(round(birth.mean() / death.mean(), 2)) if death.mean() > 0 else None,
                "demographic_stage": StatisticalProcessor._classify_demographic_stage(birth.mean(), death.mean()),
                "birth_rate_95ci": birth_ci,
                "death_rate_95ci": death_ci
            }
        
        # === CORRELATION ANALYSIS ===
        if len(birth) > 0 and len(death) > 0 and len(birth) == len(death):
            correlation = StatisticalProcessor._analyze_correlation(birth.values, death.values)
            result["correlation_analysis"] = correlation
        
        # === NORMALITY TESTS ===
        if len(birth) >= 3:
            result["normality_tests"] = {
                "birth_rate": StatisticalProcessor._test_normality(birth.values),
                "death_rate": StatisticalProcessor._test_normality(death.values) if len(death) >= 3 else None
            }
        
        # === TREND ANALYSIS (Linear Regression) ===
        if len(birth) >= 3:
            result["trend_analysis"]["birth_rate"] = StatisticalProcessor._analyze_trend(df, 'birth_rate')
        
        if len(death) >= 3:
            result["trend_analysis"]["death_rate"] = StatisticalProcessor._analyze_trend(df, 'death_rate')
        
        # === HYPOTHESIS TESTING ===
        if len(birth) >= 3:
            # Test: H0: birth rate mean = 20‰ (world average assumption)
            result["hypothesis_tests"] = {
                "birth_rate_vs_20": StatisticalProcessor._one_sample_t_test(birth.values, 20.0),
                "death_rate_vs_8": StatisticalProcessor._one_sample_t_test(death.values, 8.0) if len(death) >= 3 else None
            }
        
        # === PREDICTIONS (Linear Extrapolation) ===
        if "birth_rate" in result["trend_analysis"]:
            trend = result["trend_analysis"]["birth_rate"]
            last_year = int(df['year'].max())
            
            result["predictions"]["birth_rate_next_5_years"] = [
                {
                    "year": int(last_year + i),
                    "predicted_value": _to_python_type(round(trend["slope"] * (last_year + i) + trend["intercept"], 2))
                }
                for i in range(1, 6)
            ]
        
        return result
    
    @staticmethod
    def _analyze_trend(df: pd.DataFrame, column: str) -> Dict:
        """
        Phân tích xu hướng bằng Linear Regression
        Tính R², p-value, confidence level
        """
        series = pd.to_numeric(df[column], errors='coerce').dropna()
        years = df.loc[series.index, 'year'].values.reshape(-1, 1)
        values = series.values
        
        # Linear regression
        model = LinearRegression()
        model.fit(years, values)
        
        slope = model.coef_[0]
        intercept = model.intercept_
        r_squared = model.score(years, values)
        
        # Statistical significance (T-test)
        n = len(values)
        y_pred = model.predict(years)
        residuals = values - y_pred
        se = np.sqrt(np.sum(residuals**2) / (n - 2))
        se_slope = se / np.sqrt(np.sum((years.flatten() - years.mean())**2))
        t_stat = slope / se_slope if se_slope > 0 else 0
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2)) if n > 2 else 1
        
        # Determine direction
        if p_value > 0.05:
            direction = "stable"
            significance = "not_significant"
        elif slope > 0:
            direction = "increasing"
            significance = "significant"
        else:
            direction = "decreasing"
            significance = "significant"
        
        # Confidence level
        if r_squared > 0.7 and p_value < 0.05:
            confidence = "high"
        elif r_squared > 0.4:
            confidence = "medium"
        else:
            confidence = "low"
        
        return {
            "direction": direction,
            "slope": _to_python_type(round(slope, 4)),
            "intercept": _to_python_type(round(intercept, 2)),
            "r_squared": _to_python_type(round(r_squared, 4)),
            "p_value": _to_python_type(round(p_value, 4)),
            "significance": significance,
            "confidence": confidence,
            "interpretation": f"Xu hướng {direction} (R²={r_squared:.2f}, p={p_value:.3f})"
        }
    
    @staticmethod
    def _classify_demographic_stage(birth_rate: float, death_rate: float) -> str:
        """
        Phân loại giai đoạn chuyển đổi dân số theo lý thuyết Demographic Transition
        """
        if birth_rate > 30 and death_rate > 15:
            return "Stage 1: High Stationary (developing)"
        elif birth_rate > 25 and death_rate < 15:
            return "Stage 2: Early Expanding (industrializing)"
        elif birth_rate < 25 and birth_rate > 15 and death_rate < 10:
            return "Stage 3: Late Expanding (developed)"
        elif birth_rate < 15 and death_rate < 10:
            return "Stage 4: Low Stationary (post-industrial)"
        else:
            return "Transitional"
    
    # ========== PROBABILITY & STATISTICS METHODS ==========
    
    @staticmethod
    def _calculate_confidence_interval(data: pd.Series, confidence: float = 0.95) -> Dict:
        """
        Tính khoảng tin cậy (Confidence Interval) cho mean
        
        Formula: CI = mean ± t(α/2, n-1) × (s / √n)
        
        Args:
            data: Series dữ liệu
            confidence: Độ tin cậy (0.90, 0.95, 0.99)
        
        Returns:
            {
                "mean": float,
                "lower_bound": float,
                "upper_bound": float,
                "margin_of_error": float,
                "confidence_level": float
            }
        """
        n = len(data)
        if n < 2:
            return {"error": "Insufficient data"}
        
        mean = data.mean()
        std_err = stats.sem(data)  # Standard error of the mean
        
        # Critical value from t-distribution
        alpha = 1 - confidence
        df = n - 1
        t_critical = stats.t.ppf(1 - alpha/2, df)
        
        margin_of_error = t_critical * std_err
        
        return {
            "mean": _to_python_type(round(mean, 2)),
            "lower_bound": _to_python_type(round(mean - margin_of_error, 2)),
            "upper_bound": _to_python_type(round(mean + margin_of_error, 2)),
            "margin_of_error": _to_python_type(round(margin_of_error, 2)),
            "confidence_level": confidence,
            "sample_size": int(n),
            "interpretation": f"95% tin cậy rằng giá trị trung bình thực nằm trong [{mean - margin_of_error:.2f}, {mean + margin_of_error:.2f}]"
        }
    
    @staticmethod
    def _analyze_correlation(x: np.ndarray, y: np.ndarray) -> Dict:
        """
        Phân tích tương quan giữa 2 biến (Pearson & Spearman)
        
        Returns:
            {
                "pearson_r": float,
                "pearson_p_value": float,
                "spearman_rho": float,
                "spearman_p_value": float,
                "interpretation": str
            }
        """
        if len(x) != len(y) or len(x) < 3:
            return {"error": "Insufficient or mismatched data"}
        
        # Pearson correlation (linear relationship)
        pearson_r, pearson_p = stats.pearsonr(x, y)
        
        # Spearman correlation (monotonic relationship)
        spearman_rho, spearman_p = stats.spearmanr(x, y)
        
        # Interpret correlation strength
        abs_r = abs(pearson_r)
        if abs_r >= 0.7:
            strength = "strong"
        elif abs_r >= 0.4:
            strength = "moderate"
        elif abs_r >= 0.2:
            strength = "weak"
        else:
            strength = "very weak"
        
        direction = "positive" if pearson_r > 0 else "negative"
        
        return {
            "pearson_r": _to_python_type(round(pearson_r, 4)),
            "pearson_p_value": _to_python_type(round(pearson_p, 4)),
            "pearson_significant": bool(pearson_p < 0.05),
            "spearman_rho": _to_python_type(round(spearman_rho, 4)),
            "spearman_p_value": _to_python_type(round(spearman_p, 4)),
            "spearman_significant": bool(spearman_p < 0.05),
            "correlation_strength": strength,
            "correlation_direction": direction,
            "interpretation": f"Tương quan {direction} {strength} (r={pearson_r:.2f}, p={pearson_p:.3f})"
        }
    
    @staticmethod
    def _test_normality(data: np.ndarray) -> Dict:
        """
        Kiểm định phân phối chuẩn (Shapiro-Wilk test)
        
        H0: Dữ liệu có phân phối chuẩn
        H1: Dữ liệu KHÔNG có phân phối chuẩn
        
        Returns:
            {
                "test_statistic": float,
                "p_value": float,
                "is_normal": bool,
                "interpretation": str
            }
        """
        if len(data) < 3:
            return {"error": "Insufficient data (need n >= 3)"}
        
        # Shapiro-Wilk test
        statistic, p_value = stats.shapiro(data)
        
        is_normal = bool(p_value >= 0.05)  # Convert numpy.bool_ to Python bool
        
        return {
            "test": "Shapiro-Wilk",
            "test_statistic": _to_python_type(round(statistic, 4)),
            "p_value": _to_python_type(round(p_value, 4)),
            "is_normal": is_normal,
            "alpha": 0.05,
            "interpretation": f"Dữ liệu {'CÓ' if is_normal else 'KHÔNG CÓ'} phân phối chuẩn (p={p_value:.3f})"
        }
    
    @staticmethod
    def _one_sample_t_test(data: np.ndarray, hypothesized_mean: float) -> Dict:
        """
        One-sample t-test - Kiểm định giả thuyết về mean
        
        H0: μ = hypothesized_mean
        H1: μ ≠ hypothesized_mean (two-tailed)
        
        Returns:
            {
                "sample_mean": float,
                "hypothesized_mean": float,
                "t_statistic": float,
                "p_value": float,
                "reject_h0": bool,
                "interpretation": str
            }
        """
        if len(data) < 2:
            return {"error": "Insufficient data"}
        
        sample_mean = np.mean(data)
        
        # One-sample t-test
        t_stat, p_value = stats.ttest_1samp(data, hypothesized_mean)
        
        reject_h0 = bool(p_value < 0.05)
        
        return {
            "test": "One-sample t-test",
            "sample_mean": _to_python_type(round(sample_mean, 2)),
            "hypothesized_mean": hypothesized_mean,
            "t_statistic": _to_python_type(round(t_stat, 4)),
            "p_value": _to_python_type(round(p_value, 4)),
            "degrees_of_freedom": int(len(data) - 1),
            "reject_h0": reject_h0,
            "alpha": 0.05,
            "conclusion": f"{'Bác bỏ' if reject_h0 else 'Không bác bỏ'} H0",
            "interpretation": f"Mean mẫu ({sample_mean:.2f}) {'CÓ' if reject_h0 else 'KHÔNG CÓ'} sự khác biệt có ý nghĩa với {hypothesized_mean} (p={p_value:.3f})"
        }
    
    @staticmethod
    def compare_countries(stats1: Dict, stats2: Dict) -> Dict:
        """
        So sánh 2 quốc gia bằng logic thủ công
        """
        comparison = {
            "birth_rate_comparison": {},
            "death_rate_comparison": {},
            "trend_comparison": {},
            "demographic_comparison": {}
        }
        
        # Birth rate comparison
        b1 = stats1["birth_rate_analysis"]["mean"]
        b2 = stats2["birth_rate_analysis"]["mean"]
        comparison["birth_rate_comparison"] = {
            f"{stats1['country']}_mean": b1,
            f"{stats2['country']}_mean": b2,
            "difference": round(abs(b1 - b2), 2),
            "higher": stats1['country'] if b1 > b2 else stats2['country'],
            "percent_difference": round(abs(b1 - b2) / min(b1, b2) * 100, 2) if min(b1, b2) > 0 else None
        }
        
        # Trend comparison
        t1_birth = stats1["trend_analysis"]["birth_rate"]
        t2_birth = stats2["trend_analysis"]["birth_rate"]
        
        comparison["trend_comparison"] = {
            f"{stats1['country']}_trend": t1_birth["direction"],
            f"{stats2['country']}_trend": t2_birth["direction"],
            "same_direction": t1_birth["direction"] == t2_birth["direction"],
            f"{stats1['country']}_confidence": t1_birth["confidence"],
            f"{stats2['country']}_confidence": t2_birth["confidence"],
            f"{stats1['country']}_r_squared": t1_birth["r_squared"],
            f"{stats2['country']}_r_squared": t2_birth["r_squared"]
        }
        
        # Demographic stage comparison
        comparison["demographic_comparison"] = {
            f"{stats1['country']}_stage": stats1["demographic_indicators"]["demographic_stage"],
            f"{stats2['country']}_stage": stats2["demographic_indicators"]["demographic_stage"],
            "same_stage": stats1["demographic_indicators"]["demographic_stage"] == stats2["demographic_indicators"]["demographic_stage"]
        }
        
        return comparison
    
    @staticmethod
    def generate_summary_table(stats: Dict) -> pd.DataFrame:
        """
        Tạo bảng tóm tắt để hiển thị trong UI
        """
        rows = []
        
        # Row 1: Birth Rate
        rows.append({
            "Chỉ số": "Tỉ lệ sinh (‰)",
            "Trung bình": stats["birth_rate_analysis"]["mean"],
            "Min": stats["birth_rate_analysis"]["min"],
            "Max": stats["birth_rate_analysis"]["max"],
            "Xu hướng": stats["trend_analysis"]["birth_rate"]["direction"],
            "Độ tin cậy": stats["trend_analysis"]["birth_rate"]["confidence"]
        })
        
        # Row 2: Death Rate
        rows.append({
            "Chỉ số": "Tỉ lệ tử (‰)",
            "Trung bình": stats["death_rate_analysis"]["mean"],
            "Min": stats["death_rate_analysis"]["min"],
            "Max": stats["death_rate_analysis"]["max"],
            "Xu hướng": stats["trend_analysis"]["death_rate"]["direction"],
            "Độ tin cậy": stats["trend_analysis"]["death_rate"]["confidence"]
        })
        
        # Row 3: Natural Increase
        rows.append({
            "Chỉ số": "Tăng tự nhiên (‰)",
            "Trung bình": stats["demographic_indicators"]["natural_increase_rate"],
            "Min": "-",
            "Max": "-",
            "Xu hướng": "-",
            "Độ tin cậy": "-"
        })
        
        return pd.DataFrame(rows)
