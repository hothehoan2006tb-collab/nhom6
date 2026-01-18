"""
Test Suite cho Statistical Processor
Kiểm tra tất cả logic xử lý thống kê thủ công
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from be.statistical_processor import StatisticalProcessor


def test_process_country_statistics():
    """Test hàm chính xử lý thống kê"""
    print("\n=== Test 1: process_country_statistics ===")
    
    # Mock data
    data = {
        'year': [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
        'birth_rate': [16.5, 16.2, 15.9, 15.6, 15.3, 15.0, 14.7, 14.4, 14.1, 13.8],
        'death_rate': [6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9],
        'population': [91000000, 92000000, 93000000, 94000000, 95000000, 
                      96000000, 97000000, 98000000, 99000000, 100000000]
    }
    df = pd.DataFrame(data)
    
    processor = StatisticalProcessor()
    result = processor.process_country_statistics(df, "Test Country")
    
    # Assertions
    assert result['country'] == "Test Country"
    assert 'birth_rate_analysis' in result
    assert 'trend_analysis' in result
    assert 'demographic_indicators' in result
    assert 'correlation_analysis' in result
    assert 'normality_tests' in result
    assert 'hypothesis_tests' in result
    
    # Check birth rate analysis
    birth = result['birth_rate_analysis']
    assert 13.0 < birth['mean'] < 17.0  # Expected around 15
    assert birth['min'] == 13.8
    assert birth['max'] == 16.5
    
    # Check trend (should be decreasing)
    trend = result['trend_analysis']['birth_rate']
    assert trend['direction'] == 'decreasing'
    assert trend['r_squared'] > 0.9  # Strong trend
    assert trend['p_value'] < 0.05  # Significant
    
    # Check demographic indicators
    demo = result['demographic_indicators']
    assert demo['natural_increase_rate'] > 0  # Birth > Death
    assert 'birth_rate_95ci' in demo
    
    # Check confidence interval
    ci = demo['birth_rate_95ci']
    assert ci['lower_bound'] < ci['mean'] < ci['upper_bound']
    assert ci['margin_of_error'] > 0
    
    print("✅ PASS: process_country_statistics")
    print(f"   Birth rate mean: {birth['mean']}‰")
    print(f"   Trend: {trend['direction']} (R²={trend['r_squared']:.3f})")
    print(f"   95% CI: [{ci['lower_bound']}, {ci['upper_bound']}]")
    return result


def test_confidence_interval():
    """Test tính confidence interval"""
    print("\n=== Test 2: Confidence Interval ===")
    
    # Mock data với known distribution
    data = pd.Series([14, 15, 16, 15, 14, 16, 15, 14, 15, 16])
    
    ci = StatisticalProcessor._calculate_confidence_interval(data, confidence=0.95)
    
    # Assertions
    assert 'mean' in ci
    assert 'lower_bound' in ci
    assert 'upper_bound' in ci
    assert ci['lower_bound'] < ci['mean'] < ci['upper_bound']
    assert ci['confidence_level'] == 0.95
    
    # Mean should be 15
    assert abs(ci['mean'] - 15.0) < 0.1
    
    print("✅ PASS: Confidence Interval")
    print(f"   Mean: {ci['mean']}")
    print(f"   95% CI: [{ci['lower_bound']}, {ci['upper_bound']}]")
    print(f"   Margin of error: {ci['margin_of_error']}")


def test_correlation_analysis():
    """Test phân tích tương quan"""
    print("\n=== Test 3: Correlation Analysis ===")
    
    # Data với negative correlation
    x = np.array([16, 15, 14, 13, 12, 11, 10, 9, 8, 7])
    y = np.array([6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5])
    
    corr = StatisticalProcessor._analyze_correlation(x, y)
    
    # Assertions
    assert 'pearson_r' in corr
    assert 'spearman_rho' in corr
    assert corr['pearson_r'] < 0  # Negative correlation
    assert abs(corr['pearson_r']) > 0.9  # Strong correlation
    assert corr['correlation_direction'] == 'negative'
    assert corr['correlation_strength'] in ['strong', 'moderate', 'weak', 'very weak']
    
    print("✅ PASS: Correlation Analysis")
    print(f"   Pearson r: {corr['pearson_r']:.3f} (p={corr['pearson_p_value']:.4f})")
    print(f"   Interpretation: {corr['interpretation']}")


def test_normality_test():
    """Test kiểm định phân phối chuẩn"""
    print("\n=== Test 4: Normality Test ===")
    
    # Normal distribution
    np.random.seed(42)
    normal_data = np.random.normal(15, 2, 50)
    
    result = StatisticalProcessor._test_normality(normal_data)
    
    # Assertions
    assert 'test' in result
    assert 'p_value' in result
    assert 'is_normal' in result
    assert result['test'] == 'Shapiro-Wilk'
    
    print("✅ PASS: Normality Test")
    print(f"   Test: {result['test']}")
    print(f"   p-value: {result['p_value']:.4f}")
    print(f"   Is normal: {result['is_normal']}")
    print(f"   {result['interpretation']}")


def test_hypothesis_testing():
    """Test kiểm định giả thuyết"""
    print("\n=== Test 5: Hypothesis Testing ===")
    
    # Data với mean ~15 (khác 20)
    data = np.array([14, 15, 16, 15, 14, 16, 15, 14, 15, 16])
    
    result = StatisticalProcessor._one_sample_t_test(data, hypothesized_mean=20.0)
    
    # Assertions
    assert 'sample_mean' in result
    assert 'hypothesized_mean' in result
    assert 'p_value' in result
    assert 'reject_h0' in result
    assert result['hypothesized_mean'] == 20.0
    assert abs(result['sample_mean'] - 15.0) < 0.1
    assert result['reject_h0'] == True  # Should reject H0: mean ≠ 20
    
    print("✅ PASS: Hypothesis Testing")
    print(f"   Sample mean: {result['sample_mean']}")
    print(f"   H0: μ = {result['hypothesized_mean']}")
    print(f"   t-stat: {result['t_statistic']:.3f}")
    print(f"   p-value: {result['p_value']:.4f}")
    print(f"   Conclusion: {result['conclusion']}")


def test_compare_countries():
    """Test so sánh 2 quốc gia"""
    print("\n=== Test 6: Compare Countries ===")
    
    # Create stats for 2 countries
    stats1 = {
        'country': 'Country A',
        'birth_rate_analysis': {'mean': 15.2},
        'trend_analysis': {
            'birth_rate': {'direction': 'decreasing', 'confidence': 'high', 'r_squared': 0.92}
        },
        'demographic_indicators': {'demographic_stage': 'Stage 3'}
    }
    
    stats2 = {
        'country': 'Country B',
        'birth_rate_analysis': {'mean': 20.5},
        'trend_analysis': {
            'birth_rate': {'direction': 'stable', 'confidence': 'medium', 'r_squared': 0.45}
        },
        'demographic_indicators': {'demographic_stage': 'Stage 2'}
    }
    
    comparison = StatisticalProcessor.compare_countries(stats1, stats2)
    
    # Assertions
    assert 'birth_rate_comparison' in comparison
    assert 'trend_comparison' in comparison
    assert comparison['birth_rate_comparison']['higher'] == 'Country B'
    assert comparison['birth_rate_comparison']['difference'] > 0
    
    print("✅ PASS: Compare Countries")
    print(f"   Difference: {comparison['birth_rate_comparison']['difference']}‰")
    print(f"   Higher: {comparison['birth_rate_comparison']['higher']}")


def run_all_tests():
    """Chạy tất cả tests"""
    print("\n" + "="*60)
    print("STATISTICAL PROCESSOR TEST SUITE")
    print("="*60)
    
    try:
        test_process_country_statistics()
        test_confidence_interval()
        test_correlation_analysis()
        test_normality_test()
        test_hypothesis_testing()
        test_compare_countries()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED")
        print("="*60)
        return True
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
