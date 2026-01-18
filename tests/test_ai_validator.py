"""
Test Suite cho AI Hallucination Detector
Kiểm tra logic phát hiện AI bịa nội dung
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from be.ai_hallucination_detector import AIHallucinationDetector


def test_extract_numbers():
    """Test trích xuất số từ markdown"""
    print("\n=== Test 1: Extract Numbers ===")
    
    markdown = """
    Tỉ lệ sinh Việt Nam là 15.2‰ trong giai đoạn 2015-2025.
    Tỉ lệ tử khoảng 5.8‰.
    Tăng tự nhiên là 9.4‰, cho thấy dân số tăng 2.5% mỗi năm.
    """
    
    detector = AIHallucinationDetector()
    numbers = detector.extract_numbers_from_report(markdown)
    
    # Assertions
    assert len(numbers) > 0
    values = [n['value'] for n in numbers]
    assert 15.2 in values
    assert 5.8 in values
    assert 9.4 in values
    
    print(f"✅ PASS: Extracted {len(numbers)} numbers")
    for num in numbers[:5]:  # Show first 5
        print(f"   {num['value']}{num['unit']} - {num['context'][:50]}...")


def test_verify_statistics_correct():
    """Test verify với số liệu ĐÚNG"""
    print("\n=== Test 2: Verify Statistics (Correct) ===")
    
    # AI report với số ĐÚNG
    ai_report = """
    Tỉ lệ sinh trung bình là 15.2‰.
    Tỉ lệ tử trung bình là 6.0‰.
    Tăng tự nhiên là 9.2‰.
    """
    
    actual_stats = {
        "birth_rate_avg": 15.2,
        "death_rate_avg": 6.0,
        "natural_increase": 9.2
    }
    
    detector = AIHallucinationDetector()
    result = detector.verify_ai_statistics(ai_report, actual_stats, tolerance_percent=5.0)
    
    # Assertions
    assert 'verified' in result
    assert 'hallucinations' in result
    assert result['accuracy_score'] > 80  # Should be high
    assert len(result['verified']) >= 2  # At least 2 correct
    
    print(f"✅ PASS: Accuracy = {result['accuracy_score']}/100")
    print(f"   Verified: {len(result['verified'])}")
    print(f"   Hallucinations: {len(result['hallucinations'])}")


def test_verify_statistics_hallucination():
    """Test verify với số liệu SAI (AI bịa)"""
    print("\n=== Test 3: Verify Statistics (Hallucination) ===")
    
    # AI report với số SAI HOÀN TOÀN
    ai_report = """
    Tỉ lệ sinh trung bình là 50.5‰.
    Tỉ lệ tử trung bình là 25.8‰.
    Tăng tự nhiên là 100.0‰.
    """
    
    actual_stats = {
        "birth_rate_avg": 15.2,  # Thực tế ~15
        "death_rate_avg": 6.0,   # Thực tế ~6
        "natural_increase": 9.2   # Thực tế ~9
    }
    
    detector = AIHallucinationDetector()
    result = detector.verify_ai_statistics(ai_report, actual_stats, tolerance_percent=5.0)
    
    # Assertions
    assert len(result['hallucinations']) > 0  # Should detect hallucinations
    assert result['accuracy_score'] < 50  # Should be low
    
    print(f"✅ PASS: Detected hallucinations")
    print(f"   Accuracy = {result['accuracy_score']}/100 (LOW as expected)")
    print(f"   Hallucinations: {len(result['hallucinations'])}")
    for h in result['hallucinations']:
        print(f"   - {h['stat']}: AI={h.get('ai_value', 'N/A')}, Actual={h.get('actual_value', 'N/A')}")


def test_check_trend_accuracy_correct():
    """Test kiểm tra xu hướng ĐÚNG"""
    print("\n=== Test 4: Check Trend Accuracy (Correct) ===")
    
    ai_report = """
    Tỉ lệ sinh của Việt Nam đang giảm dần qua các năm.
    Xu hướng giảm này cho thấy...
    """
    
    actual_trend = "decreasing"
    
    detector = AIHallucinationDetector()
    result = detector.check_trend_accuracy(ai_report, actual_trend, "Viet Nam")
    
    # Assertions
    assert result['correct'] == True
    assert result['ai_claimed'] == 'decreasing'
    assert result['actual'] == 'decreasing'
    
    print(f"✅ PASS: Trend check correct")
    print(f"   AI claimed: {result['ai_claimed']}")
    print(f"   Actual: {result['actual']}")
    print(f"   Verdict: {result['verdict']}")


def test_check_trend_accuracy_wrong():
    """Test kiểm tra xu hướng SAI"""
    print("\n=== Test 5: Check Trend Accuracy (Wrong) ===")
    
    # AI nói TĂNG nhưng thực tế GIẢM
    ai_report = """
    Tỉ lệ sinh của Việt Nam đang tăng mạnh.
    Xu hướng tăng trưởng này...
    """
    
    actual_trend = "decreasing"  # Thực tế giảm
    
    detector = AIHallucinationDetector()
    result = detector.check_trend_accuracy(ai_report, actual_trend, "Viet Nam")
    
    # Assertions
    assert result['correct'] == False  # AI SAI
    assert result['ai_claimed'] == 'increasing'
    assert result['actual'] == 'decreasing'
    
    print(f"✅ PASS: Detected wrong trend")
    print(f"   AI claimed: {result['ai_claimed']}")
    print(f"   Actual: {result['actual']}")
    print(f"   Verdict: {result['verdict']}")


def test_detect_contradictions():
    """Test phát hiện mâu thuẫn"""
    print("\n=== Test 6: Detect Contradictions ===")
    
    # AI mâu thuẫn: trước nói tăng, sau nói giảm
    ai_report = """
    ## Phần 1
    Tỉ lệ sinh đang tăng mạnh trong giai đoạn này.
    
    ## Phần 2
    Như vậy, xu hướng giảm của tỉ lệ sinh cho thấy...
    """
    
    detector = AIHallucinationDetector()
    contradictions = detector.detect_contradictions(ai_report)
    
    # Assertions
    assert len(contradictions) > 0  # Should detect contradiction
    
    print(f"✅ PASS: Detected {len(contradictions)} contradictions")
    for c in contradictions:
        print(f"   Type: {c['type']}")
        print(f"   Statement 1: {c['statement_1'][:50]}...")
        print(f"   Statement 2: {c['statement_2'][:50]}...")


def test_generate_validation_report():
    """Test tạo báo cáo validation đầy đủ"""
    print("\n=== Test 7: Generate Validation Report ===")
    
    # Mock verification result
    verification = {
        'total_stats': 3,
        'correct_stats': 2,
        'suspicious_stats': 0,
        'hallucinated_stats': 1,
        'accuracy_score': 66.7,
        'verified': [
            {'stat': 'birth_rate', 'ai_value': 15.2, 'actual_value': 15.2}
        ],
        'suspicious': [],
        'hallucinations': [
            {'stat': 'death_rate', 'ai_value': 25.0, 'actual_value': 6.0}
        ]
    }
    
    trend_check = {
        'correct': True,
        'ai_claimed': 'decreasing',
        'actual': 'decreasing'
    }
    
    contradictions = []
    
    detector = AIHallucinationDetector()
    report = detector.generate_validation_report(verification, trend_check, contradictions)
    
    # Assertions
    assert 'hallucination_score' in report
    assert 'verdict' in report
    assert 'recommendations' in report
    assert report['verdict'] in ['PASS', 'WARNING', 'FAIL']
    
    print(f"✅ PASS: Generated validation report")
    print(f"   Hallucination Score: {report['hallucination_score']}/100")
    print(f"   Verdict: {report['verdict_emoji']} {report['verdict']}")
    print(f"   Message: {report['message']}")
    print(f"   Recommendations:")
    for rec in report['recommendations']:
        print(f"     - {rec}")


def run_all_tests():
    """Chạy tất cả tests"""
    print("\n" + "="*60)
    print("AI HALLUCINATION DETECTOR TEST SUITE")
    print("="*60)
    
    try:
        test_extract_numbers()
        test_verify_statistics_correct()
        test_verify_statistics_hallucination()
        test_check_trend_accuracy_correct()
        test_check_trend_accuracy_wrong()
        test_detect_contradictions()
        test_generate_validation_report()
        
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
