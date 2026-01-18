"""
AI Hallucination Detector - LOGIC QUAN TRỌNG NHẤT
Kiểm chứng AI có bịa nội dung không bằng code thủ công
"""
import re
import pandas as pd
from typing import Dict, List, Tuple, Optional


class AIHallucinationDetector:
    """
    Phát hiện AI bịa nội dung - Core validation logic
    """
    
    @staticmethod
    def extract_numbers_from_report(markdown: str) -> List[Dict]:
        """
        Trích xuất TẤT CẢ số liệu từ markdown AI
        
        Returns:
            List of {value, unit, context, line_number}
        """
        numbers = []
        lines = markdown.split('\n')
        
        # Pattern: số thập phân + đơn vị
        pattern = r'(\d+(?:\.\d+)?)\s*(‰|%|triệu|tỷ|nghìn|người)?'
        
        for line_num, line in enumerate(lines, 1):
            for match in re.finditer(pattern, line):
                value = float(match.group(1))
                unit = match.group(2) or ""
                
                # Lấy context xung quanh
                start = max(0, match.start() - 30)
                end = min(len(line), match.end() + 30)
                context = line[start:end].strip()
                
                numbers.append({
                    "value": value,
                    "unit": unit,
                    "context": context,
                    "line_number": line_num,
                    "full_line": line.strip()
                })
        
        return numbers
    
    @staticmethod
    def verify_ai_statistics(
        ai_report: str,
        actual_stats: Dict[str, float],
        tolerance_percent: float = 5.0
    ) -> Dict:
        """
        So sánh số liệu AI vs thực tế - PHÁT HIỆN BỊA
        
        Args:
            ai_report: Markdown từ AI
            actual_stats: {"birth_rate_avg": 15.2, ...} từ code thủ công
            tolerance_percent: Sai số chấp nhận được (%)
        
        Returns:
            {
                "verified": [...],  # Số đúng ✓
                "suspicious": [...],  # Số sai lệch nhẹ ⚠️
                "hallucinations": [...],  # AI bịa ❌
                "accuracy_score": float
            }
        """
        extracted = AIHallucinationDetector.extract_numbers_from_report(ai_report)
        
        verified = []
        suspicious = []
        hallucinations = []
        
        for stat_name, actual_val in actual_stats.items():
            if actual_val is None:
                continue
            
            # Tìm số liệu tương ứng trong AI report
            candidates = []
            for num in extracted:
                # Chỉ xét số có cùng đơn vị hoặc gần giá trị
                if abs(num['value'] - actual_val) < actual_val * 0.5:  # Trong vòng 50%
                    error_pct = abs(num['value'] - actual_val) / actual_val * 100
                    candidates.append({
                        **num,
                        "stat_name": stat_name,
                        "actual_value": actual_val,
                        "error_pct": error_pct
                    })
            
            if not candidates:
                # AI không nhắc đến số này - có thể bỏ sót
                hallucinations.append({
                    "stat_name": stat_name,
                    "actual_value": actual_val,
                    "ai_value": None,
                    "status": "❌ MISSING",
                    "reason": "AI không đề cập đến số liệu này"
                })
                continue
            
            # Lấy số gần nhất
            best = min(candidates, key=lambda x: x['error_pct'])
            
            if best['error_pct'] <= tolerance_percent:
                # Đúng hoặc sai số nhỏ
                verified.append({
                    "stat_name": stat_name,
                    "ai_value": best['value'],
                    "actual_value": actual_val,
                    "error_pct": round(best['error_pct'], 2),
                    "status": "✓ VERIFIED",
                    "context": best['context']
                })
            elif best['error_pct'] <= tolerance_percent * 3:
                # Sai lệch đáng ngờ
                suspicious.append({
                    "stat_name": stat_name,
                    "ai_value": best['value'],
                    "actual_value": actual_val,
                    "error_pct": round(best['error_pct'], 2),
                    "status": "⚠️ SUSPICIOUS",
                    "reason": f"Sai lệch {best['error_pct']:.1f}% - có thể AI làm tròn hoặc tính sai",
                    "context": best['context']
                })
            else:
                # SAI HOÀN TOÀN - AI bịa
                hallucinations.append({
                    "stat_name": stat_name,
                    "ai_value": best['value'],
                    "actual_value": actual_val,
                    "error_pct": round(best['error_pct'], 2),
                    "status": "❌ HALLUCINATION",
                    "reason": f"SAI {best['error_pct']:.1f}% - AI đã bịa số liệu",
                    "context": best['context']
                })
        
        # Tính accuracy score
        total = len(actual_stats)
        correct = len(verified)
        accuracy_score = (correct / total * 100) if total > 0 else 0
        
        return {
            "verified": verified,
            "suspicious": suspicious,
            "hallucinations": hallucinations,
            "accuracy_score": round(accuracy_score, 1),
            "total_stats": total,
            "correct_stats": correct,
            "suspicious_stats": len(suspicious),
            "hallucinated_stats": len(hallucinations)
        }
    
    @staticmethod
    def check_trend_accuracy(
        ai_report: str,
        actual_trend: str,
        country: str
    ) -> Dict:
        """
        Kiểm tra AI nói "tăng/giảm" có đúng không
        PHÁT HIỆN AI NÓI SAI XU HƯỚNG
        
        Args:
            ai_report: Markdown từ AI
            actual_trend: "increasing" | "decreasing" | "stable" từ code thủ công
            country: Tên quốc gia
        
        Returns:
            {
                "correct": bool,
                "ai_claimed": str,
                "actual": str,
                "evidence": List[str],
                "verdict": str
            }
        """
        ai_lower = ai_report.lower()
        
        # Keywords
        increasing_kw = ['tăng', 'gia tăng', 'tăng trưởng', 'tăng lên', 'rising', 'increase', 'upward']
        decreasing_kw = ['giảm', 'sụt giảm', 'giảm dần', 'giảm xuống', 'falling', 'decrease', 'decline', 'downward']
        stable_kw = ['ổn định', 'stable', 'không đổi', 'flat', 'unchanged']
        
        # Tìm tất cả mentions
        increasing_mentions = [kw for kw in increasing_kw if kw in ai_lower]
        decreasing_mentions = [kw for kw in decreasing_kw if kw in ai_lower]
        stable_mentions = [kw for kw in stable_kw if kw in ai_lower]
        
        # Determine AI's claim
        ai_claimed = None
        evidence = []
        
        if len(increasing_mentions) > len(decreasing_mentions) and len(increasing_mentions) > len(stable_mentions):
            ai_claimed = "increasing"
            evidence = increasing_mentions
        elif len(decreasing_mentions) > len(increasing_mentions) and len(decreasing_mentions) > len(stable_mentions):
            ai_claimed = "decreasing"
            evidence = decreasing_mentions
        elif len(stable_mentions) > 0:
            ai_claimed = "stable"
            evidence = stable_mentions
        else:
            ai_claimed = "unknown"
            evidence = []
        
        # Check correctness
        correct = (ai_claimed == actual_trend)
        
        # Verdict
        if correct:
            verdict = f"✓ AI ĐÚNG - Xu hướng thực tế là '{actual_trend}', AI nói '{ai_claimed}'"
        else:
            verdict = f"❌ AI SAI - Xu hướng thực tế là '{actual_trend}', nhưng AI nói '{ai_claimed}'"
        
        return {
            "correct": correct,
            "ai_claimed": ai_claimed,
            "actual": actual_trend,
            "evidence": evidence,
            "verdict": verdict,
            "severity": "CRITICAL" if not correct else "OK"
        }
    
    @staticmethod
    def detect_contradictions(ai_report: str) -> List[Dict]:
        """
        Phát hiện AI mâu thuẫn nội bộ
        VD: Trước nói "tỉ lệ sinh tăng", sau nói "tỉ lệ sinh giảm"
        
        FIX: Chỉ detect contradiction khi nói về CÙNG MỘT CHỈ SỐ
        """
        contradictions = []
        lines = ai_report.split('\n')
        
        # Keywords cho từng metric
        metrics_keywords = {
            'birth_rate': ['tỉ lệ sinh', 'birth rate', 'sinh sản', 'mức sinh'],
            'death_rate': ['tỉ lệ tử', 'death rate', 'tử vong', 'mức tử'],
            'population': ['dân số', 'population', 'số dân']
        }
        
        increasing_kw = ['tăng', 'gia tăng', 'tăng trưởng', 'tăng lên', 'rising', 'increase', 'upward']
        decreasing_kw = ['giảm', 'sụt giảm', 'giảm dần', 'giảm xuống', 'falling', 'decrease', 'decline', 'downward']
        
        # Tìm statements về từng metric
        metric_statements = {metric: [] for metric in metrics_keywords}
        
        for line_num, line in enumerate(lines, 1):
            line_lower = line.lower()
            
            # Chỉ xét lines có đề cập xu hướng
            has_trend = any(kw in line_lower for kw in increasing_kw + decreasing_kw)
            if not has_trend:
                continue
            
            # Xác định metric nào đang được nói đến
            for metric, keywords in metrics_keywords.items():
                if any(kw in line_lower for kw in keywords):
                    # Xác định trend direction
                    has_increase = any(kw in line_lower for kw in increasing_kw)
                    has_decrease = any(kw in line_lower for kw in decreasing_kw)
                    
                    if has_increase and not has_decrease:
                        direction = 'increasing'
                    elif has_decrease and not has_increase:
                        direction = 'decreasing'
                    else:
                        # Có cả tăng và giảm trong cùng câu - bỏ qua (likely comparison)
                        continue
                    
                    metric_statements[metric].append({
                        'line_number': line_num,
                        'statement': line.strip(),
                        'direction': direction
                    })
        
        # Kiểm tra contradiction cho từng metric
        for metric, statements in metric_statements.items():
            if len(statements) < 2:
                continue
            
            # So sánh các statements về CÙNG metric
            for i, stmt1 in enumerate(statements):
                for stmt2 in statements[i+1:]:
                    # Nếu 2 statements về cùng metric nhưng ngược chiều → contradiction
                    if stmt1['direction'] != stmt2['direction']:
                        contradictions.append({
                            'type': f'{metric}_contradiction',
                            'metric': metric,
                            'statement_1': stmt1['statement'],
                            'statement_2': stmt2['statement'],
                            'direction_1': stmt1['direction'],
                            'direction_2': stmt2['direction'],
                            'line_1': stmt1['line_number'],
                            'line_2': stmt2['line_number'],
                            'severity': 'HIGH',
                            'explanation': f"AI nói mâu thuẫn về {metric}: một chỗ nói {stmt1['direction']}, chỗ khác nói {stmt2['direction']}"
                        })
        
        print(f"[DEBUG] Contradiction detection: found {len(contradictions)} real contradictions")
        return contradictions
    
    @staticmethod
    def calculate_hallucination_score(verification_result: Dict, trend_check: Dict, contradictions: List[Dict]) -> float:
        """
        Tính điểm tổng thể - 0 = toàn bịa, 100 = hoàn hảo
        
        ADJUSTED FORMULA:
        - Base accuracy: from statistics verification (0-100)
        - Hallucination penalty: 15 per fake number (severe)
        - Trend penalty: 10 if wrong trend (moderate)
        - Contradiction penalty: 3 per contradiction, max 30 (light, capped)
        """
        # Base accuracy
        accuracy = verification_result['accuracy_score']
        print(f"[DEBUG] Hallucination score calculation:")
        print(f"  Base accuracy: {accuracy}")
        
        # Penalty for hallucinations (SEVERE - AI bịa số liệu)
        hallucination_penalty = verification_result['hallucinated_stats'] * 15
        print(f"  Hallucination penalty: {hallucination_penalty} ({verification_result['hallucinated_stats']} * 15)")
        
        # Penalty for wrong trend (MODERATE - AI sai xu hướng)
        trend_penalty = 10 if not trend_check['correct'] else 0
        print(f"  Trend penalty: {trend_penalty} (correct={trend_check['correct']})")
        
        # Penalty for contradictions (LIGHT, CAPPED - AI mâu thuẫn nội bộ)
        contradiction_penalty = min(len(contradictions) * 3, 30)  # Max 30 points
        print(f"  Contradiction penalty: {contradiction_penalty} ({len(contradictions)} * 3, capped at 30)")
        
        # Calculate final score
        final_score = max(0, accuracy - hallucination_penalty - trend_penalty - contradiction_penalty)
        print(f"  Final score: {final_score} = max(0, {accuracy} - {hallucination_penalty} - {trend_penalty} - {contradiction_penalty})")
        
        return round(final_score, 1)
    
    @staticmethod
    def generate_validation_report(
        verification_result: Dict,
        trend_check: Dict,
        contradictions: List[Dict]
    ) -> Dict:
        """
        Tạo báo cáo validation đầy đủ
        """
        hallucination_score = AIHallucinationDetector.calculate_hallucination_score(
            verification_result, trend_check, contradictions
        )
        
        # Determine verdict
        if hallucination_score >= 80:
            verdict = "PASS"
            verdict_emoji = "✅"
            message = "AI report đáng tin cậy"
        elif hallucination_score >= 60:
            verdict = "WARNING"
            verdict_emoji = "⚠️"
            message = "AI report có một số vấn đề nhỏ"
        else:
            verdict = "FAIL"
            verdict_emoji = "❌"
            message = "AI report KHÔNG đáng tin cậy - có nhiều sai sót"
        
        # Recommendations
        recommendations = []
        if verification_result['hallucinated_stats'] > 0:
            recommendations.append("❌ AI đã bịa một số số liệu - NÊN DÙNG SỐ LIỆU TỪ BẢNG THỐNG KÊ thay vì tin AI")
        if not trend_check['correct']:
            recommendations.append("❌ AI SAI về xu hướng - cần regenerate hoặc sửa prompt")
        if len(contradictions) > 0:
            recommendations.append("⚠️ AI có mâu thuẫn nội bộ - cần review kỹ")
        if verification_result['suspicious_stats'] > 0:
            recommendations.append("⚠️ Một số số liệu AI sai lệch - cần kiểm tra lại")
        if len(recommendations) == 0:
            recommendations.append("✅ AI report đạt chuẩn - có thể sử dụng")
        
        return {
            "hallucination_score": hallucination_score,
            "verdict": verdict,
            "verdict_emoji": verdict_emoji,
            "message": message,
            "statistics_verification": verification_result,
            "trend_check": trend_check,
            "contradictions": contradictions,
            "recommendations": recommendations,
            "summary": {
                "total_stats_checked": verification_result['total_stats'],
                "verified_stats": verification_result['correct_stats'],
                "suspicious_stats": verification_result['suspicious_stats'],
                "hallucinated_stats": verification_result['hallucinated_stats'],
                "trend_correct": trend_check['correct'],
                "contradictions_found": len(contradictions)
            }
        }
