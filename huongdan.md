# HƯỚNG DẪN CHẠY CODE - PHÂN TÍCH DÂN SỐ

## YÊU CẦU HỆ THỐNG

- **Python**: 3.8 trở lên
- **Pip**: Phiên bản mới nhất
- **API Keys**: Groq API key

---

## BƯỚC 1: CÀI ĐẶT

### 1.1. Mở Terminal tại thư mục project

```powershell
cd d:\laptrinhkhoahoc\nhom6
```

### 1.2. Cài đặt dependencies

```powershell
pip install -r requirements.txt
```

---

## BƯỚC 2: CẤU HÌNH API KEY

### 2.1. Mở file `.env`

```powershell
notepad be\.env
```

### 2.2. Cấu hình

```env
OPEN_MODEL=2
GROQ_API_KEY=gsk_your_key_here
BACKEND_URL=http://localhost:8001
```

**Lấy key:** https://console.groq.com/keys

---

## BƯỚC 3: CHẠY BACKEND

### Terminal 1:

```powershell
cd d:\laptrinhkhoahoc\nhom6
python -m uvicorn be.api:app --reload --port 8001
```

**Thành công khi thấy:**
```
INFO: Uvicorn running on http://127.0.0.1:8001
```

**Kiểm tra:** http://localhost:8001/health

---

## BƯỚC 4: CHẠY FRONTEND

### Terminal 2 (MỞ MỚI):

```powershell
cd d:\laptrinhkhoahoc\nhom6
python -m streamlit run fe/main.py
```

**Browser tự động mở:** http://localhost:8501

---

## BƯỚC 5: SỬ DỤNG

```
1. Chọn quốc gia → Chọn năm
2. Click "Tải dữ liệu"
3. Click "Xử lý thống kê"
4. Click "Phân tích bằng AI"
5. Xem validation score
6. Nếu < 95: Click "Regenerate"
7. Export file
```

---

## XỬ LÝ LỖI

### Lỗi: "GROQ_API_KEY not set"
→ Check file `be\.env`, restart backend

### Lỗi: Port đã dùng
```powershell
netstat -ano | findstr :8001
taskkill /F /PID [PID]
```

### Lỗi: Connection refused
→ Đảm bảo backend (Terminal 1) đang chạy

---

## DỪNG HỆ THỐNG

- Terminal 1 & 2: Nhấn `Ctrl + C`

---

## TÓM TẮT LỆNH

```powershell
# Setup (1 lần)
pip install -r requirements.txt
notepad be\.env

# Chạy hàng ngày
# Terminal 1:
python -m uvicorn be.api:app --reload --port 8001

# Terminal 2:
python -m streamlit run fe/main.py
```

**Truy cập:** http://localhost:8501 🚀
