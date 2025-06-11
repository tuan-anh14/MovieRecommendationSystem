# 🎬 HƯỚNG DẪN CHẠY DỰ ÁN MOVIE RECOMMENDER SYSTEM

## 📋 **BƯỚC 1: CHUẨN BỊ MÔI TRƯỜNG**

### **1.1 Yêu cầu hệ thống:**
- **Python 3.7+** (khuyến nghị Python 3.8 hoặc 3.9)
- **Git** để clone repository
- **Internet connection** để download datasets và gọi API
- **4GB RAM** tối thiểu (8GB khuyến nghị)

### **1.2 Cài đặt Python và pip:**
```bash
# Kiểm tra version Python
python --version
python -m pip --version

# Nếu chưa có Python, download từ python.org
```

---

## 📥 **BƯỚC 2: DOWNLOAD DỰ ÁN**

### **2.1 Clone repository:**
```bash
# Mở terminal/command prompt
git clone https://github.com/kishan0725/AJAX-Movie-Recommendation-System-with-Sentiment-Analysis.git

# Hoặc download ZIP từ GitHub và extract
```

### **2.2 Vào thư mục project:**
```bash
cd AJAX-Movie-Recommendation-System-with-Sentiment-Analysis
```

---

## 🔧 **BƯỚC 3: CÀI ĐẶT DEPENDENCIES**

### **3.1 Tạo virtual environment (khuyến nghị):**
```bash
# Tạo virtual environment
python -m venv movie_recommender_env

# Kích hoạt virtual environment
# Windows:
movie_recommender_env\Scripts\activate
# macOS/Linux:
source movie_recommender_env/bin/activate
```

### **3.2 Cài đặt packages:**
```bash
# Upgrade pip
pip install --upgrade pip

# Cài đặt từ requirements.txt
pip install -r requirements.txt

# Nếu có lỗi, cài đặt từng package:
pip install pandas numpy scikit-learn flask beautifulsoup4 requests nltk

#Cài đặt thư viện cần từ requirements
pip install -r requirements.txt --force-reinstall
#Cập nhật tránh xung đột phiên bản
pip install --upgrade urllib3 requests
```


### **3.3 Kiểm tra cài đặt:**
```bash
python -c "import pandas, numpy, sklearn, flask; print('All packages installed successfully!')"
```

---

## 🔑 **BƯỚC 4: LẤY API KEY TỪ TMDB**

### **4.1 Đăng ký tài khoản TMDB:**
1. Vào **https://www.themoviedb.org/**
2. Click **"Sign Up"** và tạo tài khoản
3. Xác nhận email

### **4.2 Xin API Key:**
1. Đăng nhập vào tài khoản
2. Vào **Settings → API** (từ sidebar bên trái)
3. Click **"Create"** để tạo API key mới
4. Chọn **"Developer"** 
5. Điền thông tin:
   - **Application Name:** Movie Recommender System
   - **Application URL:** NA (nếu không có website)
   - **Application Summary:** Movie recommendation project for education
6. Submit và đợi approve (thường tức thì)

### **4.3 Copy API Key:**
```
# API Key sẽ có dạng:
# 1234567890abcdef1234567890abcdef
```

---

## ⚙️ **BƯỚC 5: CẤU HÌNH PROJECT**

### **5.1 Cập nhật API Key:**
1. Mở file **`static/recommend.js`**
2. Tìm dòng 15 và 29 có **`YOUR_API_KEY`**
3. Thay thế bằng API key của bạn:

```javascript
// Dòng 15:
const api_key = 'YOUR_API_KEY_HERE';

// Dòng 29:
const api_key = 'YOUR_API_KEY_HERE';
```

### **5.2 Kiểm tra cấu trúc thư mục:**
```
Movie-Recommendation-System/
├── main.py
├── requirements.txt
├── static/
│   ├── recommend.js
│   ├── style.css
│   └── ...
├── templates/
│   ├── index.html
│   └── ...
└── datasets/
    └── ...
```

---

## 🚀 **BƯỚC 6: CHẠY DỰ ÁN**

### **6.1 Khởi động server:**
```bash
# Đảm bảo đang ở thư mục gốc của project
python main.py
```

### **6.2 Kiểm tra output:**
```
* Running on http://127.0.0.1:5000/
* Debug mode: on
* Restarting with stat
* Debugger is active!
```

### **6.3 Truy cập ứng dụng:**
1. Mở trình duyệt
2. Vào địa chỉ: **http://127.0.0.1:5000/**
3. Hoặc: **http://localhost:5000/**

---

## 🎯 **BƯỚC 7: KIỂM TRA HOẠT ĐỘNG**

### **7.1 Test cơ bản:**
1. Nhập tên phim (ví dụ: "Avengers")
2. Chọn phim từ dropdown
3. Click **"Recommend"**
4. Xem kết quả recommendation và sentiment analysis

### **7.2 Kiểm tra các tính năng:**
- ✅ **Movie search với autocomplete**
- ✅ **Movie recommendations (10 phim tương tự)**
- ✅ **Movie details** (poster, rating, genre, etc.)
- ✅ **Sentiment analysis** của reviews
- ✅ **AJAX loading** không reload page

---

## 🐛 **BƯỚC 8: XỬ LÝ LỖI THƯỜNG GẶP**

### **8.1 Lỗi "Module not found":**
```bash
# Cài đặt lại packages
pip install -r requirements.txt --force-reinstall
```

### **8.2 Lỗi "API Key invalid":**
- Kiểm tra lại API key trong `static/recommend.js`
- Đảm bảo API key đã được approve
- Thử tạo API key mới

### **8.3 Lỗi "Port already in use":**
```bash
# Thay đổi port trong main.py
app.run(debug=True, port=5001)  # Thay 5000 thành 5001
```

### **8.4 Lỗi "Memory Error":**
- Đóng các ứng dụng khác đang chạy
- Restart máy tính và thử lại
- Sử dụng dataset nhỏ hơn

### **8.5 Lỗi "Network timeout":**
- Kiểm tra kết nối internet
- Thử lại sau vài phút
- Sử dụng VPN nếu cần

---

## 📱 **BƯỚC 9: DEMO VÀ TESTING**

### **9.1 Các phim để test:**
- **Popular movies:** "Avengers", "Titanic", "Avatar"
- **Classic movies:** "The Godfather", "Pulp Fiction"
- **Recent movies:** "Joker", "1917", "Parasite"

### **9.2 Screenshots để báo cáo:**
1. **Home page** với search box
2. **Movie selection** dropdown
3. **Recommendation results** với posters
4. **Sentiment analysis** results
5. **Network tab** showing AJAX calls

---

## 🎨 **BƯỚC 10: CUSTOMIZATION (OPTIONAL)**

### **10.1 Thay đổi giao diện:**
- Edit `templates/index.html` cho HTML
- Edit `static/style.css` cho CSS
- Edit `static/recommend.js` cho JavaScript

### **10.2 Thêm features:**
- Thêm more recommendation algorithms
- Integrate user ratings
- Add movie trailers
- Export recommendations to PDF

---

## 📊 **BƯỚC 11: CHUẨN BỊ**

### **11.1 Screenshots cần có:**
- ✅ Project structure
- ✅ API key configuration
- ✅ Running application
- ✅ Movie search functionality
- ✅ Recommendation results
- ✅ Sentiment analysis output

### **11.2 Metrics để đo:**
- **Response time:** Thời gian load recommendations
- **Accuracy:** Độ chính xác của recommendations
- **User experience:** Ease of use, interface design

---

## 🔄 **BƯỚC 12: STOP APPLICATION**

```bash
# Trong terminal đang chạy server:
Ctrl + C  # Windows/Linux
Cmd + C   # macOS

# Deactivate virtual environment
deactivate
```


**🎉 CHÚC MỪNG! DỰ ÁN ĐÃ SETUP THÀNH CÔNG!**