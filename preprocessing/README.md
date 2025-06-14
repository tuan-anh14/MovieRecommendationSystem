# Hệ Thống Tiền Xử Lý Dữ Liệu Phim

## Tổng Quan
Đây là pipeline tiền xử lý dữ liệu cho hệ thống đề xuất phim, bao gồm 4 giai đoạn xử lý dữ liệu từ nhiều nguồn khác nhau.

## Cấu Trúc Pipeline

### Giai đoạn 1: `preprocessing_1.py`
**Mục đích**: Xử lý dữ liệu phim cơ bản từ movie_metadata.csv

**Kỹ thuật tiền xử lý**:
- **Đã học trên lớp**:
  - Xử lý giá trị thiếu (missing values)
  - Làm sạch văn bản (text cleaning)
  - Chuẩn hóa dữ liệu văn bản
  
- **Chưa học trên lớp**:
  - String manipulation với regex patterns
  - Text normalization techniques

**Input**: `../datasets/movie_metadata.csv`
**Output**: `../datasets/movies_stage1.csv`

### Giai đoạn 2: `preprocessing_2.py`
**Mục đích**: Xử lý dữ liệu phim 2017 từ TMDB API

**Kỹ thuật tiền xử lý**:
- **Đã học trên lớp**:
  - Merge/Join dữ liệu từ nhiều nguồn
  - Data type conversion
  - Text processing
  
- **Chưa học trên lớp**:
  - API data extraction (TMDB API)
  - JSON parsing với ast.literal_eval
  - Complex string manipulation

**Kỹ thuật trích chọn đặc trưng**:
- Trích xuất từ nested JSON structures
- Feature engineering từ cast/crew data
- Date/time feature extraction

**Input**: `../datasets/credits.csv`, `../datasets/movie_metadata.csv`
**Output**: `../datasets/movies_stage2.csv`

### Giai đoạn 3: `preprocessing_3.py`
**Mục đích**: Trích xuất dữ liệu phim 2018-2019 từ Wikipedia

**Kỹ thuật tiền xử lý**:
- **Đã học trên lớp**:
  - Web scraping với pandas.read_html()
  - Data concatenation và merging
  - Missing value handling
  
- **Chưa học trên lớp**:
  - Web scraping từ Wikipedia
  - TMDB API integration
  - Regular expressions cho pattern extraction

**Kỹ thuật trích chọn đặc trưng**:
- HTML table extraction
- Complex text parsing
- API data integration

**Input**: Wikipedia data (2018-2019)
**Output**: `../datasets/movies_stage3.csv`

### Giai đoạn 4: `preprocessing_4.py`
**Mục đích**: Trích xuất dữ liệu phim 2020 và hoàn thiện dataset

**Kỹ thuật tiền xử lý**:
- **Đã học trên lớp**:
  - Web scraping với BeautifulSoup
  - Data consolidation
  - String manipulation
  
- **Chưa học trên lớp**:
  - Advanced web scraping với urllib và BeautifulSoup
  - HTML parsing và table extraction
  - Complex data validation

**Kỹ thuật trích chọn đặc trưng**:
- HTML table parsing
- Feature consolidation từ multiple sources
- Data validation và cleansing

**Input**: Wikipedia data (2020) + previous stages
**Output**: `../datasets/movies_final_preprocessed.csv`

## Cách Sử Dụng

### 1. Chạy từng giai đoạn tuần tự:
```bash
cd preprocessing
python preprocessing_1.py
python preprocessing_2.py
python preprocessing_3.py
python preprocessing_4.py
```

### 2. Chạy tất cả các giai đoạn:
```bash
cd preprocessing
python run_all_preprocessing.py
```

## Cấu Trúc Dữ Liệu Output

Sau khi hoàn thành, dataset cuối cùng sẽ có các cột:
- `movie_title`: Tên phim (đã chuẩn hóa)
- `director_name`: Tên đạo diễn
- `actor_1_name`: Diễn viên chính thứ 1
- `actor_2_name`: Diễn viên chính thứ 2
- `actor_3_name`: Diễn viên chính thứ 3
- `genres`: Thể loại phim
- `comb`: Đặc trưng tổng hợp (kết hợp tất cả thông tin)

## Yêu Cầu Thư Viện

```bash
pip install pandas numpy requests beautifulsoup4 lxml urllib3 matplotlib seaborn
```

## Giải Thích Kỹ Thuật

### Tại sao cần các biện pháp tiền xử lý:

1. **Xử lý giá trị thiếu**: Đảm bảo mô hình không bị lỗi khi gặp dữ liệu rỗng
2. **Làm sạch văn bản**: Chuẩn hóa định dạng để tăng độ chính xác khi so sánh
3. **Web scraping**: Thu thập dữ liệu từ nhiều nguồn để mở rộng dataset
4. **API integration**: Lấy thông tin chi tiết từ database chuyên môn
5. **Feature engineering**: Tạo ra các đặc trưng có ý nghĩa từ raw data
6. **Data consolidation**: Kết hợp thông tin từ nhiều nguồn để có dataset phong phú

### Kỹ thuật đã học vs chưa học:

**Đã học trên lớp**:
- Data cleaning, missing value handling
- Text preprocessing
- Web scraping cơ bản
- Data merging và joining

**Chưa học trên lớp** (có trích nguồn):
- Advanced web scraping (BeautifulSoup, urllib)
- API integration (TMDB API)
- JSON parsing với ast.literal_eval
- Regular expressions cho pattern matching
- Complex string manipulation

## Lưu Ý

- Các file sẽ sử dụng dữ liệu mẫu nếu không tìm thấy dữ liệu thực
- Pipeline được thiết kế để chạy tuần tự, mỗi giai đoạn phụ thuộc vào output của giai đoạn trước
- Tất cả quá trình được log chi tiết để theo dõi
- Dữ liệu được validate ở mỗi bước để đảm bảo chất lượng 