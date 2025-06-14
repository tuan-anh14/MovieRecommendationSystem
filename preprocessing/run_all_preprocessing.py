"""
Script chính để chạy tất cả các giai đoạn tiền xử lý dữ liệu phim
Chạy tuần tự từ giai đoạn 1 đến 4

Sử dụng: python run_all_preprocessing.py
"""

import logging
import sys
import os
import time
from datetime import datetime

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocessing_log.txt'),
        logging.StreamHandler(sys.stdout)
    ]
)

def run_preprocessing_stage(stage_number, script_name):
    """Chạy một giai đoạn tiền xử lý"""
    logging.info(f"="*50)
    logging.info(f"BẮT ĐẦU GIAI ĐOẠN {stage_number}: {script_name}")
    logging.info(f"="*50)
    
    start_time = time.time()
    
    try:
        # Import và chạy module
        if stage_number == 1:
            import preprocessing_1
            preprocessing_1.main()
        elif stage_number == 2:
            import preprocessing_2
            preprocessing_2.main()
        elif stage_number == 3:
            import preprocessing_3
            preprocessing_3.main()
        elif stage_number == 4:
            import preprocessing_4
            preprocessing_4.main()
        
        end_time = time.time()
        duration = end_time - start_time
        
        logging.info(f"HOÀN THÀNH GIAI ĐOẠN {stage_number} - Thời gian: {duration:.2f} giây")
        return True
        
    except Exception as e:
        logging.error(f"LỖI TRONG GIAI ĐOẠN {stage_number}: {str(e)}")
        return False

def check_dependencies():
    """Kiểm tra các thư viện cần thiết"""
    logging.info("Kiểm tra dependencies...")
    
    required_packages = [
        'pandas', 'numpy', 'requests', 'bs4', 'matplotlib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            logging.info(f"✓ {package} - OK")
        except ImportError:
            missing_packages.append(package)
            logging.error(f"✗ {package} - THIẾU")
    
    if missing_packages:
        logging.error(f"Cần cài đặt các package: {', '.join(missing_packages)}")
        logging.error("Chạy lệnh: pip install " + " ".join(missing_packages))
        return False
    
    logging.info("Tất cả dependencies đã sẵn sàng!")
    return True

def create_directories():
    """Tạo các thư mục cần thiết"""
    logging.info("Tạo thư mục datasets nếu chưa có...")
    
    datasets_dir = '../datasets'
    if not os.path.exists(datasets_dir):
        os.makedirs(datasets_dir)
        logging.info(f"Đã tạo thư mục: {datasets_dir}")
    else:
        logging.info(f"Thư mục đã tồn tại: {datasets_dir}")

def main():
    """Hàm chính chạy tất cả các giai đoạn tiền xử lý"""
    start_time = datetime.now()
    logging.info(f"BẮT ĐẦU QUÁ TRÌNH TIỀN XỬ LÝ DỮ LIỆU PHIM - {start_time}")
    logging.info("="*70)
    
    # Kiểm tra dependencies
    if not check_dependencies():
        logging.error("Không thể tiếp tục do thiếu dependencies")
        sys.exit(1)
    
    # Tạo thư mục cần thiết
    create_directories()
    
    # Danh sách các giai đoạn
    stages = [
        (1, "preprocessing_1.py - Xử lý dữ liệu cơ bản"),
        (2, "preprocessing_2.py - Xử lý dữ liệu TMDB 2017"),
        (3, "preprocessing_3.py - Xử lý dữ liệu Wikipedia 2018-2019"),
        (4, "preprocessing_4.py - Xử lý dữ liệu 2020 và hoàn thiện")
    ]
    
    successful_stages = 0
    failed_stages = []
    
    # Chạy từng giai đoạn
    for stage_num, description in stages:
        success = run_preprocessing_stage(stage_num, description)
        
        if success:
            successful_stages += 1
            logging.info(f"✓ Giai đoạn {stage_num} thành công")
        else:
            failed_stages.append(stage_num)
            logging.error(f"✗ Giai đoạn {stage_num} thất bại")
            
            # Hỏi có muốn tiếp tục không
            user_input = input(f"Giai đoạn {stage_num} thất bại. Tiếp tục? (y/n): ")
            if user_input.lower() != 'y':
                logging.info("Người dùng chọn dừng quá trình")
                break
    
    # Tổng kết
    end_time = datetime.now()
    total_duration = end_time - start_time
    
    logging.info("="*70)
    logging.info("TỔNG KẾT QUÁ TRÌNH TIỀN XỬ LÝ")
    logging.info("="*70)
    logging.info(f"Thời gian bắt đầu: {start_time}")
    logging.info(f"Thời gian kết thúc: {end_time}")
    logging.info(f"Tổng thời gian: {total_duration}")
    logging.info(f"Số giai đoạn thành công: {successful_stages}/{len(stages)}")
    
    if failed_stages:
        logging.error(f"Các giai đoạn thất bại: {failed_stages}")
    
    if successful_stages == len(stages):
        logging.info("🎉 TẤT CẢ CÁC GIAI ĐOẠN HOÀN THÀNH THÀNH CÔNG!")
        logging.info("Dataset cuối cùng: ../datasets/movies_final_preprocessed.csv")
    else:
        logging.warning("⚠️  Một số giai đoạn chưa hoàn thành")
    
    logging.info("="*70)

if __name__ == '__main__':
    main() 