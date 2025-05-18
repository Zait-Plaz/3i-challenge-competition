#  Hệ Thống Nhận Diện Món Ăn và Định dạng giá tiền 

## Tổng quan dự án

Dự án này xây dựng một hệ thống trí tuệ nhân tạo có khả năng **tự động nhận diện các món ăn trên khay** tại quầy thanh toán thông qua **phân tích ảnh đầu vào** từ người dùng tải lên. Hệ thống sử dụng **YOLOv8 để phát hiện vị trí món ăn** và **mô hình CNN (.keras) để phân loại từng món**, từ đó tra cứu bảng giá và tính tổng chi phí thanh toán.

 **Mục tiêu**:
- Giảm thời gian chờ tại quầy thu ngân.
- Nâng cao độ chính xác trong tính tiền.
- Hỗ trợ giao diện web hiện đại, dễ triển khai nội bộ.
- Là nền tảng mở rộng cho các hệ thống POS thông minh.

---

## Hướng dẫn cài đặt

### 1. Môi trường lập trình
- **Ngôn ngữ**: Python
- **Mô hình phát hiện**: YOLOv8 (Ultralytics)
- **Mô hình phân loại**: TensorFlow (.keras)
- **Thư viện hỗ trợ**:
  - `gradio` (Giao diện web)
  - `opencv-python` (Xử lý ảnh)
  - `numpy` (Xử lý dữ liệu)
  - `ultralytics` (YOLOv8)
  - `tensorflow` (Keras model)

### 2. Cài đặt thư viện
```bash
pip install gradio ultralytics tensorflow opencv-python numpy
```

### 3. Chuẩn bị file mô hình
- `yolov8n.pt` – mô hình YOLOv8 nhận diện khay thức ăn.
- `cnn.keras` – mô hình phân loại món ăn.

---

## Hướng dẫn sử dụng

1. **Chạy ứng dụng**:
```bash
python Main.py
```

2. **Tải ảnh khay cơm** qua giao diện Gradio.

3. **Hệ thống sẽ**:
   - Phát hiện bounding box các món ăn bằng YOLOv8.
   - Crop từng vùng ảnh và phân loại món ăn bằng mô hình CNN.
   - Tra cứu bảng giá và tính tổng chi phí.

4. **Hiển thị kết quả**:
   - Danh sách món ăn kèm giá từng món.
   - Tổng tiền cần thanh toán.

---

## Thư viện

| Thành phần | Vai trò |
|------------|---------|
| Python ≥ 3.8 | Ngôn ngữ chính |
| YOLOv8 (Ultralytics) | Nhận diện vật thể |
| TensorFlow / Keras | Phân loại món ăn |
| Gradio | Giao diện web |
| OpenCV | Xử lý ảnh |
| NumPy | Xử lý mảng số |
| Camera (tuỳ chọn) | Nếu muốn tích hợp thời gian thực |

---

© 2025 – Đồ án môn Trí tuệ Nhân tạo – UEH
