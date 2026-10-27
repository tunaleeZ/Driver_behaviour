# Handoff tổng thể — Driver Behavior Monitoring

## Mục tiêu và trạng thái

Repo triển khai pipeline nhận diện hành vi tài xế với 5 lớp cấu hình tại `configs/labels.yaml`, các mô hình chuỗi Geo-LSTM/Landmark-LSTM, CNN-LSTM, baseline, công cụ xử lý dữ liệu và workflow Kaggle. Phase 1–7 theo `prompt.md` đã được triển khai. Hiện chưa có video hoặc metadata thật; chưa có kết quả huấn luyện/benchmark thực nghiệm.

## Thành phần

- `src/dbm/` và `driver_behavior_dataset_v01/scripts/`: cấu hình/đường dẫn, metadata, chia subject chống leakage, cắt video, trích frame/landmark, train/evaluate/inference/benchmark/sweep.
- `configs/`: labels, schema, đường dẫn, feature, model và sweep.
- `tests/`: kiểm tra unit/pipeline và synthetic smoke; dữ liệu tổng hợp không đại diện kết quả thực tế.
- `notebooks/kaggle_01_preprocess.ipynb`, `notebooks/kaggle_02_train.ipynb`: notebook mẫu Kaggle, cần thay slug/URL giữ chỗ và gắn dataset riêng.
- `docs/ASSUMPTIONS.md`, `docs/KAGGLE_GUIDE.md`, `docs/EXPERIMENTS.md`: giả định, triển khai Kaggle, bảng thực nghiệm còn trống.

## Cách bắt đầu

1. Cài Python >= 3.10, PyTorch phù hợp môi trường và `pip install -e . --no-deps`; cài dependency bổ sung theo `requirements.txt`.
2. Tạo cấu trúc dataset bằng `python driver_behavior_dataset_v01/scripts/init_dataset.py`; thêm video thật và annotation thật theo hướng dẫn README.
3. Validate metadata, cắt segment, chia split theo subject, trích frame/landmark và kiểm tra trực quan trước khi train.
4. Chạy các cấu hình E0 → E3; chọn checkpoint theo validation, đánh giá test một lần ở cuối.
5. Trên Kaggle, xem hai notebook và `docs/KAGGLE_GUIDE.md`. Giữ raw, output và dữ liệu nặng ngoài Git.

## Kiểm chứng

Lệnh kiểm tra tổng thể:

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q
python -m compileall -q src driver_behavior_dataset_v01/scripts
python -m dbm.run_sweep --dry-run
```

Các smoke run tổng hợp chỉ xác nhận đường chạy. Không chuyển số liệu smoke/benchmark thành kết quả báo cáo. Kaggle notebook chưa được chạy trên phiên Kaggle thực tế.

## Việc nhóm cần làm khi có dữ liệu

- Rà soát quyền sử dụng video và quyền riêng tư; điền metadata thật, không dùng dòng ví dụ trong template.
- Kiểm tra các subject ID, split, landmark index và chất lượng trích xuất; điều chỉnh các tham số chưa chốt theo `docs/ASSUMPTIONS.md`.
- Chạy thực nghiệm trên Kaggle, ghi lại cấu hình/commit/môi trường và chỉ cập nhật `docs/EXPERIMENTS.md` bằng kết quả đã chạy thật.
