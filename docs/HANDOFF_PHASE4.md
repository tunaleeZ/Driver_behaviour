# Handoff sau Phase 4

## Trạng thái

- Phase 2: rà soát logic và test tổng hợp đạt. Pipeline metadata → cắt segment → chia split đã có test end-to-end.
- Phase 3: rà soát logic và test hình học/trích xuất đạt. Chưa chạy trên video người thật theo yêu cầu điều chỉnh của người dùng vì hiện chưa có video mẫu.
- Phase 4: đã triển khai các thành phần chuỗi đặc trưng, Geo-LSTM/Landmark-LSTM, baseline ngưỡng, losses, metrics, train và evaluate. Smoke train 1 epoch trên NPZ tổng hợp tạo đủ checkpoint, stats, log, kết quả và confusion matrix.
- Toàn bộ pytest: 120 passed. `compileall` và `git diff --check` cũng hoàn tất không có lỗi.

## Tệp chính được bổ sung/sửa

- `src/dbm/data/windows.py`: window theo từng segment, hỗ trợ pad/drop.
- `src/dbm/data/datasets.py`: ghép feature group và tạo GeoSequenceDataset, loại cửa sổ thiếu tỷ lệ frame hợp lệ.
- `src/dbm/data/stats.py`: fit/áp dụng z-score; train gọi fit trên train split.
- `src/dbm/models/lstm_classifier.py`, `pooling.py`, `build.py`: LSTM một chiều với last/mean/attention.
- `src/dbm/baselines/threshold_rule.py`: đặc trưng và luật baseline chỉ dùng mặt; không nhận diện phone/drinking.
- `src/dbm/losses.py`, `src/dbm/metrics.py`: CE, weighted CE, focal, cost-sensitive và metric theo window/segment.
- `src/dbm/train.py`, `src/dbm/evaluate.py`: train với validation-based checkpoint, test cuối cùng, lưu outputs và đánh giá checkpoint.
- `src/dbm/data/splits.py`: split subject-level greedy có xét thời lượng và phân bố lớp; lọc quality=bad.
- `tests/test_phase4_sequence.py`: kiểm thử CPU cho window, stats, pooling, model, loss và metrics.

## Chạy

```powershell
python -m pytest -q
python -m compileall -q src driver_behavior_dataset_v01/scripts
python -m dbm.train --config configs/models/geo_lstm.yaml
python -m dbm.train --config configs/models/landmark_lstm.yaml
python -m dbm.evaluate --checkpoint outputs/<run_name>/best.pt --split test
```

Để dùng lệnh `python -m dbm...`, cài package theo README bằng `pip install -e . --no-deps` hoặc thêm `src` vào `PYTHONPATH`.

## Giới hạn và việc cần làm tiếp

- Chưa có dữ liệu video/feature thật nên chưa có kết quả huấn luyện thực nghiệm. Kết quả smoke tổng hợp chỉ xác nhận pipeline, không dùng trong báo cáo.
- Với ít subject, split theo subject có thể lệch tỷ lệ 70/20/10 hoặc thiếu lớp ở val/test; xem cảnh báo trong `make_splits.py`. Cần nhiều subject hơn để tỷ lệ và phân bố lớp có ý nghĩa.
- Cần xác minh hình học/landmark bằng mắt khi có video mẫu và rà lại giả định normalize-landmark so với danh sách landmark thực tế trước khi huấn luyện báo cáo.
- Phase 5 (CNN-LSTM), phase 6 (sweep/benchmark/inference) và các notebook/hoàn thiện Kaggle thuộc phase 7 chưa được thực hiện.
