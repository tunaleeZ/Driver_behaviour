# Handoff sau Phase 5

## Trạng thái

- Phase 1–4 đã hoàn thành theo handoff trước.
- Phase 5 đã bổ sung CNN backbone bốn khối, CNN-LSTM dùng chung CNN cho các frame, ba kiểu temporal pooling, bộ dựng model và FrameClipDataset.
- FrameClipDataset đọc JPG theo từng segment, loại quality=bad, tạo clip theo window/stride, hỗ trợ pad/drop và resize về input size. Khi train, crop, xoay nhẹ, brightness và contrast được lấy một lần rồi áp dụng nhất quán cho tất cả frame trong clip. Khi đánh giá, dataset dùng center crop; không lật ngang.
- Kiểm thử shape đã xác nhận input `(8, 16, 3, 112, 112)` cho output `(8, 5)`; convolution đầu có 896 tham số. Có test pooling, attention weights, padding và optimizer step CPU với tensor tổng hợp.
- Toàn bộ pytest: 124 passed. `compileall` và `git diff --check` hoàn tất.

## Tệp chính Phase 5

- `src/dbm/models/cnn_backbone.py`: CNN 4 block Conv-BN-ReLU-MaxPool và global average pooling.
- `src/dbm/models/cnn_lstm.py`: encoder dùng chung theo frame, LSTM, pooling và classifier.
- `src/dbm/models/build.py`: thêm `build_cnn_lstm`.
- `src/dbm/data/datasets.py`: thêm `FrameClipDataset`.
- `tests/test_phase5_cnn.py`: shape, số tham số, pooling, padding/augmentation và CPU optimizer step.

## Lệnh xác nhận

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q
python -m compileall -q src driver_behavior_dataset_v01/scripts
git diff --check
```

## Giới hạn còn lại

- CNN-LSTM và FrameClipDataset hiện chưa được tích hợp vào CLI `train.py`/`evaluate.py`; bước optimizer CPU trong test là smoke ở mức model với tensor tổng hợp, chưa phải train end-to-end qua pipeline frame.
- Chưa có frame/video thật; không có kết quả thực nghiệm. Không dùng dữ liệu tổng hợp để báo cáo kết quả.
- Phase 6 cần triển khai `run_sweep.py`, `benchmark.py`, `infer_video.py`, kiểm tra dry-run lưới E0–E3, quick smoke, benchmark model bằng tensor và inference với extractor mock/video tổng hợp. Cần đảm bảo `synthetic=true` không ghi vào `outputs/results_summary.csv`.
- Chưa có dữ liệu người thật để xác minh trực quan landmark; giữ lưu ý từ handoff Phase 4.

## Hướng triển khai Phase 6

1. Đọc `prompt.md` mục 9.4, 9.5 và Phase 6 mục 12; kiểm kê API hiện có trong `train.py`, `evaluate.py`, `metrics.py`, `models/build.py` và cấu hình `configs/sweeps/`.
2. Xây `run_sweep.py` theo cấu hình, hỗ trợ `--dry-run` in danh sách run E0–E3 và `--quick` chạy kiểm tra tổng hợp ngắn. Phát hiện sớm phần CNN-LSTM chưa có train/evaluate CLI và tích hợp cần thiết.
3. Xây `benchmark.py`: đo model trên tensor với warmup, CPU/GPU sẵn có; phần landmark/video thật phải bỏ qua kèm cảnh báo nếu thiếu model/video. Không ghi số đo vào docs.
4. Xây `infer_video.py` hỗ trợ hai họ model, cửa sổ trượt, làm mượt vote/EMA, CSV timeline và tùy chọn video chú thích; kiểm tra bằng video tổng hợp và extractor mock.
5. Thêm test tập trung cho danh sách sweep, chế độ dry-run, guard synthetic, benchmark tensor và inference mock. Không tải dữ liệu/video từ Internet.
6. Cập nhật docs/EXPERIMENTS.md chỉ với bảng chưa có kết quả; chạy pytest, compileall và `git diff --check` trước khi kết thúc phase.
