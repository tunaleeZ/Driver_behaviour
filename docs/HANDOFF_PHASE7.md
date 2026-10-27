# Handoff tiếp theo: Phase 7

## Trạng thái

- Phase 1–5 theo handoff trước.
- Phase 6 đã thêm CLI mở rộng lưới sweep, benchmark forward model bằng tensor tổng hợp và API inference trượt cho Geo/Landmark-LSTM.
- Sweep `--dry-run --quick` liệt kê E0–E3; dry-run đầy đủ liệt kê các cấu hình E1–E3. E0 là entry mô tả baseline luật ngưỡng.
- Benchmark không ghi file kết quả; thời gian in ra chỉ là chẩn đoán trên máy đang chạy, không dùng làm kết quả báo cáo.
- Toàn bộ test tại thời điểm bàn giao: 127 passed. `compileall` và CLI help đã chạy.

## Tệp Phase 6

- `src/dbm/run_sweep.py`: mở rộng config sweep và dry-run.
- `src/dbm/benchmark.py`: benchmark forward trên tensor tổng hợp, warmup, chọn CPU/GPU; cảnh báo và bỏ qua landmark benchmark nếu thiếu tài nguyên thật.
- `src/dbm/infer_video.py`: inference trượt, smoothing EMA/majority vote, xuất CSV; có API nhận extractor để mock.
- `tests/test_phase6_tools.py`: kiểm tra cấu hình quick, tensor benchmark và smoothing.
- `docs/HANDOFF_PHASE7.md`: ghi trạng thái và các giới hạn cần xử lý trước khi kết thúc phase 7.

## Lệnh xác nhận

```powershell
$env:PYTHONPATH = "src"
python -m dbm.run_sweep --dry-run --quick
python -m dbm.run_sweep --dry-run
python -m dbm.benchmark --config configs/models/geo_lstm.yaml
python -m pytest -q
python -m compileall -q src driver_behavior_dataset_v01/scripts
```

## Việc còn thiếu cần xử lý trong phase 7

1. Handoff phase 5 dự kiến `src/dbm/data/synthetic.py`, nhưng file không tồn tại. Bổ sung generator được gắn cờ `synthetic=true`, chỉ dùng cho test/smoke.
2. `run_sweep --quick` hiện mới mở rộng danh sách, chưa chạy train tổng hợp; lệnh thực thi hiện báo rõ rằng trainer dữ liệu tổng hợp chưa có.
3. `train.py`/`evaluate.py` hiện chỉ hỗ trợ Geo/Landmark-LSTM; CNN-LSTM chưa được tích hợp end-to-end. Sweep E3 chỉ được dry-run, không chạy train.
4. `infer_video` hiện dành cho checkpoint Geo/Landmark-LSTM. Chưa có kiểm thử end-to-end bằng video tổng hợp và extractor mock; CNN-LSTM chưa hỗ trợ inference.
5. Benchmark landmark/video thật chưa được cài đặt; hiện tự bỏ qua và cảnh báo. Không có số đo nào được thêm vào docs.
6. Thực hiện các hạng mục Phase 7 trong `prompt.md`: hai notebook Kaggle dùng nbformat và clear outputs, kiểm thử notebook/CLI, cập nhật README/KAGGLE_GUIDE/ASSUMPTIONS, rà soát artifact và secrets.

Các giới hạn trên được giữ rõ ràng để không báo cáo quá mức trạng thái triển khai. Không chạy huấn luyện thật hoặc thêm số liệu thực nghiệm.
