# Handoff tiếp theo: Phase 7

## Trạng thái

- Phase 1–3: hoàn tất theo các handoff trước.
- Phase 4: đã có dữ liệu tổng hợp `.npz`, frame và video, có cờ `synthetic=true`; Geo-LSTM/Landmark-LSTM train/evaluate smoke, overfit test và synthetic summary guard đều được kiểm tra.
- Phase 5: CNN-LSTM được train/evaluate qua `FrameClipDataset`; smoke train CPU và kiểm tra augmentation nhất quán trong clip pass.
- Phase 6: `run_sweep --dry-run` liệt kê grid; `--quick` train tổng hợp E1–E3 với `last`, `mean`, `attention`. Inference end-to-end cho Geo, Landmark và CNN dùng video tổng hợp/extractor mock. Benchmark hỗ trợ tensor và đo extract/compose feature; thiếu model/video thì cảnh báo và bỏ qua.
- Toàn bộ pytest: 137 passed. `compileall`, `git diff --check` và sweep dry-run pass. Có cảnh báo deprecation PyTorch ở `scheduler.step(epoch)` trong smoke train.
- Không có kết quả benchmark hoặc kết quả thực nghiệm tổng hợp được ghi vào tài liệu. Các giá trị từ smoke chỉ xác nhận đường chạy.

## Tệp và thành phần chính đã hoàn tất

- `src/dbm/data/synthetic.py`: sinh feature, frame, video, metadata/split tổng hợp.
- `src/dbm/train.py`, `src/dbm/evaluate.py`, `src/dbm/data/datasets.py`: train/evaluate Geo, Landmark và CNN-LSTM; normalization chỉ fit trên train.
- `src/dbm/run_sweep.py`, `configs/sweeps/*.yaml`: grid dry-run và quick smoke train trên dữ liệu tổng hợp.
- `src/dbm/infer_video.py`: sliding-window timeline, EMA/majority vote cho cả ba model type.
- `src/dbm/benchmark.py`: benchmark forward model và API benchmark extractor mock/thật.
- `tests/test_synthetic_data.py`, `tests/test_phase6_tools.py`: smoke, overfit, guard synthetic, inference và benchmark.

## Lệnh xác nhận

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q
python -m compileall -q src driver_behavior_dataset_v01/scripts
python -m dbm.run_sweep --dry-run
python -m dbm.run_sweep --dry-run --quick
python -m dbm.run_sweep --quick
```

## Tiếp theo: Phase 7

Chỉ thực hiện các hạng mục Kaggle và hoàn thiện được mô tả trong `prompt.md`: tạo hai notebook bằng nbformat, clear output, kiểm tra notebook/CLI, cập nhật README, `KAGGLE_GUIDE.md` và `ASSUMPTIONS.md`, rồi rà soát artifact/secrets. Chưa có video hoặc metadata thật nên không ghi nhận kết quả thực nghiệm.
