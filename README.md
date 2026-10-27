# Hệ Thống Nhận Diện Hành Vi Tài Xế Lái Xe Ô Tô

**Nhóm 10 — ĐH Bách Khoa Hà Nội**

> **Lưu ý quan trọng:** Repo này chứa code, config và tài liệu.
> **Video và dữ liệu nặng KHÔNG được commit lên GitHub** vì video chứa khuôn mặt
> người (quyền riêng tư) và kích thước quá lớn.
> Video được upload riêng thành Kaggle Dataset (private).

---

## Giới Thiệu

Hệ thống nhận diện 5 lớp hành vi của tài xế lái xe ô tô:

| Index | Nhãn | Mô tả |
|-------|------|--------|
| 0 | normal | Tư thế ổn định, không có hành vi bất thường |
| 1 | drowsy | Mất tỉnh táo rõ (đầu gục, mắt nhắm lâu) |
| 2 | sleepy | Buồn ngủ nhưng chưa gục (mắt khép, chớp chậm) |
| 3 | drinking | Uống nước/đồ uống khi lái xe |
| 4 | phone | Sử dụng điện thoại khi lái xe |

**Pipeline:** Video 30 FPS → MediaPipe Face Landmarker (478 landmark 3D) →
Trích đặc trưng (EAR, MAR, Yaw/Pitch/Roll, landmark) → LSTM / CNN-LSTM → Phân loại 5 lớp.

### Kiến Trúc Mô Hình (Đề Xuất, Chưa Có Kết Quả Thực Nghiệm)

- **Geo-LSTM (Variant A):** LSTM với 4 đặc trưng hình học [EAR, MAR, ΔYaw, ΔPitch].
- **Landmark-LSTM (Variant B):** LSTM với K landmark 2D ghép 4 đặc trưng hình học.
- **CNN-LSTM:** CNN 4 khối (tự train) + LSTM nhận chuỗi frame RGB.
- **Baseline luật ngưỡng:** PERCLOS, pitch, ngưỡng EAR/MAR cho normal/sleepy/drowsy.

> **Lưu ý:** Các kiến trúc trên là đề xuất chưa có kết quả huấn luyện.
> Mọi bảng metric trong `docs/EXPERIMENTS.md` để trống cho đến khi chạy thật.

---

## Cài Đặt

### Yêu Cầu

- Python ≥ 3.10
- PyTorch (đã cài sẵn trên Kaggle; tự cài nếu chạy local)
- Các gói bổ sung: xem `requirements.txt`

### Cài Đặt Local

```bash
# Clone repo
git clone https://github.com/<your-org>/driver-behavior-monitoring.git
cd driver-behavior-monitoring

# Cài package
pip install -e . --no-deps

# Cài thêm dependency
pip install -r requirements.txt
```

### Trên Kaggle

Xem `docs/KAGGLE_GUIDE.md` để biết hướng dẫn chi tiết.

---

## Quy Trình End-to-End

### Bước 1: Khởi tạo cấu trúc thư mục

```bash
python driver_behavior_dataset_v01/scripts/init_dataset.py
```

### Bước 2: Tạo template metadata

```bash
python driver_behavior_dataset_v01/scripts/make_metadata_template.py
```

Điền thông tin vào `driver_behavior_dataset_v01/annotations/metadata.xlsx`.

### Bước 3: Kiểm kê video

```bash
python driver_behavior_dataset_v01/scripts/inventory.py
```

### Bước 4: Validate metadata

```bash
python driver_behavior_dataset_v01/scripts/validate_metadata.py
# Thêm --assign-ids để tự động gán segment_id mới
python driver_behavior_dataset_v01/scripts/validate_metadata.py --assign-ids
```

### Bước 5: Cắt video thành segment

```bash
python driver_behavior_dataset_v01/scripts/cut_segments.py
# Xem lệnh sẽ chạy mà không thực thi
python driver_behavior_dataset_v01/scripts/cut_segments.py --dry-run
```

### Bước 6: Chia tập train/val/test

```bash
python driver_behavior_dataset_v01/scripts/make_splits.py
```

### Bước 7: Trích frame (cho CNN-LSTM)

```bash
python driver_behavior_dataset_v01/scripts/extract_frames.py
```

### Bước 8: Trích landmark (cho Landmark-LSTM)

```bash
# Cần file models/face_landmarker.task
# Script tự tải nếu có Internet; hoặc đặt sẵn vào models/
python driver_behavior_dataset_v01/scripts/extract_landmarks.py
```

### Bước 9: Huấn luyện

```bash
# Geo-LSTM (Variant A)
python -m dbm.train --config configs/models/geo_lstm.yaml

# Landmark-LSTM (Variant B)
python -m dbm.train --config configs/models/landmark_lstm.yaml

# CNN-LSTM
python -m dbm.train --config configs/models/cnn_lstm.yaml

# Với override tham số
python -m dbm.train --config configs/models/geo_lstm.yaml --override lr=0.001 batch_size=32
```

### Bước 10: Đánh giá

```bash
python -m dbm.evaluate --checkpoint outputs/<run_name>/best.pt --split test
```

### Bước 11: Suy luận trên video dài

```bash
python -m dbm.infer_video --checkpoint outputs/<run_name>/best.pt --video path/to/video.mp4
```

### Sweep thí nghiệm

```bash
python -m dbm.run_sweep --sweep configs/sweeps/e1_geo_lstm.yaml
# Chạy nhanh để kiểm tra pipeline
python -m dbm.run_sweep --sweep configs/sweeps/e1_geo_lstm.yaml --quick
```

---

## Cấu Trúc Repo

```
driver-behavior-monitoring/
  README.md
  pyproject.toml          package "dbm"
  requirements.txt
  .gitignore, .gitattributes
  configs/
    labels.yaml           nguồn duy nhất định nghĩa 5 lớp
    paths.yaml, schema.yaml, mediapipe.yaml, features.yaml
    cost_matrix.yaml
    models/               geo_lstm.yaml, landmark_lstm.yaml, cnn_lstm.yaml
    sweeps/               e1_geo_lstm.yaml, e2_landmark_lstm.yaml, e3_cnn_lstm.yaml
  driver_behavior_dataset_v01/
    annotations/          metadata.xlsx, metadata.csv, splits.csv (trong git)
    scripts/              các script xử lý dữ liệu (trong git)
    raw/                  video gốc (KHÔNG trong git)
    segments/             video đã cắt (KHÔNG trong git)
    processed/            frames đã trích (KHÔNG trong git)
    features/             landmark .npz (KHÔNG trong git)
  src/dbm/                package Python
    paths.py, config.py, labels.py, seed.py, logging_utils.py, timecode.py
    data/                 metadata.py, splits.py, windows.py, datasets.py, ...
    features/             mediapipe_extractor.py, geometry.py, head_pose.py, ...
    models/               lstm_classifier.py, cnn_backbone.py, cnn_lstm.py, ...
    baselines/            threshold_rule.py
    losses.py, metrics.py
    train.py, evaluate.py, infer_video.py, benchmark.py, run_sweep.py
  notebooks/              kaggle_01_preprocess.ipynb, kaggle_02_train.ipynb
  tests/                  pytest test suite
  docs/                   ASSUMPTIONS.md, KAGGLE_GUIDE.md, EXPERIMENTS.md
  models/                 face_landmarker.task (KHÔNG trong git)
  outputs/                kết quả training (KHÔNG trong git)
```

---

## Chạy Tests

```bash
pytest tests/ -v
```

Tests chạy được trên CPU, không cần dữ liệu thật, không cần Internet, dưới 2 phút.

---

## Lưu Ý Về Quyền Riêng Tư

- Video chứa khuôn mặt người **KHÔNG được commit lên GitHub**.
- Video được lưu trên Kaggle Dataset private.
- Với video nguồn public không biết danh tính người quay, dùng `subject_id = PUB_<mã>`.
- Không đảm bảo subject-independent hoàn toàn nếu cùng một người xuất hiện ở nhiều video public.

---

## Giới Hạn Đã Biết

- **Baseline luật ngưỡng** chỉ phân biệt được normal/sleepy/drowsy từ face landmark;
  không phân biệt được drinking và phone vì cần thông tin về tay/vật thể.
- **Dữ liệu public** có thể không đảm bảo tính độc lập giữa các tập train/val/test
  nếu cùng một người xuất hiện ở nhiều video.
- **Chưa có kết quả thực nghiệm** — xem `docs/EXPERIMENTS.md`.
- **Mọi tham số** (K, T, hidden size, lambda, ngưỡng...) là giá trị thử nghiệm,
  chưa được tối ưu hóa. Xem `docs/ASSUMPTIONS.md`.

---

## Tài Liệu Bổ Sung

- [docs/ASSUMPTIONS.md](docs/ASSUMPTIONS.md) — Các giả định và quyết định thiết kế.
- [docs/KAGGLE_GUIDE.md](docs/KAGGLE_GUIDE.md) — Hướng dẫn chạy trên Kaggle.
- [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) — Nhật ký thực nghiệm.

