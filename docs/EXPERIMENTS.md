# EXPERIMENTS.md — Nhật Ký Thực Nghiệm

> **Lưu ý:** File này chưa có kết quả thực nghiệm.
> Mọi bảng kết quả sẽ được điền sau khi chạy thật trên Kaggle.
> TUYỆT ĐỐI KHÔNG điền số liệu bịa hoặc số liệu từ paper khác mà chưa
> xác nhận trên dataset của nhóm.

---

## Thiết Lập Thực Nghiệm

| Tham số | Giá trị |
|---------|---------|
| Split | 70% train / 20% val / 10% test (subject-independent) |
| Metric chính | Macro-F1 trên validation |
| Đánh giá test | 1 lần duy nhất với best checkpoint |
| Seed | 42 (và 123 cho multi-seed) |

---

## E0: Baseline Luật Ngưỡng (threshold_rule.py)

| Ngưỡng | Metric | Kết quả |
|--------|--------|---------|
| EAR < ? | PERCLOS | chưa có kết quả |
| MAR > ? | - | chưa có kết quả |
| Pitch > ? | - | chưa có kết quả |

**Ghi chú:** Baseline luật ngưỡng chỉ phân biệt được normal/sleepy/drowsy,
không có thông tin về tay/vật thể nên không tách được drinking và phone.

---

## E1: Geo-LSTM (Variant A, D=4)

| T | Pooling | Loss | Val Macro-F1 | Test Macro-F1 |
|---|---------|------|-------------|--------------|
| 30 | mean | CE | chưa có kết quả | - |
| 30 | mean | weighted CE | chưa có kết quả | - |
| ... | ... | ... | ... | ... |

---

## E2: Landmark-LSTM (Variant B, D=2K+4)

| T | K | Pooling | Loss | Val Macro-F1 | Test Macro-F1 |
|---|---|---------|------|-------------|--------------|
| 30 | ? | mean | weighted CE | chưa có kết quả | - |
| 60 | ? | mean | weighted CE | chưa có kết quả | - |
| ... | ... | ... | ... | ... | ... |

---

## E3: CNN-LSTM

| T | Pooling | Loss | Val Macro-F1 | Test Macro-F1 | FPS (GPU) |
|---|---------|------|-------------|--------------|-----------|
| 16 | mean | weighted CE | chưa có kết quả | - | - |
| 32 | mean | weighted CE | chưa có kết quả | - | - |
| ... | ... | ... | ... | ... | ... |

---

## Confusion Matrix

*(Chưa có — sẽ được điền sau khi chạy thật)*

---

## Benchmark (FPS và Độ Trễ)

| Mô hình | Thiết bị | FPS | Latency (ms) | p95 Latency |
|---------|----------|-----|-------------|-------------|
| Geo-LSTM | CPU | - | - | - |
| Landmark-LSTM | CPU | - | - | - |
| CNN-LSTM | GPU | - | - | - |

---

## Nhận Xét và Kết Luận

*(Chưa có — sẽ được điền sau khi có kết quả thực nghiệm)*

