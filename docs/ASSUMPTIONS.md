# ASSUMPTIONS.md — Các Giả Định và Quyết Định Thiết Kế

> **Lưu ý:** File này ghi lại tất cả các tham số chưa được xác nhận bằng thực nghiệm,
> các quyết định thiết kế mặc định, và các điểm nhóm cần tự biện luận trước khi
> sử dụng kết quả trong báo cáo chính thức.

---

## 1. Landmark Indices (MediaPipe Face Mesh 478-point)

**Trạng thái: CHƯA XÁC NHẬN** — Cần kiểm tra bằng `visualize_landmarks.py`.

### EAR (Eye Aspect Ratio)

| Mắt | p1 | p2 | p3 | p4 | p5 | p6 |
|-----|----|----|----|----|----|----|
| Phải | 33 | 160 | 158 | 133 | 153 | 144 |
| Trái | 362 | 385 | 387 | 263 | 373 | 380 |

### MAR (Mouth Aspect Ratio)

| Điểm | p1 | p2 | p3 | p4 | p5 | p6 |
|------|----|----|----|----|----|-----|
| Môi trong | 78 | 82 | 312 | 308 | 317 | 87 |

### Head Pose (solvePnP)

6 điểm: đầu mũi (1), cằm (152), khóe mắt ngoài trái (263),
khóe mắt ngoài phải (33), khóe miệng trái (291), khóe miệng phải (61).

**Mô hình 3D khuôn mặt tổng quát (mm):** giá trị thông dụng trong tài liệu OpenCV/MediaPipe.
Nhóm cần kiểm tra xem mô hình này có phù hợp với dữ liệu thu thập hay không.

### Landmark được chọn cho Landmark-LSTM (K)

Danh sách mặc định gồm viền mắt trái/phải (12 điểm), viền môi ngoài + trong (18 điểm),
đầu mũi (1), cằm (1), trán (1), hai điểm viền mặt 234 và 454 (2 điểm) — tổng K ≈ 35.
**Đây là giá trị thử nghiệm, chưa được tối ưu.**

---

## 2. Tham Số Mô Hình (Chưa Chốt)

| Tham số | Giá trị mặc định | Ghi chú |
|---------|-----------------|---------|
| T (Landmark-LSTM) | 30 | Thử thêm 60, 90 |
| T (CNN-LSTM) | 16 | Thử thêm 32 |
| Stride | T/2 | Có thể điều chỉnh |
| LSTM hidden size (Landmark) | 64 | "64 units — dự kiến" trong báo cáo |
| LSTM hidden size (CNN-LSTM) | 128 | Chưa xác nhận |
| LSTM num_layers | 1 | Thử thêm 2 |
| Dropout | 0.3 (LSTM), 0.4 (CNN head) | Chưa xác nhận |
| Pooling mặc định | mean | Thử last, attention |

---

## 3. Ma Trận Chi Phí (Cost-Sensitive Loss)

**PLACEHOLDER — Nhóm phải tự biện luận trước khi dùng trong báo cáo.**

```
              normal  drowsy  sleepy  drinking  phone
normal          0       1       1       1        1
drowsy          4       0       1       2        2
sleepy          3       1       0       2        2
drinking        2       2       2       0        1
phone           3       2       2       1        0
```

Lý do đặt giá trị cao cho drowsy → normal (4): bỏ sót tài xế ngủ gật nguy hiểm nhất.
Nhóm cần xem lại và biện luận dựa trên phân tích rủi ro thực tế.

---

## 4. Chuẩn Hóa Landmark (Landmark-LSTM)

- **Gốc tọa độ:** điểm mũi (index 1) — giả định ổn định theo đầu.
- **Scale:** khoảng cách hai mắt (inter-ocular distance).
- **Xoay bù roll:** tắt theo mặc định (`normalize_roll: false`).

Nhóm có thể thử các cấu hình khác nhau bằng cách thay đổi `features.yaml`.

---

## 5. Chính Sách Segment Ngắn

Mặc định: `pad` (lặp frame cuối, đánh dấu padded).
Lý do: tránh mất dữ liệu khi segment chỉ ngắn hơn T vài frame.
Nhóm nên thống kê số segment bị pad và xem xét tác động lên kết quả.

---

## 6. Feature FPS

- Landmark: 30 FPS (mặc định, giữ nguyên fps video nếu video là 30fps).
- Frame cho CNN-LSTM: 10 FPS.

---

## 7. Subject ID cho Video Public

Video public không biết danh tính → dùng `PUB_<mã video>` làm subject_id.
**Giới hạn:** Không đảm bảo subject-independent hoàn toàn nếu cùng một người
xuất hiện ở nhiều video khác nhau.

---

## 8. Các Điểm Cần Nhóm Quyết Định

- [ ] Xác nhận index landmark EAR/MAR/head pose bằng `visualize_landmarks.py`.
- [ ] Xác nhận danh sách K landmark cho Landmark-LSTM.
- [ ] Biện luận và chỉnh ma trận chi phí.
- [ ] Quyết định enum `camera_view/lighting/glasses/occlusion` dựa trên dữ liệu thực tế.
- [ ] Chọn T tối ưu sau khi có kết quả thực nghiệm.
- [ ] Chọn pooling tối ưu sau khi có kết quả thực nghiệm.

