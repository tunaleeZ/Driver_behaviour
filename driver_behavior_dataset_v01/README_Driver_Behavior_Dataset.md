# Driver Behavior Dataset v01

**Nhóm 10 — ĐH Bách Khoa Hà Nội**

## Mô tả

Dataset phục vụ hệ thống nhận diện hành vi tài xế lái xe ô tô.

## 5 Lớp Hành Vi

| Index | Tên | Mô tả |
|-------|-----|--------|
| 0 | normal | Tư thế ổn định, không phone, không uống, không có dấu hiệu buồn ngủ |
| 1 | drowsy | Mất tỉnh táo rõ (đầu gục rồi phục hồi, mắt nhắm lâu) |
| 2 | sleepy | Buồn ngủ nhưng chưa gục (mắt khép lâu, chớp chậm, đầu hơi cúi) |
| 3 | drinking | Chuỗi tay lên, chai/cốc tới miệng, uống, hạ xuống |
| 4 | phone | Điện thoại trong tay và người lái đang nhìn/tương tác |

## Quy Tắc Gán Nhãn

- Một frame mắt nhắm chưa đủ để gán **sleepy**. Cần chuỗi liên tục.
- Chỉ thấy chai/cốc chưa phải **drinking** — cần thấy cả chuỗi tay lên-uống-hạ.
- Chỉ thấy điện thoại mà không rõ đang dùng thì chưa phải **phone**.
- Label gốc của nguồn công khai khác label của project cần review trước khi gán.
- Không ép mẫu mơ hồ vào class chỉ để cân bằng dữ liệu.
- **Đơn vị annotation là VIDEO SEGMENT**, không phải frame.

## Cấu Trúc Thư Mục

```
driver_behavior_dataset_v01/
  annotations/   metadata.xlsx, metadata.csv, splits.csv
  features/      đặc trưng đã trích (.npz)
  processed/     frames/ — frame JPG đã trích
  raw/           video gốc (BẤT BIẾN, không sửa/xóa)
  scripts/       các script xử lý
  segments/      video đã cắt theo nhãn
```

## Schema Metadata

Cột trong metadata.xlsx/csv:

| Cột | Mô tả |
|-----|--------|
| segment_id | Mã định danh duy nhất (SEG000001...) |
| source | Nguồn video (public, self_recorded, ...) |
| source_video | Đường dẫn tương đối so với raw/ |
| subject_id | Mã người (S001, S002, ...; PUB_xxx cho public) |
| start_time | Thời điểm bắt đầu (HH:MM:SS.mmm) |
| end_time | Thời điểm kết thúc |
| duration | Thời lượng (giây, tính lại từ start/end) |
| label | Nhãn hành vi (normal/drowsy/sleepy/drinking/phone) |
| camera_view | Góc quay (front/left/right/dashboard/unknown) |
| lighting | Ánh sáng (day/night/low_light/mixed/unknown) |
| glasses | Kính (none/glasses/sunglasses/unknown) |
| occlusion | Che khuất (none/partial/heavy/unknown) |
| quality | Chất lượng (good/acceptable/bad) |
| notes | Ghi chú tự do |

## Quy Trình Xử Lý Dữ Liệu

```
raw/ -> segments/ (FFmpeg cắt theo metadata) ->
processed/frames/ (frame JPG) ->
features/landmarks/ (landmark .npz) ->
Train / Val / Test
```

## Chống Data Leakage

- Split **theo SUBJECT** trước, rồi mới tạo segment, frame, window.
- Thống kê chuẩn hóa (mean/std) CHỈ tính từ tập train.
- Tập test chỉ đánh giá MỘT LẦN với checkpoint tốt nhất theo validation.

## Lưu Ý Quyền Riêng Tư

Video chứa khuôn mặt người. **Không commit video lên GitHub**.
Video được upload riêng thành Kaggle Dataset (private).
Khi dùng video public không biết danh tính người quay, sử dụng `subject_id = PUB_<mã>`.
Không đảm bảo subject-independent hoàn toàn với video public.

## Giới Hạn Đã Biết

- Dữ liệu public có thể không đảm bảo tính độc lập giữa các tập.
- Với video self-recorded, đảm bảo không có người nào xuất hiện ở cả train lẫn test.
- Kết quả thực nghiệm chưa có — xem outputs/results_summary.csv sau khi chạy thật.
