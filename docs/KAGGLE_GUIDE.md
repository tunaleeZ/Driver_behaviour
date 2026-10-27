# KAGGLE_GUIDE.md — Hướng Dẫn Chạy Trên Kaggle

## 1. Chuẩn Bị Tài Khoản Kaggle

### Bật Internet (tùy chọn)
- Vào Settings → Phone Verification để xác minh số điện thoại.
- Sau khi xác minh, khi tạo notebook chọn **Internet → On**.

### Chọn GPU
- Trong Notebook settings: Accelerator → **GPU T4 x2** hoặc **P100**.
- Hạn mức GPU và dung lượng thay đổi theo tài khoản/chính sách Kaggle; xem thông tin hiển thị trong phiên, không dựa vào con số cố định trong tài liệu.

---

## 2. Upload Dữ Liệu Video lên Kaggle Dataset

Video thô **không được commit lên GitHub**. Thay vào đó:

1. Vào [kaggle.com/datasets](https://www.kaggle.com/datasets) → **New Dataset**.
2. Đặt tên dataset (ví dụ: `driver-behavior-raw-videos`), chọn **Private**.
3. Upload toàn bộ thư mục `driver_behavior_dataset_v01/raw/` giữ nguyên cấu trúc.
4. Ghi lại `<dataset-slug>` (ví dụ: `username/driver-behavior-raw-videos`).

---

## 3. Tạo Kaggle Secrets cho GitHub Token

Để clone repo private:
1. Vào **Account Settings → API → New Token** để lấy `GITHUB_TOKEN`.
2. Trong Notebook → Add-ons → **Secrets** → thêm secret:
   - Key: `GITHUB_TOKEN`
   - Value: `ghp_...` (personal access token, scope: `repo`)

**KHÔNG hard-code token vào notebook.**

---

## 4. Giới Hạn Storage Kaggle

- `/kaggle/working/`: giới hạn **20 GB**, tối đa **500 file output** khi Save Version.
- `/kaggle/input/`: read-only, tổng cộng có thể mount nhiều dataset.

**Mẹo:** Sau khi trích đặc trưng (`extract_landmarks`, `extract_frames`), đóng gói
`/kaggle/working/dbd_work` thành một Kaggle Dataset mới để tái dùng mà không cần
trích đặc trưng lại mỗi phiên:
```bash
# Trong notebook, sau bước extract xong:
!zip -r /kaggle/working/dbd_features.zip /kaggle/working/dbd_work/features
```
Sau đó upload `dbd_features.zip` thành dataset và mount vào notebook tiếp theo.

---

## 5. Cài Đặt Gói Khi Internet Bật

```python
# Trong notebook kaggle_01_preprocess.ipynb
import subprocess
subprocess.run(["pip", "install", "-q", "-r", "requirements.txt"], check=True)
```

---

## 6. Khi Internet Tắt (Offline Mode)

### Upload sẵn face_landmarker.task

1. Tải `face_landmarker.task` từ URL trong `configs/mediapipe.yaml` về máy.
2. Upload file này thành một Kaggle Dataset riêng (ví dụ: `mediapipe-face-landmarker`).
3. Mount dataset đó vào notebook.
4. Set biến môi trường:
   ```python
   import os
   os.environ["DBM_FACE_LANDMARKER_PATH"] = "/kaggle/input/mediapipe-face-landmarker/face_landmarker.task"
   ```

### Upload wheel cho mediapipe

Nếu mediapipe không thể pip install offline:
1. Tải wheel phù hợp từ [PyPI](https://pypi.org/project/mediapipe/) về máy.
2. Upload thành Kaggle Dataset.
3. Cài bằng: `pip install /kaggle/input/<dataset>/mediapipe-xxx.whl`

---

## 7. Xử Lý Xung Đột Phiên Bản mediapipe/numpy/protobuf

`mediapipe` có thể yêu cầu phiên bản Python, NumPy hoặc protobuf khác với môi trường Kaggle hiện tại. Hãy kiểm tra ràng buộc của wheel tương thích trước khi cài; không nâng cấp NumPy/protobuf một cách mù quáng vì có thể làm hỏng PyTorch/OpenCV đã cài sẵn. Thử cài thêm package mà không thay các package nền:
```bash
pip install mediapipe --no-deps --quiet
```
Nếu import lỗi, kiểm tra phiên bản và thông báo dependency resolver:
```python
import mediapipe; print(mediapipe.__version__)
import numpy; print(numpy.__version__)
import google.protobuf; print(google.protobuf.__version__)
```
Chọn wheel mediapipe phù hợp với Python/NumPy có sẵn hoặc khởi tạo môi trường tương thích riêng; không pin torch/numpy trong `requirements.txt`.

## 8. Chạy Notebook

Mở `notebooks/kaggle_01_preprocess.ipynb` để cấu hình repo, raw dataset và thư mục work. Raw dataset phải được mount read-only; mọi artifact trung gian ghi vào `/kaggle/working/dbd_work`. Sau khi kiểm tra metadata và split, lưu đặc trưng đã trích thành Kaggle Dataset riêng nếu cần dùng lại. Mở `notebooks/kaggle_02_train.ipynb`, gắn dataset đặc trưng và lưu checkpoint trong `/kaggle/working`. Không commit output của notebook vào repo.

---

## 9. Mẹo Tăng Tốc Đọc Dữ Liệu

Kaggle I/O với nhiều file nhỏ (JPG) rất chậm. Dùng cờ `--pack` khi trích frame:
```bash
python driver_behavior_dataset_v01/scripts/extract_frames.py --pack
```
Cờ này lưu thêm file `.npy` uint8 cho mỗi segment để đọc nhanh hơn.

---

## 10. Cấu Trúc Thư Mục trên Kaggle

```
/kaggle/input/<raw-dataset-slug>/   <- dataset video gốc (read-only)
/kaggle/input/<feature-slug>/       <- dataset đặc trưng đã trích (nếu có)
/kaggle/working/dbd_work/           <- thư mục làm việc (ghi được)
  segments/
  processed/
  features/
/kaggle/working/outputs/            <- kết quả training
```

Set biến môi trường trong notebook:
```python
import os
os.environ["DBM_RAW_ROOT"] = "/kaggle/input/<raw-dataset-slug>"
os.environ["DBM_WORK_ROOT"] = "/kaggle/working/dbd_work"
```

