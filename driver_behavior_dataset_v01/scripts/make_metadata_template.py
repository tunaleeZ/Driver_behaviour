#!/usr/bin/env python3
"""
make_metadata_template.py — Generate annotations/metadata.xlsx template.

Tạo file metadata.xlsx với:
  - Sheet "metadata": header + 3 dòng ví dụ (ghi rõ là ví dụ)
  - Sheet "guide": giải thích cột, enum, quy tắc gán nhãn
  - Data validation dropdown cho label, quality, camera_view, lighting, glasses, occlusion

Ví dụ sử dụng:
    python driver_behavior_dataset_v01/scripts/make_metadata_template.py
    python driver_behavior_dataset_v01/scripts/make_metadata_template.py --overwrite
    python driver_behavior_dataset_v01/scripts/make_metadata_template.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from dbm.config import load_config
from dbm.data.metadata import REQUIRED_COLUMNS
from dbm.labels import label_names
from dbm.logging_utils import setup_logging
from dbm.paths import DatasetPaths

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Example data (clearly marked as examples)
# ---------------------------------------------------------------------------

_EXAMPLE_ROWS = [
    {
        "segment_id": "SEG000001",
        "source": "public",
        "source_video": "public/video_001.mp4",
        "subject_id": "PUB_v001",
        "start_time": "00:00:10.000",
        "end_time": "00:00:40.000",
        "duration": "30.000",
        "label": "normal",
        "camera_view": "front",
        "lighting": "day",
        "glasses": "none",
        "occlusion": "none",
        "quality": "good",
        "notes": "[EXAMPLE] Xoá dòng này khi điền dữ liệu thật",
    },
    {
        "segment_id": "SEG000002",
        "source": "public",
        "source_video": "public/video_001.mp4",
        "subject_id": "PUB_v001",
        "start_time": "00:01:05.000",
        "end_time": "00:01:25.000",
        "duration": "20.000",
        "label": "drowsy",
        "camera_view": "front",
        "lighting": "day",
        "glasses": "none",
        "occlusion": "none",
        "quality": "good",
        "notes": "[EXAMPLE] Xoá dòng này khi điền dữ liệu thật",
    },
    {
        "segment_id": "SEG000003",
        "source": "self_recorded",
        "source_video": "self_recorded/rec_001.mp4",
        "subject_id": "S001",
        "start_time": "00:00:05.000",
        "end_time": "00:00:20.000",
        "duration": "15.000",
        "label": "drinking",
        "camera_view": "front",
        "lighting": "day",
        "glasses": "none",
        "occlusion": "none",
        "quality": "acceptable",
        "notes": "[EXAMPLE] Xoá dòng này khi điền dữ liệu thật",
    },
]

# ---------------------------------------------------------------------------
# Guide content
# ---------------------------------------------------------------------------

_GUIDE_ROWS = [
    ("Cột", "Mô tả", "Giá trị hợp lệ / Ghi chú"),
    ("segment_id", "Mã định danh duy nhất, bất biến", "SEGxxxxxx (6 chữ số). Để trống để script tự gán."),
    ("source", "Nguồn video", "public, self_recorded, hoặc tên tổ chức khác"),
    ("source_video", "Đường dẫn video tương đối so với raw/", "VD: public/video_001.mp4"),
    ("subject_id", "Mã người. Video public dùng PUB_<mã>", "S001, S002, ..., PUB_v001, ..."),
    ("start_time", "Thời điểm bắt đầu segment", "HH:MM:SS.mmm (đặt cột này là Text trong Excel)"),
    ("end_time", "Thời điểm kết thúc segment", "HH:MM:SS.mmm (đặt cột này là Text trong Excel)"),
    ("duration", "Thời lượng (giây) — script tính lại tự động", "Có thể để trống; không dùng giá trị này làm gốc"),
    ("label", "Nhãn hành vi", "normal | drowsy | sleepy | drinking | phone"),
    ("camera_view", "Góc quay camera", "front | left | right | dashboard | unknown"),
    ("lighting", "Điều kiện ánh sáng", "day | night | low_light | mixed | unknown"),
    ("glasses", "Kính của người lái", "none | glasses | sunglasses | unknown"),
    ("occlusion", "Mức độ che khuất khuôn mặt", "none | partial | heavy | unknown"),
    ("quality", "Chất lượng segment", "good | acceptable | bad (bad = không dùng để train)"),
    ("notes", "Ghi chú tự do", "VD: lý do loại, vấn đề ánh sáng, ..."),
    ("", "", ""),
    ("=== QUY TẮC GÁN NHÃN ===", "", ""),
    ("normal", "Tư thế ổn định, không phone, không uống, không có dấu hiệu buồn ngủ", ""),
    ("drowsy", "Mất tỉnh táo rõ: đầu gục rồi phục hồi, mắt nhắm lâu. Cần chuỗi thời gian rõ ràng.", ""),
    ("sleepy", "Buồn ngủ nhưng chưa gục: mắt khép lâu, chớp chậm, đầu hơi cúi.", ""),
    ("drinking", "Chuỗi hoàn chỉnh: tay lên → chai/cốc tới miệng → uống → hạ xuống.", ""),
    ("phone", "Điện thoại trong tay VÀ người lái đang nhìn/tương tác với điện thoại.", ""),
    ("", "", ""),
    ("=== CHỐNG DATA LEAKAGE ===", "", ""),
    ("", "Split theo SUBJECT trước, rồi mới tạo segment, frame, window.", ""),
    ("", "Một subject chỉ xuất hiện trong đúng MỘT tập (train/val/test).", ""),
    ("", "Video public: dùng PUB_<mã video> làm subject_id.", ""),
]


# ---------------------------------------------------------------------------
# Main creation logic
# ---------------------------------------------------------------------------

def create_template(output_path: Path, schema: dict, dry_run: bool = False) -> None:
    """Create the metadata.xlsx template file.

    Parameters
    ----------
    output_path:
        Path to write the xlsx file.
    schema:
        Parsed schema.yaml dict for valid enum values.
    dry_run:
        If True, only log what would be done.
    """
    try:
        import openpyxl  # type: ignore
        from openpyxl.styles import Font, PatternFill, Alignment  # type: ignore
        from openpyxl.utils import get_column_letter  # type: ignore
        from openpyxl.worksheet.datavalidation import DataValidation  # type: ignore
    except ImportError:
        logger.error("openpyxl is required. Install with: pip install openpyxl")
        raise

    import pandas as pd

    if dry_run:
        logger.info("[DRY-RUN] Would create: %s", output_path)
        return

    wb = openpyxl.Workbook()

    # ----------------------------------------------------------------
    # Sheet 1: metadata
    # ----------------------------------------------------------------
    ws = wb.active
    ws.title = "metadata"

    # Header row
    header_fill = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
    header_font = Font(bold=True, color="FFFFFF")

    for col_idx, col_name in enumerate(REQUIRED_COLUMNS, start=1):
        cell = ws.cell(row=1, column=col_idx, value=col_name)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center")

    # Example rows
    example_fill = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")
    for row_idx, example in enumerate(_EXAMPLE_ROWS, start=2):
        for col_idx, col_name in enumerate(REQUIRED_COLUMNS, start=1):
            cell = ws.cell(row=row_idx, column=col_idx, value=example.get(col_name, ""))
            cell.fill = example_fill

    # Auto-size columns
    for col_idx, col_name in enumerate(REQUIRED_COLUMNS, start=1):
        col_letter = get_column_letter(col_idx)
        max_len = max(len(col_name), 12)
        ws.column_dimensions[col_letter].width = max_len + 4

    # Freeze header row
    ws.freeze_panes = "A2"

    # Data validations (dropdowns)
    def _dv(formula: str, col_letter: str, start_row: int = 2, end_row: int = 10000) -> None:
        dv = DataValidation(
            type="list", formula1=formula, allow_blank=True, showDropDown=False
        )
        ws.add_data_validation(dv)
        dv.sqref = f"{col_letter}{start_row}:{col_letter}{end_row}"

    col_map = {name: get_column_letter(i + 1) for i, name in enumerate(REQUIRED_COLUMNS)}

    _dv('"normal,drowsy,sleepy,drinking,phone"', col_map["label"])
    _dv('"good,acceptable,bad"', col_map["quality"])
    _dv('"front,left,right,dashboard,unknown"', col_map["camera_view"])
    _dv('"day,night,low_light,mixed,unknown"', col_map["lighting"])
    _dv('"none,glasses,sunglasses,unknown"', col_map["glasses"])
    _dv('"none,partial,heavy,unknown"', col_map["occlusion"])

    # ----------------------------------------------------------------
    # Sheet 2: guide
    # ----------------------------------------------------------------
    ws_guide = wb.create_sheet("guide")
    header_font2 = Font(bold=True, color="FFFFFF")
    header_fill2 = PatternFill(start_color="2E75B6", end_color="2E75B6", fill_type="solid")

    for col_idx, header in enumerate(["Cột", "Mô tả", "Giá trị / Ghi chú"], start=1):
        cell = ws_guide.cell(row=1, column=col_idx, value=header)
        cell.font = header_font2
        cell.fill = header_fill2

    for row_idx, guide_row in enumerate(_GUIDE_ROWS, start=2):
        for col_idx, val in enumerate(guide_row, start=1):
            ws_guide.cell(row=row_idx, column=col_idx, value=val)

    ws_guide.column_dimensions["A"].width = 20
    ws_guide.column_dimensions["B"].width = 60
    ws_guide.column_dimensions["C"].width = 50

    # ----------------------------------------------------------------
    # Save
    # ----------------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output_path)
    logger.info("Template saved to: %s", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="make_metadata_template.py",
        description=(
            "Tạo file annotations/metadata.xlsx với sheet 'metadata' (header + ví dụ)\n"
            "và sheet 'guide' (giải thích cột, enum, quy tắc gán nhãn)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--root", type=Path, default=None,
                        help="Dataset root. Mặc định: DatasetPaths.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Đường dẫn file xlsx đầu ra. Mặc định: annotations/metadata.xlsx.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Không tạo file, chỉ in ra thao tác sẽ thực hiện.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Ghi đè file xlsx đã tồn tại.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    paths = DatasetPaths(dataset_root=args.root) if args.root else DatasetPaths()
    output = args.output or paths.metadata_xlsx

    if output.exists() and not args.overwrite and not args.dry_run:
        logger.warning("File already exists: %s (use --overwrite to replace)", output)
        return 0

    schema_path = _REPO_ROOT / "configs" / "schema.yaml"
    schema = load_config(schema_path) if schema_path.exists() else {}

    try:
        create_template(output, schema, dry_run=args.dry_run)
    except Exception as exc:
        logger.error("Failed to create template: %s", exc, exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

