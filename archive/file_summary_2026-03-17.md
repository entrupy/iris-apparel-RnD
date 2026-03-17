# File Summary — 2026-03-17

Summary of recently viewed files in `/root/iris/resources/`.

---

## 1. `apparel_supreme_until_dec_2025_front/yolo_results/train/camera/front/0/yolo_results.csv`
- **Lines:** 2
- **Has content:** NO — file contains only blank lines, no YOLO detection results.

## 2. `.resource/installed/apparel_supreme_until_dec_2025.json`
- **Lines:** 1
- **Has content:** YES
- **Details:** JSON metadata for an S3 resource.
  - **type:** s3_files
  - **name:** apparel_supreme_until_dec_2025
  - **version:** 2026.3.16a2
  - **s3prefix:** `s3://entrupy-exp/resources/s3_files/apparel_supreme_until_dec_2025/2026.3.16a2/`
  - **total_file_size:** ~25.5 MB
  - **files:** 8

## 3. `apparel_supreme_until_dec_2025_front_exterior_logo/yolo_results/train/camera/front_exterior_logo/0/yolo_results.csv`
- **Lines:** 2
- **Has content:** NO — file contains only blank lines, no YOLO detection results.

## 4. `apparel_supreme_until_dec_2025_front/train/metadata.csv`
- **Lines:** 9,761 (1 header + 9,760 data rows)
- **Has content:** YES
- **Details:** CSV with columns `session_uuid`, `result_voted`, `internal_merged_result_id`.
  - Contains 9,760 session entries.
  - Most rows have `result_voted = result_id_1` and `internal_merged_result_id = 1`.
  - Some rows have `result_voted = uncertain_result_id_None`.

## 5. `apparel_supreme_until_dec_2025_brand_tag/yolo_results/train/camera/brand_tag/0/yolo_results.csv`
- **Lines:** 2
- **Has content:** NO — file contains only blank lines, no YOLO detection results.

## 6. `apparel_supreme_until_dec_2025/num_skipped.txt`
- **Lines:** 1
- **Has content:** YES
- **Details:** Contains `0` — zero items were skipped during processing.

## 7. `apparel_supreme_until_dec_2025/num_items.txt`
- **Lines:** 1
- **Has content:** YES
- **Details:** Contains `9759` — total number of items in the dataset.

---

## Quick Reference

| # | File | Has Content? | Summary |
|---|------|:---:|---------|
| 1 | `front/0/yolo_results.csv` | EMPTY | Blank — no YOLO results |
| 2 | `apparel_supreme_until_dec_2025.json` | YES | S3 resource metadata (8 files, ~25 MB) |
| 3 | `front_exterior_logo/0/yolo_results.csv` | EMPTY | Blank — no YOLO results |
| 4 | `front/train/metadata.csv` | YES | 9,760 session entries (UUID, vote, merged ID) |
| 5 | `brand_tag/0/yolo_results.csv` | EMPTY | Blank — no YOLO results |
| 6 | `num_skipped.txt` | YES | `0` (no items skipped) |
| 7 | `num_items.txt` | YES | `9759` total items |

**Overall:** 4 out of 7 files have actual content. All 3 YOLO result CSVs are empty.
