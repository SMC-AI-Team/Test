import os
import json
import csv
import shutil
import warnings
from collections import OrderedDict

import nibabel as nib
import numpy as np


# =========================================================
# 0. 경로 설정
# =========================================================
# [필수] 1~9번을 로컬에서 완료한 뒤, 내부망 서버로 옮긴 processed_raw_niftis 경로
# - 이 폴더 아래에는 환자별 하위 폴더가 있어야 함
# - 각 환자 폴더 안에는 아래 SOURCE_IMAGE_NAME / SOURCE_LABEL_NAME 파일이 있어야 함
# 예시 구조:
#   PROCESSED_ROOT/
#     환자ID_1/orig_series_preprocess_clahe_train_percentile.nii.gz
#     환자ID_1/mask_series_preprocess_clahe_train_percentile.nii.gz
PROCESSED_ROOT = ""

# [필수] 6번에서 생성한 split_6_2_2.json을 내부망 서버로 옮긴 경로
# - train / val / test 환자 ID list가 들어있는 JSON
SPLIT_JSON = ""

# [필수] nnU-Net raw dataset root
# - 내부망에서 export nnUNet_raw=/path/to/nnUNet_raw 를 해두면 자동 사용됨
# - export를 안 쓸 경우 아래 두 번째 인자 "" 안에 직접 경로 입력
NNUNET_RAW = os.environ.get("nnUNet_raw", "")

DATASET_ID = 431
DATASET_NAME = "Dataset431_SMC_CLAHE_PCTL"
DATASET_ROOT = os.path.join(NNUNET_RAW, DATASET_NAME)

IMAGES_TR = os.path.join(DATASET_ROOT, "imagesTr")
LABELS_TR = os.path.join(DATASET_ROOT, "labelsTr")
IMAGES_TS = os.path.join(DATASET_ROOT, "imagesTs")

# nnU-Net 공식 필수 폴더는 아니지만, test set 평가용 GT 보관
LABELS_TS_FOR_EVAL = os.path.join(DATASET_ROOT, "labelsTs_for_eval")

DATASET_JSON_PATH = os.path.join(DATASET_ROOT, "dataset.json")
DATA_MATCHING_CSV_PATH = os.path.join(DATASET_ROOT, "data_matching.csv")
DATA_MATCHING_JSON_PATH = os.path.join(DATASET_ROOT, "data_matching.json")

# 9a에서 저장한 CLAHE train-percentile 방식의 image/mask 파일명
SOURCE_IMAGE_NAME = "orig_series_preprocess_clahe_train_percentile.nii.gz"
SOURCE_LABEL_NAME = "mask_series_preprocess_clahe_train_percentile.nii.gz"


# =========================================================
# 1. 설정
# =========================================================
OVERWRITE_DATASET = True

# 사용자가 원한 파일명 형식
CASE_PREFIX = "case"

# 이미 전처리한 intensity를 유지하기 위한 channel name
# nnU-Net v2에서 channel_names는 normalization behavior에 영향을 줌.
CHANNEL_NAME = "noNorm"


# =========================================================
# 2. 유틸 함수
# =========================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def ensure_clean_dir(path: str):
    if OVERWRITE_DATASET and os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def get_spacing_from_affine(affine: np.ndarray):
    return np.array(nib.affines.voxel_sizes(affine), dtype=np.float32)


def validate_preprocessed_pair(image_path: str, label_path: str):
    """
    nnU-Net raw dataset으로 복사하기 전에 image/mask pair가 정상인지 확인.
    """
    if not os.path.exists(image_path):
        return False, f"image missing: {image_path}", None

    if not os.path.exists(label_path):
        return False, f"label missing: {label_path}", None

    try:
        img_nii = nib.load(image_path)
        lab_nii = nib.load(label_path)

        img = img_nii.get_fdata()
        lab = lab_nii.get_fdata()

        if img.shape != lab.shape:
            return False, f"shape mismatch: image {img.shape}, label {lab.shape}", None

        if not np.allclose(img_nii.affine, lab_nii.affine, atol=1e-4):
            return False, "affine mismatch", None

        if not np.isfinite(img).all():
            return False, "image has NaN or Inf", None

        unique_labels = np.unique(lab)
        if not np.all(np.isin(unique_labels, [0, 1])):
            return False, f"label is not binary: {unique_labels.tolist()}", None

        positive_voxels = int((lab > 0).sum())
        if positive_voxels <= 0:
            return False, "empty label mask", None

        spacing = get_spacing_from_affine(img_nii.affine)

        info = {
            "shape_x": int(img.shape[0]),
            "shape_y": int(img.shape[1]),
            "shape_z": int(img.shape[2]),
            "spacing_x": float(spacing[0]),
            "spacing_y": float(spacing[1]),
            "spacing_z": float(spacing[2]),
            "positive_voxels": positive_voxels,
            "label_values": ",".join([str(int(x)) for x in unique_labels]),
        }

        return True, "ok", info

    except Exception as e:
        return False, str(e), None


def copy_or_raise(src: str, dst: str):
    if not os.path.exists(src):
        raise FileNotFoundError(src)
    shutil.copy2(src, dst)


def write_data_matching_csv(rows, path: str):
    fieldnames = [
        "case_number",
        "case_id",
        "split",
        "original_patient_id",
        "nnunet_image_filename",
        "nnunet_label_filename",
        "source_image_path",
        "source_label_path",
        "target_image_path",
        "target_label_path",
        "shape_x",
        "shape_y",
        "shape_z",
        "spacing_x",
        "spacing_y",
        "spacing_z",
        "positive_voxels",
        "label_values",
        "status",
        "reason",
    ]

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)



def check_required_paths():
    """실행 전 사용자가 채워야 하는 경로를 확인한다."""
    missing = []

    if not PROCESSED_ROOT:
        missing.append("PROCESSED_ROOT: 1~9번 결과물이 들어있는 processed_raw_niftis 경로")
    if not SPLIT_JSON:
        missing.append("SPLIT_JSON: split_6_2_2.json 경로")
    if not NNUNET_RAW:
        missing.append("NNUNET_RAW: nnUNet_raw 경로 또는 환경변수 nnUNet_raw")

    if missing:
        msg = "\n[경로 설정 필요]\n" + "\n".join([f"  - {x}" for x in missing])
        raise RuntimeError(msg)


# =========================================================
# 3. main
# =========================================================
def main():
    check_required_paths()

    print("=" * 80)
    print("make_dataset431_smc_clahe.py")
    print("processed_raw_niftis CLAHE -> nnUNet_raw/Dataset431_SMC_CLAHE")
    print("=" * 80)

    if not os.path.isdir(PROCESSED_ROOT):
        raise RuntimeError(f"PROCESSED_ROOT not found: {PROCESSED_ROOT}")

    if not os.path.exists(SPLIT_JSON):
        raise RuntimeError(f"SPLIT_JSON not found: {SPLIT_JSON}")

    split = load_json(SPLIT_JSON)

    for key in ["train", "val", "test"]:
        if key not in split:
            raise RuntimeError(f"split json must contain '{key}' key: {SPLIT_JSON}")

    ensure_clean_dir(DATASET_ROOT)
    ensure_dir(IMAGES_TR)
    ensure_dir(LABELS_TR)
    ensure_dir(IMAGES_TS)
    ensure_dir(LABELS_TS_FOR_EVAL)

    data_matching_rows = []
    data_matching_json = OrderedDict()

    total_saved_trainval = 0
    total_saved_test = 0
    total_skipped = 0
    case_counter = 1

    # -----------------------------------------------------
    # train + val -> imagesTr / labelsTr
    # test -> imagesTs / labelsTs_for_eval
    # -----------------------------------------------------
    for split_name in ["train", "val", "test"]:
        patient_ids = split[split_name]

        print(f"\n[{split_name.upper()}] {len(patient_ids)} cases")

        for original_pid in patient_ids:
            case_id = f"{CASE_PREFIX}_{case_counter:03d}"
            case_number = case_counter

            source_image_path = os.path.join(PROCESSED_ROOT, original_pid, SOURCE_IMAGE_NAME)
            source_label_path = os.path.join(PROCESSED_ROOT, original_pid, SOURCE_LABEL_NAME)

            ok, reason, pair_info = validate_preprocessed_pair(source_image_path, source_label_path)

            if split_name in ["train", "val"]:
                target_image_filename = f"{case_id}_0000.nii.gz"
                target_label_filename = f"{case_id}.nii.gz"
                target_image_path = os.path.join(IMAGES_TR, target_image_filename)
                target_label_path = os.path.join(LABELS_TR, target_label_filename)
            else:
                target_image_filename = f"{case_id}_0000.nii.gz"
                target_label_filename = f"{case_id}.nii.gz"
                target_image_path = os.path.join(IMAGES_TS, target_image_filename)
                target_label_path = os.path.join(LABELS_TS_FOR_EVAL, target_label_filename)

            row = {
                "case_number": case_number,
                "case_id": case_id,
                "split": split_name,
                "original_patient_id": original_pid,
                "nnunet_image_filename": target_image_filename,
                "nnunet_label_filename": target_label_filename,
                "source_image_path": source_image_path,
                "source_label_path": source_label_path,
                "target_image_path": target_image_path,
                "target_label_path": target_label_path,
                "shape_x": "",
                "shape_y": "",
                "shape_z": "",
                "spacing_x": "",
                "spacing_y": "",
                "spacing_z": "",
                "positive_voxels": "",
                "label_values": "",
                "status": "skip" if not ok else "ok",
                "reason": reason,
            }

            if not ok:
                warnings.warn(f"[SKIP] {split_name} | {original_pid} | {reason}")
                data_matching_rows.append(row)
                data_matching_json[case_id] = row
                total_skipped += 1
                case_counter += 1
                continue

            # pair_info 추가
            for k, v in pair_info.items():
                row[k] = v

            copy_or_raise(source_image_path, target_image_path)
            copy_or_raise(source_label_path, target_label_path)

            data_matching_rows.append(row)
            data_matching_json[case_id] = row

            if split_name in ["train", "val"]:
                total_saved_trainval += 1
            else:
                total_saved_test += 1

            print(
                f"[OK] {split_name:5s} | {original_pid} -> "
                f"{target_image_filename}, {target_label_filename}"
            )

            case_counter += 1

    # -----------------------------------------------------
    # dataset.json 생성
    # -----------------------------------------------------
    # labels는 정렬 순서가 중요할 수 있으므로 OrderedDict 사용
    dataset_json = OrderedDict()
    dataset_json["channel_names"] = OrderedDict()
    dataset_json["channel_names"]["0"] = CHANNEL_NAME

    dataset_json["labels"] = OrderedDict()
    dataset_json["labels"]["background"] = 0
    dataset_json["labels"]["tumor"] = 1

    dataset_json["numTraining"] = int(total_saved_trainval)
    dataset_json["file_ending"] = ".nii.gz"

    # 선택 사항이지만 기록용으로 넣어둠
    dataset_json["name"] = "SMC"
    dataset_json["description"] = (
        "Preprocessed retroperitoneal sarcoma CT dataset with CLAHE. "
        "Images are already resampled, train-foreground HU clipped, CLAHE-enhanced, and saved as 0~1 noNorm input."
    )
    dataset_json["reference"] = "SMC retroperitoneal sarcoma dataset"
    dataset_json["release"] = "1.0"

    with open(DATASET_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset_json, f, ensure_ascii=False, indent=2)

    # -----------------------------------------------------
    # matching 저장
    # -----------------------------------------------------
    write_data_matching_csv(data_matching_rows, DATA_MATCHING_CSV_PATH)
    save_json(data_matching_json, DATA_MATCHING_JSON_PATH)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Dataset root       : {DATASET_ROOT}")
    print(f"imagesTr           : {IMAGES_TR}")
    print(f"labelsTr           : {LABELS_TR}")
    print(f"imagesTs           : {IMAGES_TS}")
    print(f"labelsTs_for_eval  : {LABELS_TS_FOR_EVAL}")
    print(f"dataset.json       : {DATASET_JSON_PATH}")
    print(f"data_matching.csv  : {DATA_MATCHING_CSV_PATH}")
    print(f"data_matching.json : {DATA_MATCHING_JSON_PATH}")
    print(f"numTraining        : {total_saved_trainval}")
    print(f"numTest            : {total_saved_test}")
    print(f"skipped            : {total_skipped}")

    print("\n[Next check]")
    print(f"cat {DATASET_JSON_PATH}")
    print(f"head -20 {DATA_MATCHING_CSV_PATH}")
    print(f"find {DATASET_ROOT} -maxdepth 2 -type f | head -50")


if __name__ == "__main__":
    main()