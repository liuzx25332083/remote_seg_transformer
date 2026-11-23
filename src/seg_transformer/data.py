from __future__ import annotations

import hashlib
import io
import json
import random
import time
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from .config import resolve_path


HF_BASE = "https://huggingface.co/datasets"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(
    url: str,
    destination: Path,
    retries: int = 3,
    verify_image: bool = True,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "deepglobe-cpu-course-project/0.1"},
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                temporary.write_bytes(response.read())
            if verify_image:
                with Image.open(temporary) as image:
                    image.verify()
            temporary.replace(destination)
            return
        except Exception as error:  # pragma: no cover - network path
            last_error = error
            temporary.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(2 * attempt)
    raise RuntimeError(f"Failed to download {url}: {last_error}")


def download_source_subset(config: dict[str, Any]) -> pd.DataFrame:
    data_root = resolve_path(config["data_root"])
    raw_root = data_root / "raw" / "deepglobe"
    manifest_dir = data_root / "manifests"
    raw_root.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    repo = config["source_repo"]
    revision = config["source_revision"]
    metadata_path = manifest_dir / "source_metadata.csv"
    encoded_repo = "/".join(urllib.parse.quote(part) for part in repo.split("/"))
    metadata_url = (
        f"{HF_BASE}/{encoded_repo}/resolve/"
        f"{urllib.parse.quote(revision)}/metadata.csv?download=true"
    )
    _download(metadata_url, metadata_path, verify_image=False)
    metadata = pd.read_csv(metadata_path, dtype={"image_id": str})
    metadata = metadata.loc[
        (metadata["split"] == config["source_split"])
        & metadata["mask_path"].fillna("").astype(str).ne("")
    ].copy()
    if metadata.empty:
        raise RuntimeError("No labeled training images were found in source metadata")

    metadata = metadata.sort_values("image_id").reset_index(drop=True)
    rng = random.Random(int(config["source_sampling_seed"]))
    count = min(int(config["source_image_count"]), len(metadata))
    selected_indices = sorted(rng.sample(range(len(metadata)), count))
    selected = metadata.iloc[selected_indices].copy()

    def fetch_one(row: Any) -> dict[str, Any]:
        sat_relative = str(row.sat_image_path)
        mask_relative = str(row.mask_path)
        sat_path = raw_root / sat_relative
        mask_path = raw_root / mask_relative
        for relative, destination in (
            (sat_relative, sat_path),
            (mask_relative, mask_path),
        ):
            encoded_path = urllib.parse.quote(relative, safe="/")
            url = (
                f"{HF_BASE}/{encoded_repo}/resolve/"
                f"{urllib.parse.quote(revision)}/{encoded_path}?download=true"
            )
            _download(url, destination)

        with Image.open(sat_path) as image:
            image_size = image.size
            image_mode = image.mode
        with Image.open(mask_path) as mask:
            mask_size = mask.size
            mask_mode = mask.mode
        if image_size != mask_size:
            raise ValueError(
                f"Image/mask size mismatch for {row.image_id}: "
                f"{image_size} vs {mask_size}"
            )
        return {
            "image_id": str(row.image_id),
            "source_split": str(row.split),
            "sat_path": sat_relative,
            "mask_path": mask_relative,
            "width": image_size[0],
            "height": image_size[1],
            "image_mode": image_mode,
            "mask_mode": mask_mode,
            "sat_sha256": _sha256(sat_path),
            "mask_sha256": _sha256(mask_path),
        }

    rows = []
    workers = int(config.get("download_workers", 8))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(fetch_one, row): str(row.image_id)
            for row in selected.itertuples(index=False)
        }
        for index, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            print(f"downloaded {index:03d}/{len(selected)} image_id={row['image_id']}")
    rows.sort(key=lambda item: item["image_id"])

    result = pd.DataFrame(rows)
    output = manifest_dir / "downloaded_source_manifest.csv"
    result.to_csv(output, index=False, encoding="utf-8")
    summary = {
        "source_repo": repo,
        "source_revision": revision,
        "requested_images": int(config["source_image_count"]),
        "downloaded_images": len(result),
        "unique_image_ids": int(result["image_id"].nunique()),
        "unique_image_hashes": int(result["sat_sha256"].nunique()),
        "image_sizes": sorted({(int(r.width), int(r.height)) for r in result.itertuples()}),
        "manifest": str(output),
    }
    (manifest_dir / "download_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved source manifest: {output}")
    return result


def rgb_mask_to_labels(mask_rgb: np.ndarray, palette: list[list[int]]) -> np.ndarray:
    if mask_rgb.ndim != 3 or mask_rgb.shape[2] != 3:
        raise ValueError("Mask must have shape H x W x 3")
    labels = np.full(mask_rgb.shape[:2], -1, dtype=np.int64)
    for label, color in enumerate(palette):
        color_array = np.asarray(color, dtype=np.uint8)
        labels[np.all(mask_rgb == color_array, axis=-1)] = label
    return labels


def _dominant_label(labels: np.ndarray, ignore_index: int) -> int:
    counts = np.bincount(
        labels.ravel(),
        minlength=ignore_index + 1,
    )
    counts[ignore_index] = 0
    return int(np.argmax(counts))


def _stratified_image_split(
    source: pd.DataFrame,
    ratios: dict[str, float],
    seed: int,
) -> pd.DataFrame:
    rng = random.Random(seed)
    assignments: dict[str, str] = {}
    for dominant_label, group in source.groupby("dominant_label", sort=True):
        ids = sorted(group["image_id"].astype(str).tolist())
        rng.shuffle(ids)
        count = len(ids)
        if count <= 2:
            split_names = ["train"] * count
        else:
            n_test = max(1, int(round(count * ratios["test"])))
            n_val = max(1, int(round(count * ratios["val"])))
            if n_test + n_val >= count:
                n_test = 1
                n_val = 1
            split_names = (
                ["test"] * n_test
                + ["val"] * n_val
                + ["train"] * (count - n_test - n_val)
            )
            rng.shuffle(split_names)
        for image_id, split_name in zip(ids, split_names):
            assignments[image_id] = split_name

    result = source.copy()
    result["split"] = result["image_id"].map(assignments)
    if set(result["split"]) != {"train", "val", "test"}:
        raise RuntimeError("Image-level split did not produce all three splits")
    return result


def _grid_origins(width: int, height: int, crop_size: int, patches: int) -> list[tuple[int, int]]:
    side = int(round(np.sqrt(patches)))
    if side * side != patches:
        raise ValueError("patches_per_image must be a perfect square")
    xs = np.linspace(0, max(0, width - crop_size), side).round().astype(int)
    ys = np.linspace(0, max(0, height - crop_size), side).round().astype(int)
    return [(int(y), int(x)) for y in ys for x in xs]


def prepare_patches(config: dict[str, Any]) -> pd.DataFrame:
    data_root = resolve_path(config["data_root"])
    source_root = resolve_path(config.get("source_data_root", config["data_root"]))
    (data_root / "manifests").mkdir(parents=True, exist_ok=True)
    raw_root = source_root / "raw" / "deepglobe"
    source_path = source_root / "manifests" / "downloaded_source_manifest.csv"
    if not source_path.exists():
        raise FileNotFoundError("Run the download command first")
    source = pd.read_csv(source_path, dtype={"image_id": str})

    palette = config["palette"]
    ignore_index = int(config["ignore_index"])
    records: list[dict[str, Any]] = []
    for row in source.itertuples(index=False):
        mask_path = raw_root / str(row.mask_path)
        with Image.open(mask_path) as mask_image:
            labels = rgb_mask_to_labels(
                np.asarray(mask_image.convert("RGB")),
                palette,
            )
        if np.any(labels < 0):
            bad_count = int(np.sum(labels < 0))
            raise ValueError(
                f"Mask {row.mask_path} contains {bad_count} pixels outside the palette"
            )
        records.append(
            {
                "image_id": str(row.image_id),
                "sat_path": str(row.sat_path),
                "mask_path": str(row.mask_path),
                "dominant_label": _dominant_label(labels, ignore_index),
            }
        )
    source = pd.DataFrame(records)
    source = _stratified_image_split(
        source,
        config["split_ratios"],
        int(config["split_seed"]),
    )
    source.to_csv(
        data_root / "manifests" / "image_split_manifest.csv",
        index=False,
        encoding="utf-8",
    )

    patch_root = data_root / "patches"
    rows: list[dict[str, Any]] = []
    class_counts = np.zeros(int(config["num_classes"]), dtype=np.int64)
    train_class_counts = np.zeros(int(config["num_classes"]), dtype=np.int64)
    rgb_sum = np.zeros(3, dtype=np.float64)
    rgb_sq_sum = np.zeros(3, dtype=np.float64)
    train_pixels = 0

    for image_row in source.itertuples(index=False):
        sat_path = raw_root / str(image_row.sat_path)
        mask_path = raw_root / str(image_row.mask_path)
        with Image.open(sat_path) as image_file:
            image = image_file.convert("RGB")
        with Image.open(mask_path) as mask_file:
            mask_rgb = np.asarray(mask_file.convert("RGB"))
        labels = rgb_mask_to_labels(mask_rgb, palette)
        origins = _grid_origins(
            image.width,
            image.height,
            int(config["crop_size"]),
            int(config["patches_per_image"]),
        )
        split_dir = patch_root / str(image_row.split)
        image_dir = split_dir / "images"
        mask_dir = split_dir / "masks"
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        for patch_index, (top, left) in enumerate(origins):
            crop_box = (
                left,
                top,
                left + int(config["crop_size"]),
                top + int(config["crop_size"]),
            )
            image_patch = image.crop(crop_box).resize(
                (int(config["image_size"]), int(config["image_size"])),
                Image.Resampling.BILINEAR,
            )
            label_patch = labels[
                top : top + int(config["crop_size"]),
                left : left + int(config["crop_size"]),
            ]
            label_patch = np.asarray(
                Image.fromarray(label_patch.astype(np.uint8)).resize(
                    (int(config["image_size"]), int(config["image_size"])),
                    Image.Resampling.NEAREST,
                )
            )
            image_relative = (
                Path(str(image_row.split))
                / "images"
                / f"{image_row.image_id}_p{patch_index:02d}.png"
            )
            mask_relative = (
                Path(str(image_row.split))
                / "masks"
                / f"{image_row.image_id}_p{patch_index:02d}.png"
            )
            image_patch.save(patch_root / image_relative)
            Image.fromarray(label_patch).save(patch_root / mask_relative)
            rows.append(
                {
                    "image_id": image_row.image_id,
                    "patch_index": patch_index,
                    "split": image_row.split,
                    "top": top,
                    "left": left,
                    "image_path": image_relative.as_posix(),
                    "mask_path": mask_relative.as_posix(),
                }
            )
            counts = np.bincount(
                label_patch.ravel(),
                minlength=int(config["num_classes"]),
            )
            class_counts += counts
            if image_row.split == "train":
                train_class_counts += counts
                array = np.asarray(image_patch, dtype=np.float64) / 255.0
                valid = label_patch != ignore_index
                pixels = array[valid]
                rgb_sum += pixels.sum(axis=0)
                rgb_sq_sum += np.square(pixels).sum(axis=0)
                train_pixels += len(pixels)

    manifest = pd.DataFrame(rows)
    manifest_path = data_root / "manifests" / "patch_manifest.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8")
    mean = rgb_sum / max(1, train_pixels)
    variance = np.maximum(rgb_sq_sum / max(1, train_pixels) - np.square(mean), 1e-12)
    stats = {
        "source_images": len(source),
        "images_by_split": source.groupby("split")["image_id"].nunique().to_dict(),
        "patches_by_split": manifest.groupby("split").size().to_dict(),
        "class_pixel_counts": class_counts.tolist(),
        "train_class_pixel_counts": train_class_counts.tolist(),
        "normalization": {"mean": mean.tolist(), "std": np.sqrt(variance).tolist()},
        "crop_size": int(config["crop_size"]),
        "image_size": int(config["image_size"]),
        "patch_manifest": str(manifest_path),
    }
    (data_root / "manifests" / "patch_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return manifest
