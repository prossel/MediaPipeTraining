#!/usr/bin/env python3
"""
Split a Label Studio "COCO with images" export into train/val sets.

Features
- Works with standard COCO annotations (instances-like): images, annotations, categories.
- Keeps "info" and "licenses" if present.
- Copies or symlinks images into output/train/images and output/val/images.
- Optional stratification by each image's "primary category" (the category with the most annotations on that image).
- Prints a per-split summary (#images, #annotations, per-category image counts).

Usage
------
python split_coco_dataset.py \
  --coco /path/to/export/annotations.json \
  --images-dir /path/to/export/images \
  --out-dir /path/to/output \
  --val-ratio 0.2 \
  --seed 42 \
  --link-method copy \
  --stratify primary-category

Notes
-----
- If --images-dir is omitted, the script assumes there is a folder named "images"
  next to the annotations json.
- Valid --link-method values: copy, symlink, hardlink.
- Valid --stratify values: none, primary-category.
"""

import argparse
import json
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Set


def _parse_args():
    p = argparse.ArgumentParser(description="Split a COCO-with-images export into train/val.")
    p.add_argument("--coco", type=str, required=True,
                   help="Path to the COCO annotations JSON (e.g., annotations.json).")
    p.add_argument("--images-dir", type=str, default=None,
                   help="Path to the images directory. If omitted, assumes '<coco_dir>/images'.")
    p.add_argument("--out-dir", type=str, required=True,
                   help="Where to write train/val splits.")
    p.add_argument("--val-ratio", type=float, default=0.2,
                   help="Fraction of images to place in the validation split (default 0.2).")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default 42).")
    p.add_argument("--link-method", type=str, default="copy",
                   choices=["copy", "symlink", "hardlink"],
                   help="How to place images in split folders (default: copy).")
    p.add_argument("--stratify", type=str, default="none",
                   choices=["none", "primary-category"],
                   help="Stratify by image's primary category (default: none).")
    p.add_argument("--keep-unused-categories", action="store_true",
                   help="Keep categories even if they do not appear in a split.")
    p.add_argument("--fail-on-missing-images", action="store_true",
                   help="Fail if an image file referenced in the JSON is missing on disk.")
    return p.parse_args()


def _load_coco(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # basic sanity
    for key in ["images", "annotations", "categories"]:
        if key not in data:
            print(f"[ERROR] Missing key '{key}' in COCO JSON.", file=sys.stderr)
            sys.exit(1)
    # remove path from image file names
    for im in data["images"]:
        fname = im.get("file_name") or im.get("filename")
        if fname:
            im["file_name"] = Path(fname).name
            
    return data


def _ensure_out_dirs(base: Path):
    for split in ["train", "val"]:
        (base / split / "images").mkdir(parents=True, exist_ok=True)


def _image_primary_category(
    ann_by_image: Dict[int, List[Dict]],
) -> Dict[int, int]:
    """
    For each image_id, return the category_id that appears most among its annotations.
    If an image has no annotations, return -1.
    """
    result = {}
    for img_id, anns in ann_by_image.items():
        if not anns:
            result[img_id] = -1
            continue
        cnt = Counter([a["category_id"] for a in anns if "category_id" in a])
        if cnt:
            result[img_id] = cnt.most_common(1)[0][0]
        else:
            result[img_id] = -1
    return result


def _stratified_split(
    image_ids: List[int],
    primary_cat: Dict[int, int],
    val_ratio: float,
    rng: random.Random,
) -> Tuple[Set[int], Set[int]]:
    """
    Simple stratification by primary category.
    For each category bucket, split proportionally.
    """
    # bucket images by their primary category
    buckets = defaultdict(list)
    for iid in image_ids:
        buckets[primary_cat.get(iid, -1)].append(iid)

    train_ids: Set[int] = set()
    val_ids: Set[int] = set()

    for cat, bucket in buckets.items():
        rng.shuffle(bucket)
        n_val = int(round(len(bucket) * val_ratio))
        val_ids.update(bucket[:n_val])
        train_ids.update(bucket[n_val:])

    return train_ids, val_ids


def _random_split(
    image_ids: List[int], val_ratio: float, rng: random.Random
) -> Tuple[Set[int], Set[int]]:
    rng.shuffle(image_ids)
    n_val = int(round(len(image_ids) * val_ratio))
    val_ids = set(image_ids[:n_val])
    train_ids = set(image_ids[n_val:])
    return train_ids, val_ids


def _subset_coco(
    coco: Dict, keep_image_ids: Set[int], keep_unused_categories: bool
) -> Dict:
    keep_images = [im for im in coco["images"] if im["id"] in keep_image_ids]
    keep_image_ids = set(im["id"] for im in keep_images)

    keep_annotations = [a for a in coco["annotations"] if a["image_id"] in keep_image_ids]

    if keep_unused_categories:
        keep_categories = coco["categories"]
    else:
        used_cats = set(a["category_id"] for a in keep_annotations)
        keep_categories = [c for c in coco["categories"] if c["id"] in used_cats]

    subset = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": keep_images,
        "annotations": keep_annotations,
        "categories": keep_categories,
    }
    return subset


def _link_or_copy(src: Path, dst: Path, method: str):
    if method == "copy":
        shutil.copy2(src, dst)
    elif method == "symlink":
        # create relative symlink if possible
        rel = os.path.relpath(src, start=dst.parent)
        if dst.exists():
            dst.unlink()
        dst.symlink_to(rel)
    elif method == "hardlink":
        if dst.exists():
            dst.unlink()
        os.link(src, dst)
    else:
        raise ValueError(f"Unknown method: {method}")


def _place_images(
    images: List[Dict], images_dir: Path, out_split_dir: Path, method: str, fail_on_missing: bool
):
    out_img_dir = out_split_dir / "images"
    for im in images:
        fname = im.get("file_name") or im.get("filename")
        if not fname:
            print(f"[WARN] image without file_name: id={im.get('id')}")
            continue
        src = images_dir / fname
        dst = out_img_dir / Path(fname).name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not src.exists():
            msg = f"[WARN] Missing image on disk: {src}"
            if fail_on_missing:
                print("[ERROR]", msg, file=sys.stderr)
                sys.exit(2)
            else:
                print(msg, file=sys.stderr)
                continue
        _link_or_copy(src, dst, method)


def _summarize(split_name: str, coco_subset: Dict):
    n_images = len(coco_subset["images"])
    n_anns = len(coco_subset["annotations"])
    print(f"[{split_name}] #images={n_images}, #annotations={n_anns}")
    # per-category image counts (images having >=1 ann of that cat)
    anns_by_img = defaultdict(list)
    for a in coco_subset["annotations"]:
        anns_by_img[a["image_id"]].append(a)
    img_primary = _image_primary_category(anns_by_img)

    per_cat = Counter([c for c in img_primary.values() if c != -1])
    print(f"[{split_name}] per-category image counts (by primary category): {dict(per_cat)}")


def main():
    args = _parse_args()
    rng = random.Random(args.seed)

    coco_path = Path(args.coco).expanduser().resolve()
    coco_dir = coco_path.parent
    if args.images_dir is None:
        images_dir = (coco_dir / "images").resolve()
    else:
        images_dir = Path(args.images_dir).expanduser().resolve()

    if not coco_path.exists():
        print(f"[ERROR] COCO JSON not found: {coco_path}", file=sys.stderr)
        sys.exit(1)
    if not images_dir.exists():
        print(f"[ERROR] Images dir not found: {images_dir}", file=sys.stderr)
        sys.exit(1)

    coco = _load_coco(coco_path)

    # Build helper maps
    ann_by_image: Dict[int, List[Dict]] = defaultdict(list)
    for a in coco["annotations"]:
        ann_by_image[a["image_id"]].append(a)

    image_ids = [im["id"] for im in coco["images"]]

    # Split
    if args.stratify == "primary-category":
        primary = _image_primary_category(ann_by_image)
        train_ids, val_ids = _stratified_split(image_ids, primary, args.val_ratio, rng)
    else:
        train_ids, val_ids = _random_split(image_ids, args.val_ratio, rng)

    # Subset JSONs
    coco_train = _subset_coco(coco, train_ids, args.keep_unused_categories)
    coco_val = _subset_coco(coco, val_ids, args.keep_unused_categories)

    # Prepare output dirs
    out_base = Path(args.out_dir).expanduser().resolve()
    _ensure_out_dirs(out_base)

    # Write JSONs
    (out_base / "train").mkdir(parents=True, exist_ok=True)
    (out_base / "val").mkdir(parents=True, exist_ok=True)

    train_json = out_base / "train" / "labels.json"
    val_json = out_base / "val" / "labels.json"

    with train_json.open("w", encoding="utf-8") as f:
        json.dump(coco_train, f, ensure_ascii=False, indent=2)
    with val_json.open("w", encoding="utf-8") as f:
        json.dump(coco_val, f, ensure_ascii=False, indent=2)

    # Place/copy images
    _place_images(coco_train["images"], images_dir, out_base / "train", args.link_method, args.fail_on_missing_images)
    _place_images(coco_val["images"], images_dir, out_base / "val", args.link_method, args.fail_on_missing_images)

    # Final summary
    print("\n=== Split summary ===")
    _summarize("train", coco_train)
    _summarize("val", coco_val)

    print("\nDone.")
    print(f"Train JSON: {train_json}")
    print(f"Val JSON:   {val_json}")
    print(f"Train imgs: {out_base/'train'/'images'}")
    print(f"Val imgs:   {out_base/'val'/'images'}")


if __name__ == "__main__":
    main()
