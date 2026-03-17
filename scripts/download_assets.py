#!/usr/bin/env python3
"""Download object assets for CRA experiments.

Supported datasets:
  - ycb        : YCB object set (subset of common manipulation objects)
  - gso        : Google Scanned Objects (selected subset)
  - primitives : programmatically generate primitive shapes (no download)

Usage:
    python scripts/download_assets.py --dataset ycb --output-dir assets/objects
    python scripts/download_assets.py --dataset all
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import urllib.request
import zipfile
import shutil


ASSET_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "assets",
)

# ======================================================================
# YCB Object Set
# ======================================================================

YCB_BASE_URL = "http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/data/berkeley"

# Subset of YCB objects suitable for in-hand manipulation
YCB_OBJECTS = {
    "002_master_chef_can": "Master Chef Can",
    "003_cracker_box": "Cracker Box",
    "004_sugar_box": "Sugar Box",
    "005_tomato_soup_can": "Tomato Soup Can",
    "006_mustard_bottle": "Mustard Bottle",
    "007_tuna_fish_can": "Tuna Fish Can",
    "008_pudding_box": "Pudding Box",
    "009_gelatin_box": "Gelatin Box",
    "010_potted_meat_can": "Potted Meat Can",
    "011_banana": "Banana",
    "021_bleach_cleanser": "Bleach Cleanser",
    "024_bowl": "Bowl",
    "025_mug": "Mug",
    "035_power_drill": "Power Drill",
    "036_wood_block": "Wood Block",
    "037_scissors": "Scissors",
    "040_large_marker": "Large Marker",
    "042_adjustable_wrench": "Adjustable Wrench",
    "051_large_clamp": "Large Clamp",
    "052_extra_large_clamp": "Extra Large Clamp",
    "061_foam_brick": "Foam Brick",
    "065-a_cups": "Cups (a)",
}


def download_ycb(output_dir: str, objects: list[str] | None = None) -> None:
    """Download YCB object meshes."""
    target = os.path.join(output_dir, "ycb")
    os.makedirs(target, exist_ok=True)

    if objects is None:
        objects = list(YCB_OBJECTS.keys())

    print(f"Downloading {len(objects)} YCB objects to {target}...")
    for obj_id in objects:
        obj_dir = os.path.join(target, obj_id)
        if os.path.exists(obj_dir):
            print(f"  [skip] {obj_id} (already exists)")
            continue

        url = f"{YCB_BASE_URL}/{obj_id}/{obj_id}_berkeley_meshes.tgz"
        tgz_path = os.path.join(target, f"{obj_id}.tgz")

        print(f"  Downloading {obj_id}...")
        try:
            urllib.request.urlretrieve(url, tgz_path)
            subprocess.run(
                ["tar", "xzf", tgz_path, "-C", target],
                check=True, capture_output=True,
            )
            os.remove(tgz_path)
            print(f"  [done] {obj_id}")
        except Exception as e:
            print(f"  [FAILED] {obj_id}: {e}")

    print(f"YCB download complete. Objects in: {target}")


# ======================================================================
# Google Scanned Objects (subset)
# ======================================================================

GSO_REPO_URL = "https://github.com/google-research/google-scanned-objects"

GSO_NOTE = """
Google Scanned Objects (GSO):

The full GSO dataset is available at:
  https://app.gazebosim.org/GoogleResearch/fuel/collections/Google%%20Research

For use with IsaacGym, download URDF/mesh files from:
  https://github.com/google-research/google-scanned-objects

Alternatively, use the Objaverse dataset:
  pip install objaverse
  python -c "import objaverse; objaverse.load_objects()"

For best results with CRA, we recommend starting with
simple geometric primitives and gradually adding mesh objects.
"""


def download_gso_info(output_dir: str) -> None:
    """Print download instructions for GSO."""
    target = os.path.join(output_dir, "gso")
    os.makedirs(target, exist_ok=True)
    info_path = os.path.join(target, "DOWNLOAD_INSTRUCTIONS.md")
    with open(info_path, "w") as f:
        f.write(GSO_NOTE)
    print(GSO_NOTE)
    print(f"Instructions saved to: {info_path}")


# ======================================================================
# Primitive generation (no download needed)
# ======================================================================

def generate_primitives_info(output_dir: str) -> None:
    """Document that primitives are generated procedurally."""
    target = os.path.join(output_dir, "primitives")
    os.makedirs(target, exist_ok=True)

    info = """Primitive objects are generated procedurally by IsaacGym:
  - cube    : gym.create_box(sim, size, size, size, options)
  - sphere  : gym.create_sphere(sim, radius, options)
  - cylinder: gym.create_capsule(sim, radius, length, options)

No download is needed. Set object_type in your config:
  object_type: cube     # or sphere, cylinder
"""
    info_path = os.path.join(target, "README.md")
    with open(info_path, "w") as f:
        f.write(info)
    print(info)


# ======================================================================
# Shadow Hand asset
# ======================================================================

HAND_ASSET_NOTE = """
Shadow Hand MJCF/URDF assets:

These are included with IsaacGym (Preview 4).  After installing
IsaacGym, assets are located at:

    $ISAACGYM_ROOT/assets/mjcf/open_ai_assets/hand/shadow_hand.xml

If IsaacGym is installed as a Python package, find the asset root:

    python -c "import isaacgym; print(isaacgym.__path__)"

Then look in the 'assets' subdirectory.

Alternative hand models:
  - Allegro Hand: https://github.com/NYUDimESLab/allegro_hand_ros
  - LEAP Hand:    https://github.com/leap-hand/LEAP_Hand_Sim
  - Ability Hand:  https://github.com/psyonic-inc/IsaacGymAbilityHand
"""


def download_hand_info(output_dir: str) -> None:
    """Print information about hand asset sources."""
    target = os.path.join(output_dir, "hand")
    os.makedirs(target, exist_ok=True)
    info_path = os.path.join(target, "DOWNLOAD_INSTRUCTIONS.md")
    with open(info_path, "w") as f:
        f.write(HAND_ASSET_NOTE)
    print(HAND_ASSET_NOTE)


def main():
    parser = argparse.ArgumentParser(description="Download CRA assets")
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["ycb", "gso", "primitives", "hand", "all"],
                        help="Which dataset to download")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(ASSET_DIR, "objects"),
                        help="Output directory for assets")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.dataset in ("ycb", "all"):
        download_ycb(args.output_dir)

    if args.dataset in ("gso", "all"):
        download_gso_info(args.output_dir)

    if args.dataset in ("primitives", "all"):
        generate_primitives_info(args.output_dir)

    if args.dataset in ("hand", "all"):
        download_hand_info(args.output_dir)

    print("\nDone! See assets/ directory for downloaded and generated data.")


if __name__ == "__main__":
    main()
