import os
import argparse
from pathlib import Path


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def parse_args():
    parser = argparse.ArgumentParser("Script to process the recording and move files into the correct subfolders")

    parser.add_argument("src", type=str, help="directory containing recording")

    return parser.parse_args()

def is_left(filename: str):
    return "left" in filename.lower()

def is_right(filename: str):
    return "right" in filename.lower()

def is_color(filename):
    return "beauty" in filename.lower()

def is_depth(f):
    return "depth" in f.lower()

def is_motion(f):
    return "motion" in f.lower()

def main(src: str):
    files = os.listdir(src)
    
    ensure_dir(os.path.join(src, "left", "color"))
    ensure_dir(os.path.join(src, "left", "depth"))
    ensure_dir(os.path.join(src, "left", "motion"))
    ensure_dir(os.path.join(src, "right", "color"))
    ensure_dir(os.path.join(src, "right", "depth"))
    ensure_dir(os.path.join(src, "right", "motion"))

    structured = {
        "left": {
            "color": [],
            "depth": [],
            "motion": [],
        },
        "right": {
            "color": [],
            "depth": [],
            "motion": []
        }
    }

    # print(files)
    for f in files:
        if not f.endswith(".png") and not f.endswith(".exr"):
            continue


        view = None
        if is_left(f):
            view = "left"
        elif is_right(f):
            view = "right"
        else:
            raise ValueError(f"Error parsing filename: {f}")

        recorder = None
        if is_color(f):
            recorder = "color"
        elif is_depth(f):
            recorder = "depth"
        elif is_motion(f):
            recorder = "motion"
        else:
            raise ValueError(f"Error parsing filename: {f}")
        structured[view][recorder].append( f)

    for view in structured:
        for recorder in structured[view]:
            for f in structured[view][recorder]:
                os.rename(os.path.join(src, f), os.path.join(src, view, recorder, f))

if __name__ == "__main__":
    args = parse_args()
    main(args.src)