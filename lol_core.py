"""
EsportsLABgg League Vision Challenge

How to use:
python lol_core.py <path_to_image>
python lol_core.py starter/task_examples/screen1.jpg
"""

import argparse
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from cv2 import TM_CCOEFF, TM_CCOEFF_NORMED, TM_CCORR, TM_CCORR_NORMED, TM_SQDIFF, TM_SQDIFF_NORMED
from skimage import io

TEMPLATES_DIR = "templates"

NAME_TO_METHOD = {
    "TM_SQDIFF": TM_SQDIFF,
    "TM_SQDIFF_NORMED": TM_SQDIFF_NORMED,
    "TM_CCORR": TM_CCORR,
    "TM_CCORR_NORMED": TM_CCORR_NORMED,
    "TM_CCOEFF": TM_CCOEFF,
    "TM_CCOEFF_NORMED": TM_CCOEFF_NORMED
}

ALLOW_MASK = {
    TM_SQDIFF: True,
    TM_SQDIFF_NORMED: False,
    TM_CCORR: False,
    TM_CCORR_NORMED: True,
    TM_CCOEFF: False,
    TM_CCOEFF_NORMED: False
}

THRESHOLD_FUNCTION = {
    TM_SQDIFF: np.less_equal,
    TM_SQDIFF_NORMED: np.less_equal,
    TM_CCORR: np.greater_equal,
    TM_CCORR_NORMED: np.greater_equal,
    TM_CCOEFF: np.greater_equal,
    TM_CCOEFF_NORMED: np.greater_equal
}

THRESHOLDS = {
    (TM_SQDIFF, "l"): 70_000,
    (TM_SQDIFF, "t"): 200_000,
    (TM_SQDIFF, "r"): 40_000,
    (TM_SQDIFF, "b"): 200_000,
    (TM_CCORR_NORMED, "l"): 0.95,
    (TM_CCORR_NORMED, "t"): 0.95,
    (TM_CCORR_NORMED, "r"): 0.95,
    (TM_CCORR_NORMED, "b"): 0.95,
}


HEART_DELTA = {
    "l": (0, 0),
    "t": (0, -2),
    "r": (2, 130),
    "b": (22, -2)
}

RECTANGLE_TOLERANCE = 2

EDGES = ("l", "t", "r", "b")


def suppression(loc: Tuple[np.ndarray, np.ndarray], supp_dim: str) -> Tuple[np.ndarray, np.ndarray]:
    """Reduce amount of matches found along one of dimensions and pick the center one"""
    # s, sup - suppression dimension
    # r, reg - regular dimension
    if supp_dim == "v":
        reg = loc[1]  # x
        sup = loc[0]  # y
    elif supp_dim == "h":
        reg = loc[0]  # y
        sup = loc[1]  # x
    else:
        raise ValueError

    # ensure values are sorted
    df = pd.DataFrame.from_dict({"sup": sup, "reg": reg}).sort_values(['reg', 'sup'])
    sup = df['sup'].values
    reg = df['reg'].values

    prev_r = None
    prev_s = None
    cache_r = []
    cache_s = []
    result_s = []
    result_r = []

    for r, s in zip(reg, sup):
        if prev_r is None and prev_s is None:
            cache_r.append(r)
            cache_s.append(s)
        elif r == prev_r and s - prev_s:
            cache_r.append(r)
            cache_s.append(s)
        else:
            result_r.append(cache_r[0])
            result_s.append(round(np.array(cache_s).mean()))
            cache_r = [r]
            cache_s = [s]
        prev_r = r
        prev_s = s
    result_r.append(cache_r[0])
    result_s.append(int(np.array(cache_s).mean()))

    result_r = np.array(result_r)
    result_s = np.array(result_s)

    if supp_dim == "v":
        return result_s, result_r
    elif supp_dim == "h":
        return result_r, result_s


EDGES_SUPPRESSION = {
    "l": lambda x: suppression(x, "v"),
    "r": lambda x: suppression(x, "v"),
    "t": lambda x: suppression(x, "h"),
    "b": lambda x: suppression(x, "h"),
}


def get_template_paths(edge_name: str) -> Tuple[str, str]:
    """Get template and mask path"""
    template_path = os.path.join(TEMPLATES_DIR, f"{edge_name}_template.npy")
    mask_path = os.path.join(TEMPLATES_DIR, f"{edge_name}_mask.txt")
    return template_path, mask_path


def save_template(edge_name: str, template: np.ndarray, mask: np.ndarray) -> None:
    """Dump template and mask"""
    assert mask.shape == template.shape[:-1]
    template_path, mask_path = get_template_paths(edge_name)
    np.save(template_path, template)
    # noinspection PyTypeChecker
    np.savetxt(mask_path, mask, fmt="%d")


def read_template(edge_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load template and mask"""
    template_path, mask_path = get_template_paths(edge_name)
    template = np.load(template_path)
    mask = np.loadtxt(mask_path, dtype=np.uint8)
    return template, mask


def find_edges(image_path: str,
               edge_name: str,
               method_name: str) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray, int, int, int]:
    """Find all matches of one edge in the given image"""
    method = NAME_TO_METHOD[method_name]
    template, mask = read_template(edge_name)
    if ALLOW_MASK[method] is False:
        mask = None
        print("mask is not used!")
    threshold = THRESHOLDS[(method, edge_name)]
    threshold_function = THRESHOLD_FUNCTION[method]
    image = io.imread(image_path)

    # COLOR_BGR2GRAY(TOLERANCE=2) works better than COLOR_RGB2GRAY(requires TOLERANCE=3)
    gray_method = cv2.COLOR_BGR2GRAY
    # gray_method = cv2.COLOR_RGB2GRAY
    image_gray = cv2.cvtColor(image, gray_method)
    template_gray = cv2.cvtColor(template, gray_method)
    h, w = template_gray.shape
    res = cv2.matchTemplate(image=image_gray, templ=template_gray, method=method, mask=mask)
    loc = np.where(threshold_function(res, threshold))
    if len(loc[0]) > 0:
        suppr_loc = EDGES_SUPPRESSION[edge_name](loc)
    else:
        suppr_loc = loc
    return suppr_loc, image, h, w, threshold


def build_rectangles(edges_found: Dict[str, Tuple[Tuple[np.ndarray, np.ndarray], int, int]]) \
        -> List[List[Optional[Tuple[int, int]]]]:
    """Group edges into rectangles (health bars)"""
    all_rectangles = []
    edges_coords = {}
    for edge_name, (loc, h, v) in edges_found.items():
        edges_coords[edge_name] = [(y, x) for y, x in zip(*loc)]

    while True:
        heart_candidates = [(edge_name, p) for edge_name, point_list in edges_coords.items() for p in point_list]
        if not heart_candidates:
            break
        heart_edge_name, (heart_y, heart_x) = heart_candidates[0]
        heart_y -= HEART_DELTA[heart_edge_name][0]
        heart_x -= HEART_DELTA[heart_edge_name][1]

        rectangle = []
        for edge_name in EDGES:
            close_edge = None
            for e in edges_coords[edge_name]:
                dy = abs(e[0] - heart_y - HEART_DELTA[edge_name][0])
                # print('dy', dy)
                dx = abs(e[1] - heart_x - HEART_DELTA[edge_name][1])
                # print('dx', dx)
                if max(dy, dx) <= RECTANGLE_TOLERANCE:
                    close_edge = e
                    edges_coords[edge_name].remove(e)
                    break
            rectangle.append(close_edge)
        all_rectangles.append(rectangle)
    return all_rectangles


def find_healthbars(image_path: str, method_name: str) \
        -> Tuple[List[List[Optional[Tuple[int, int]]]], List[Tuple[int, int]]]:
    """Find healthbars in the image"""
    edges_found = {}
    edge_dimensions = []
    for edge_name in EDGES:
        suppr_loc, image, h, w, threshold = find_edges(image_path, edge_name, method_name)
        edges_found[edge_name] = (suppr_loc, h, w)
        edge_dimensions.append((h, w))

    all_rectangles = build_rectangles(edges_found)
    return all_rectangles, edge_dimensions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LOL health bar counter.')
    parser.add_argument('image_path', type=str, help='Image path')
    args = parser.parse_args()
    image_path = args.image_path
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found! {image_path}")
    all_rectangles, _ = find_healthbars(image_path, "TM_SQDIFF")
    print(len(all_rectangles))
