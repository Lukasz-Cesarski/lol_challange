import os
from glob import glob
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
from skimage import io
from cv2 import TM_CCOEFF, TM_CCOEFF_NORMED, TM_CCORR, TM_CCORR_NORMED, TM_SQDIFF, TM_SQDIFF_NORMED

STARTER_DIR = "starter"
TASK_EXAMPLES_DIR = "task_examples"
HEALTH_BARS_DIR = "healthbars"

DATA_DIR = os.path.join(STARTER_DIR, TASK_EXAMPLES_DIR, "")
IMAGE_PATHS = sorted(glob(DATA_DIR + "**.jpg") + glob(DATA_DIR + "**.png"))
RESULTS_DIR = "results"
TEMPLATES_DIR = "templates"

# HB - HEALTH BAR
HB_WIDTH = 138
HB_HEIGHT = 30
LEFT_SIZE = 7
TOP_SIZE = 7
RIGHT_SIZE = 7
BOT_SIZE = 15

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


def read_image_path(image_path):
    image = io.imread(image_path)
    health_bars_save_path = os.path.join(HEALTH_BARS_DIR,
                                         TASK_EXAMPLES_DIR,
                                         os.path.splitext(os.path.basename(image_path))[0] + ".csv")
    if os.path.isfile(health_bars_save_path):
        df_health_bars = pd.read_csv(health_bars_save_path)
    else:
        df_health_bars = None
    return image, health_bars_save_path, df_health_bars


def plot_screen(image, df_health_bars):
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    ax.imshow(image)
    for x, y, _, _, _, _ in df_health_bars.values:
        ax.add_patch(plt.Rectangle((x, y), HB_WIDTH, HB_HEIGHT, edgecolor='black', lw=2, facecolor='none'))


def plot_health_bars(image, df_health_bars):
    fig, ax = plt.subplots(len(df_health_bars), 1, figsize=(16, 20))
    nickname_margin = 0
    # nickname_margin = 15
    colors = ["r", "b"]

    sa = 0.5  # shade alpha
    lo = 3   # border line offset
    lw = 10  # border line width
    for idx, (crop_x, crop_y, l, t, r, b) in enumerate(df_health_bars.values):
        cropped = image[max(0, crop_y - nickname_margin):crop_y + HB_HEIGHT, max(crop_x, 0):crop_x + HB_WIDTH]
        c_height, c_width, _ = cropped.shape
        ax[idx].imshow(cropped)
        ax[idx].set_title(f"IDX={idx}, X={crop_x}, Y={crop_y}")
        # ax[idx].vlines([-lo], 0, cropped.shape[0], linewidth=lw, colors=colors[l])
        # ax[idx].hlines([-lo], 0, cropped.shape[1], linewidth=lw, colors=colors[t])
        # ax[idx].vlines([cropped.shape[1] + lo], 0, cropped.shape[0], linewidth=lw, colors=colors[r])
        # ax[idx].hlines([cropped.shape[0] + lo], 0, cropped.shape[1], linewidth=lw, colors=colors[b])
        if l:
            ax[idx].add_patch(plt.Rectangle((0, 0), LEFT_SIZE, c_height, lw=0, facecolor='white', alpha=sa))
        if t:
            ax[idx].add_patch(plt.Rectangle((0, 0), c_width, TOP_SIZE, lw=0, facecolor='white', alpha=sa))
        if r:
            ax[idx].add_patch(
                plt.Rectangle((c_width-RIGHT_SIZE, 0), RIGHT_SIZE, c_height, lw=0, facecolor='white', alpha=sa))
        if b:
            ax[idx].add_patch(
                plt.Rectangle((0, c_height-BOT_SIZE), c_width, BOT_SIZE, lw=0, facecolor='white', alpha=sa))


def extract_edges(image, df_health_bars):
    el, et, er, eb = [[], [], [], []]  # edges left, edges top, ...
    nickname_margin = 0
    for idx, (crop_x, crop_y, l, t, r, b) in enumerate(df_health_bars.values):
        cropped = image[max(0, crop_y - nickname_margin):crop_y + HB_HEIGHT, max(crop_x, 0):crop_x + HB_WIDTH]
        c_height, c_width, _ = cropped.shape
        if l:
            el.append(cropped[:, :LEFT_SIZE])
        if t:
            et.append(cropped[:TOP_SIZE, :])
        if r:
            er.append(cropped[:, c_width - RIGHT_SIZE:])
        if b:
            eb.append(cropped[c_height - BOT_SIZE:, :])

    return el, et, er, eb


def plot_edges(all_edges):
    h, w, _ = all_edges[0].shape
    length = len(all_edges)
    figsize = (length, 6)
    if h > w:
        n_rows = 1
        n_cols = length
    else:
        n_cols = 2
        n_rows = ceil(length / 2)
    fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize)
    for idx, edge in enumerate(all_edges):
        ax.flat[idx].imshow(edge)


def scatter_edges_map(correct_edges, ref_edges, other_edges, hasher, model):
    correct_hash = np.vstack([hasher.compute(e) for e in correct_edges])
    print("correct_hash", correct_hash.shape)
    ref_hash = np.vstack([hasher.compute(e) for e in ref_edges])
    print("ref_hash", ref_hash.shape)
    other_hash = np.vstack([hasher.compute(e) for e in other_edges])
    print("other_edges", other_hash.shape)

    X = np.vstack([correct_hash, ref_hash, other_hash])
    model.fit(X)

    correct_proj = model.transform(correct_hash)
    ref_proj = model.transform(ref_hash)
    other_proj = model.transform(other_hash)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(correct_proj[:, 0], correct_proj[:, 1], ".r", label="correct")
    ax.plot(ref_proj[:, 0], ref_proj[:, 1], ".b", label="reference")
    ax.plot(other_proj[:, 0], other_proj[:, 1], ".y", label="other")
    ax.legend()


def vertical_suppression(loc):
    """
    IN: (array([ 63,  64,  83,  84,  85, 103, 104, 105, 179, 190, 191, 192]),
        array([475, 475, 435, 435, 435, 297, 297, 297, 626, 629, 629, 629]))
    OUT: (array([ 64,  84, 104, 179, 191]),
          array([475, 435, 297, 626, 629]))
    """
    x_prev = None
    y_prev = None
    cache_x = []
    cache_y = []
    result_y = []
    result_x = []

    for y,x in zip(*loc):
        if x_prev is None and y_prev is None:
            cache_x.append(x)
            cache_y.append(y)
        elif x==x_prev and y-y_prev:
            cache_x.append(x)
            cache_y.append(y)
        else:
            result_x.append(cache_x[0])
            result_y.append(round(np.array(cache_y).mean()))
            cache_x=[x]
            cache_y=[y]
        x_prev = x
        y_prev = y
    result_x.append(cache_x[0])
    result_y.append(int(np.array(cache_y).mean()))
    return np.array(result_y), np.array(result_x)


def horizontal_suppression(loc):
    """
    IN: (array([ 64,  64,  64,  64,  64,  64,  64, 104, 104, 104, 104, 104, 104]),
         array([472, 473, 474, 475, 476, 477, 478, 295, 296, 297, 298, 299, 300]))
    OUT: (array([ 64, 104]),
          array([475, 297]))
    """
    x_prev = None
    y_prev = None
    cache_x = []
    cache_y = []
    result_y = []
    result_x = []

    for y,x in zip(*loc):
        if x_prev is None and y_prev is None:
            cache_x.append(x)
            cache_y.append(y)
        elif y==y_prev and x-x_prev:
            cache_x.append(x)
            cache_y.append(y)
        else:
            result_y.append(cache_y[0])
            result_x.append(round(np.array(cache_x).mean()))
            cache_x=[x]
            cache_y=[y]
        x_prev = x
        y_prev = y
    result_y.append(cache_y[0])
    result_x.append(int(np.array(cache_x).mean()))
    return np.array(result_y), np.array(result_x)

EDGES_SUPPRESSION = {
    "l": vertical_suppression,
    "r" : vertical_suppression,
    "t" : horizontal_suppression,
    "b" : horizontal_suppression,
}


def make_match(image_path, template, method_name, threshold, suppression, plot=True, save=True, mask=None):
    method = NAME_TO_METHOD[method_name]
    if ALLOW_MASK[method] is False:
        assert mask is None, f"Method {method_name} does not support masking!"
    threshold_function = THRESHOLD_FUNCTION[method]
    image, _, df_health_bars = read_image_path(image_path)
    #TODO jpg 3 channels, png 4 channels
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    h, w = template_gray.shape
    res = cv2.matchTemplate(image=image_gray, templ=template_gray, method=method, mask=mask)
    img_result = image.copy()
    loc = np.where(threshold_function(res, threshold))
    #TODO if nothing found
    suppr_loc = suppression(loc)
    matches_num = len(suppr_loc[0])
    for pt in zip(*suppr_loc[::-1]):
        cv2.rectangle(img_result, pt, (pt[0] + w, pt[1] + h), (255, 255, 255), 1)

    image_name = os.path.splitext(os.path.basename(image_path))[0]
    result_name = f'{image_name}_{method_name}_t_{threshold}_m_{matches_num}.jpg'
    result_path = os.path.join(RESULTS_DIR, result_name)
    if plot:
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.imshow(img_result)
        ax.set_title(f"Method={method_name}, threshold={threshold}, Loc={matches_num}")
        # if save:
        #     fig.savefig(result_path)
    if save:
        cv2.imwrite(result_path, img_result)


def get_template_paths(edge_name):
    template_path = os.path.join(TEMPLATES_DIR, f"{edge_name}_template.npy")
    mask_path = os.path.join(TEMPLATES_DIR, f"{edge_name}_mask.txt")
    return template_path, mask_path


def save_template(edge_name, template, mask):
    assert mask.shape == template.shape[:-1]
    template_path, mask_path = get_template_paths(edge_name)
    np.save(template_path, template)
    np.savetxt(mask_path, mask, fmt="%d")


def read_template(edge_name):
    template_path, mask_path = get_template_paths(edge_name)
    template = np.load(template_path)
    mask = np.loadtxt(mask_path, dtype=np.uint8)
    return template, mask


