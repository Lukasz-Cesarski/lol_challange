import os
from glob import glob
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
from skimage import io
from lol_core import find_edges, find_healthbars, build_rectangles, EDGES


STARTER_DIR = "starter"
TASK_EXAMPLES_DIR = "task_examples"
HEALTH_BARS_DIR = "healthbars"

DATA_DIR = os.path.join(STARTER_DIR, TASK_EXAMPLES_DIR, "")
IMAGE_PATHS = sorted(glob(DATA_DIR + "**.jpg") + glob(DATA_DIR + "**.png"))
RESULTS_DIR = "results"

# HB - HEALTH BAR
HB_WIDTH = 138
HB_HEIGHT = 30
LEFT_SIZE = 7
TOP_SIZE = 7
RIGHT_SIZE = 7
BOT_SIZE = 15


COLORS = [
    (255, 0, 0, 255),
    (0, 255, 0, 255),
    (0, 0, 255, 255),
    (0, 255, 255, 255),
    (255, 0, 255, 255),
    (128, 128, 128, 255),
    (250, 110, 99, 255),
    (59, 85, 239, 255),
    (250, 99, 171, 255),
    (146, 102, 255, 255),
]


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

    sa = 0.5  # shade alpha
    for idx, (crop_x, crop_y, l, t, r, b) in enumerate(df_health_bars.values):
        cropped = image[max(0, crop_y):crop_y + HB_HEIGHT, max(crop_x, 0):crop_x + HB_WIDTH]
        c_height, c_width, _ = cropped.shape
        ax[idx].imshow(cropped)
        ax[idx].set_title(f"IDX={idx}, X={crop_x}, Y={crop_y}")
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

    x = np.vstack([correct_hash, ref_hash, other_hash])
    model.fit(x)

    correct_proj = model.transform(correct_hash)
    ref_proj = model.transform(ref_hash)
    other_proj = model.transform(other_hash)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(correct_proj[:, 0], correct_proj[:, 1], ".r", label="correct")
    ax.plot(ref_proj[:, 0], ref_proj[:, 1], ".b", label="reference")
    ax.plot(other_proj[:, 0], other_proj[:, 1], ".y", label="other")
    ax.legend()


def plot_match(image_path, edge_name, method_name, plot=True, save=True):
    suppr_loc, image, h, w, threshold = find_edges(image_path, edge_name, method_name)
    img_result = image.copy()
    matches_num = len(suppr_loc[0])
    for pt in zip(*suppr_loc[::-1]):
        cv2.rectangle(img_result, pt, (pt[0] + w, pt[1] + h), (255, 255, 255, 255), 1)

    image_name = os.path.splitext(os.path.basename(image_path))[0]
    result_name = f'{image_name}_{edge_name}_{method_name}_t_{threshold}_m_{matches_num}.jpg'
    result_path = os.path.join(RESULTS_DIR, result_name)
    if plot:
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.imshow(img_result)
        ax.set_title(f"Edge={edge_name}, Method={method_name}, threshold={threshold}, Loc={matches_num}")
    if save:
        cv2.imwrite(result_path, cv2.cvtColor(img_result, cv2.COLOR_RGB2BGR))


def plot_rectangles(image_path, method_name, plot=True, save=True):
    all_rectangles, edge_dimensions = find_healthbars(image_path, method_name)
    image = io.imread(image_path)
    rect_amount = len(all_rectangles)

    img_result = image.copy()
    for rectangle, color in zip(all_rectangles, COLORS):
        for point, (h, w) in zip(rectangle, edge_dimensions):
            if point is not None:
                y, x = point
                cv2.rectangle(img_result, (x, y), (x + w, y + h), color, -1)

    image_name = os.path.splitext(os.path.basename(image_path))[0]
    result_name = f'REC_{image_name}_{method_name}_rect_{rect_amount}.jpg'
    result_path = os.path.join(RESULTS_DIR, result_name)
    if plot:
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.imshow(img_result)
        ax.set_title(f"Rectangles={rect_amount}, Method={method_name}")
    if save:
        cv2.imwrite(result_path, cv2.cvtColor(img_result, cv2.COLOR_RGB2BGR))
