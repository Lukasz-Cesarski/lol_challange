from glob import glob
import os
import matplotlib.pyplot as plt
from skimage import io
import pandas as pd

STARTER_DIR = "starter"
TASK_EXAMPLES_DIR = "task_examples"
HEALTHBARS_DIR = "healthbars"

DATA_DIR = os.path.join(STARTER_DIR, TASK_EXAMPLES_DIR, "")
IMAGE_PATHS = sorted(glob(DATA_DIR + "**.jpg") + glob(DATA_DIR + "**.png"))

# HB = HEALTH BAR
HB_WIDTH = 138
HB_HEIGHT = 30


def read_image_path(image_path):
    image = io.imread(image_path)
    health_bars_save_path = os.path.join(HEALTHBARS_DIR, TASK_EXAMPLES_DIR, os.path.splitext(os.path.basename(image_path))[0] + ".csv")
    if os.path.isfile(health_bars_save_path):
        df_health_bars = pd.read_csv(health_bars_save_path)
    else:
        df_health_bars = None
    return image, health_bars_save_path, df_health_bars


def plot_screen(image, df_health_bars):
    fig, ax = plt.subplots(1,1, figsize=(16,9))
    ax.imshow(image)
    for x, y, _, _, _, _ in df_health_bars.values:
        ax.add_patch(plt.Rectangle((x, y), HB_WIDTH, HB_HEIGHT, edgecolor='black', lw=2, facecolor='none'))


def plot_health_bars(image, df_health_bars):
    fig, ax = plt.subplots(len(df_health_bars), 1, figsize=(16, 20))
    nickname_margin = 0
    # nickname_margin = 15
    colors = ["r", "b"]

    lo = 3   # border line offset
    lw = 10  # border line width
    for idx, (crop_x, crop_y, l, t, r, b) in enumerate(df_health_bars.values):
        cropped = image[max(0, crop_y - nickname_margin):crop_y + HB_HEIGHT, max(crop_x, 0):crop_x + HB_WIDTH]
        ax[idx].imshow(cropped)
        ax[idx].set_title(f"IDX={idx}, X={crop_x}, Y={crop_y}")
        ax[idx].vlines([-lo], 0, cropped.shape[0], linewidth=lw, colors=colors[l])
        ax[idx].hlines([-lo], 0, cropped.shape[1], linewidth=lw, colors=colors[t])
        ax[idx].vlines([cropped.shape[1] + lo], 0, cropped.shape[0], linewidth=lw, colors=colors[r])
        ax[idx].hlines([cropped.shape[0] + lo], 0, cropped.shape[1], linewidth=lw, colors=colors[b])
