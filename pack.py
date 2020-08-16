"""
Packing script
"""

import os
from shutil import copyfile, make_archive, rmtree, copytree

from lol_core import TEMPLATES_DIR

build_dir = "build"
build_lol_dir = os.path.join(build_dir, "lol_core")
core_file_name = "lol_core.py"
req_file_name = "requirements.txt"
try:
    rmtree(build_dir)
except FileNotFoundError:
    pass
os.makedirs(build_lol_dir)
copyfile(core_file_name, os.path.join(build_lol_dir, core_file_name))
copyfile(req_file_name, os.path.join(build_lol_dir, req_file_name))
copytree(TEMPLATES_DIR, os.path.join(build_lol_dir, TEMPLATES_DIR))
make_archive(os.path.join(build_dir, "lol_core"), 'zip', build_lol_dir)
