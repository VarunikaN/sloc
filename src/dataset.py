import os, glob, random
import errno
from dataclasses import dataclass
from functools import lru_cache
import logging
from PIL import Image
import torchvision
import atexit
import pydicom
import pandas as pd

class RSNASource:
    def __init__(self, base_path):
        self.base_path = base_path
        # Standard Kaggle RSNA paths
        self.image_dir = os.path.join(base_path, 'stage_2_train_images')
        self.csv_path = os.path.join(base_path, 'stage_2_train_labels.csv')

    @lru_cache(maxsize=None)
    def get_all_images(self):
        df = pd.read_csv(self.csv_path)
        # Drop duplicates as CSV has rows per bounding box; we just need patientId
        df = df.drop_duplicates(subset=['patientId'])
        
        images = {}
        for _, row in df.iterrows():
            p_id = row['patientId']
            path = os.path.join(self.image_dir, f"{p_id}.dcm")
            if os.path.exists(path):
                # Target 1 for Pneumonia, 0 for Normal
                target = int(row['Target'])
                images[p_id] = ImageInfo(path=path, name=p_id, target=target, desc="rsna")
        return images

def load_rsna_as_pil(path):
    """Helper to convert DICOM to RGB PIL Image for the ModelEnv."""
    ds = pydicom.dcmread(path)
    img_array = ds.pixel_array.astype(float)
    # Rescale to 0-255
    img_array = (np.maximum(img_array, 0) / img_array.max()) * 255.0
    # Convert to 3-channel RGB so ImageNet models don't crash
    return Image.fromarray(img_array.astype(np.uint8)).convert('RGB')
    
@dataclass
class ImageInfo:
    path : str
    name : str
    target: int
    desc : str = "unknown"


class ImagenetSource:

    def __init__(self, 
                 base_path, 
                 image_dir_ptrn,
                 targets_filename=None,
                 selection_filename=None):
        self.base_path = base_path
        self.image_dir_ptrn = image_dir_ptrn
        self.targets_path = (os.path.join(self.base_path, targets_filename) 
                             if targets_filename else None)
        self.selection_filename = selection_filename

    @lru_cache(maxsize=None)     
    def read_selection(self):
        logging.debug("loading selection")
        selection_path = os.path.join(self.base_path, self.selection_filename)
        with open(selection_path, "rt") as sf:
            return [self.get_image_name(x.strip()) for x in sf]

    @lru_cache(maxsize=None)     
    def get_all_images(self):
        logging.debug(f"imagenet base path: {self.base_path}")
        ptrn = os.path.join(self.base_path, self.image_dir_ptrn, "*.JPEG")
        all_images = glob.glob(ptrn)
        logging.debug(f"found {len(all_images)} images at {ptrn}")
        image_targets = self.get_image_targets()

        images = {}
        for path in all_images:
            image_name = self.get_image_name(path)
            images[image_name] = ImageInfo(
                path=path, 
                name=image_name,
                target=image_targets.get(image_name,0))

        if self.selection_filename:
            selection = self.read_selection()
            images = {name : img for name, img in images.items() if name in selection}

        return images

    @lru_cache(maxsize=None)
    def get_image_targets(self):
        rv = {}
        if not self.targets_path:
            return rv
        with open(self.targets_path, "rt") as tf:
            for line in tf:
                file_name, target = line.split()
                target = int(target)
                image_name = self.get_image_name(file_name)
                rv[image_name] = target
        return rv

    def get_image_name(self, path):
        image_file_name = os.path.basename(path)
        image_name = image_file_name[0:image_file_name.find(".")]
        return image_name    
