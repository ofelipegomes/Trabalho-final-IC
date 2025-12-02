import os
import numpy as np
import cv2

    
def imread_rgb(path):
    img = cv2.imread(path)
    if img is None:
        raise IOError(f"Cannot read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def load_dataset(root_dir, subfolders=('Train', 'Valid'), resize=None):
    data = {}
    for sub in subfolders:
        folder = os.path.join(root_dir, sub)
        if not os.path.isdir(folder):
            data[sub] = ([], [])
            continue
        # check for class subfolders
        classes = sorted([d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))])
        X_paths = []
        y = []
        if len(classes) > 0:
            for c in classes:
                class_folder = os.path.join(folder, c)
                for fname in os.listdir(class_folder):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                        X_paths.append(os.path.join(class_folder, fname))
                        y.append(c)
        else:
            for fname in os.listdir(folder):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                    X_paths.append(os.path.join(folder, fname))
                    base = os.path.splitext(fname)[0]
                    label = ''
                    for ch in base:
                        if ch.isalpha():
                            label += ch
                        else:
                            break
                    if label == '':
                        label = 'unknown'
                    y.append(label)
        data[sub] = (X_paths, np.array(y))
    return data


def load_images(paths, resize=None):
    imgs = []
    for p in paths:
        img = imread_rgb(p)
        if resize is not None:
            import skimage.transform as tf
            img = tf.resize(img, resize, preserve_range=True)
        imgs.append(img)
    return imgs
