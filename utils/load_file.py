import os
import numpy as np
from PIL import Image
from joblib import Parallel, delayed
from numpy.ma.core import asarray
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from utils.process_single_image import process_single_image
from utils.save_as_npy import save_as_npy


def load_file(raw_path, processed_path, x_filename, y_filename):

    x_npy_file_path = os.path.join(processed_path, x_filename)
    y_npy_file_path = os.path.join(processed_path, y_filename)

    if os.path.isfile(x_npy_file_path) and os.path.isfile(y_npy_file_path):
        print(f"Found {x_filename} and {y_filename} files, loading them")
        x = np.load(x_npy_file_path)
        y = np.load(y_npy_file_path)
        print("Data loaded successfully")
        return x, y
    else:

        print(f"{x_filename} and {y_filename} not found, started processing data")

        image_path_and_label = []

        for label, subfolder in enumerate(os.listdir(raw_path)):

            subfolder_path = os.path.join(raw_path, subfolder)

            if os.path.isdir(subfolder_path):

                for i, file_name in enumerate(os.listdir(subfolder_path)):

                    #if i % 5 == 0:
                    file_path = os.path.join(subfolder_path, file_name)

                    if os.path.isfile(file_path):

                        image_path_and_label.append((file_path, label))

        results = Parallel(n_jobs=-1)(
            delayed(process_single_image)(file_path, label)
            for file_path, label in image_path_and_label
        )

        x = []
        y = []
        for np_img, label in results:
            if np_img is not None:
                x.append(np_img)
                y.append(label)

        x_np = np.array(x)
        y_np = np.array(y)

        scaler = StandardScaler()
        x_np = scaler.fit_transform(x_np)

        pca = PCA(n_components=1000)
        x_np = pca.fit_transform(x_np)

        save_as_npy(processed_path, x_np, y_np, x_filename, y_filename)

        return x_np, y_np