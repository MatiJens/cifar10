import os.path

import numpy as np

def save_as_npy(path, x, y, x_filename : str, y_filename : str):

    if not os.path.exists(path):
        os.makedirs(path)

    x_path = os.path.join(path, x_filename)
    y_path = os.path.join(path, y_filename)

    np.save(x_path, x)
    np.save(y_path, y)

    print(f"Data saved as {x_filename} and {y_filename}")