import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image
import glob
import os
from scipy.signal import medfilt

def load_multiple_npys(path):
    """
    Load multiple .npy files from a directory

    Parameters
    ----------
    path : str
        The path to the directory containing the .npy files

    Returns
    -------
    numpy array
        A numpy array containing the concatenated contents of each .npy file
    """
    files = glob.glob(os.path.join(path, '*.npy'))
    arrays = [np.load(file) for file in files]
    return np.concatenate(arrays, axis=0)

def normalize_lidars(lidars):
    # Normalize every lidar data by dividing 20
    normalized_lidars = lidars.copy()
    normalized_lidars[normalized_lidars < 0] = 20.0
    normalized_lidars /= 20.0

    #* should i do a noise filtering here?
    # smoothed_data = medfilt(normalize_lidars, kernel_size=3)  # Noise Filtering
    return normalized_lidars

def bin_lidar_data(lidar_data, num_bins= 18):
    binneds = []
    for k in range(len(lidar_data)):
        bin_edges = np.linspace(0, len(lidar_data[k]), num_bins + 1)
        binned = [np.mean(lidar_data[k][int(bin_edges[i]):int(bin_edges[i+1])]) for i in range(num_bins)]
        binneds.append(binned)
    return np.array(binneds)


# # Load the saved LiDAR data
# path = "/home/o/Documents/donkeycar_rl/data/pack/generated_track_human/"

# lidars = load_multiple_npys(path)
# lidars = normalize_lidars(lidars)
# lidars = bin_lidar_data(lidars)

