import numpy as np
import uproot
import os
import glob
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
from multiprocessing import Pool

def process_root_file(file_path, output_dir, tree_name="photon_sim", bins=(100,100)):
    """
    Process a ROOT file and save the extracted data as an HDF5 file.
    
    Parameters:
    - file_path: str, path to the ROOT file.
    - output_dir: str, directory to save the HDF5 file.
    - tree_name: str, name of the tree inside the ROOT file.
    - bins: tuple, number of bins in (x, y, z) dimensions.
    
    Returns:
    - None (saves an HDF5 file)
    """
    file_name = os.path.basename(file_path).replace(".root", ".h5")
    output_path = os.path.join(output_dir, file_name)
    
    with uproot.open(f"{file_path}:{tree_name}") as file:
        x = file["x"].array(library="np")
        # y = file["y"].array(library="np")
        z = file["z"].array(library="np")+250.
        dE = file["dE"].array(library="np")
        evt = file["EventID"].array(library="np")
        E0 = file["primaryE"].array(library="np")

    unique_events = np.unique(evt)

    with h5py.File(output_path, "w") as hf:
        for event_id in unique_events:
            mask = evt == event_id
            H = plt.hist2d(x[mask], z[mask], 
                                              bins=bins, 
                                              weights=dE[mask], 
                                              range=[[-200, 200], [0, 400]])
            H3 = H[0]
            edges_x = H[1]
            edges_z = H[2]
            
            # Compute bin centers
            x_centers = (edges_x[:-1] + edges_x[1:]) / 2
            z_centers = (edges_z[:-1] + edges_z[1:]) / 2

            # Flatten grid and values
            X, Z = np.meshgrid(x_centers, z_centers, indexing="ij")
            X, Z, values = X.flatten(), Z.flatten(), H3.flatten()
            E_event = np.full_like(X, np.unique(E0[mask])[0])

            event_data = np.stack([X, Z, values, E_event], axis=-1)

            # Save each event as a dataset
            hf.create_dataset(f"event_{event_id}", data=event_data, compression="gzip")

    print(f"Saved {output_path}")

def process_all_files(input_folder, output_folder, num_workers=4):
    """
    Process all ROOT files in the folder and save them as HDF5 files in parallel.
    
    Parameters:
    - input_folder: str, path to folder containing ROOT files.
    - output_folder: str, directory to save processed HDF5 files.
    - num_workers: int, number of parallel processes.
    
    Returns:
    - None (saves HDF5 files)
    """
    os.makedirs(output_folder, exist_ok=True)
    file_list = glob.glob(os.path.join(input_folder, "*.root"))

    args = [(file_path, output_folder) for file_path in file_list]

    with Pool(num_workers) as pool:
        list(tqdm(pool.starmap(process_root_file, args), total=len(file_list)))

# Example usage
input_folder = "/media/disk/g4_showers/unif"
output_folder = "/media/disk/g4_showers/2D"
process_all_files(input_folder, output_folder, num_workers=6)