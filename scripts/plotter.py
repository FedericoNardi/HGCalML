import ROOT
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D


def load_event_from_root(root_file_path, event_index=0):
    """
    Loads a single event's centroids and BIB densities (dE) from a ROOT file.

    Parameters:
    - root_file_path: path to the ROOT file
    - event_index: index of the event to load (default: 0)

    Returns:
    - centroids: np.ndarray of shape (N, 3)
    - bib_density: np.ndarray of shape (N,)
    """
    root_file = ROOT.TFile.Open(root_file_path)
    tree = root_file.Get("Events")

    if event_index >= tree.GetEntries():
        raise IndexError(f"Event index {event_index} out of range. File contains {tree.GetEntries()} entries.")

    tree.GetEntry(event_index)

    n = getattr(tree, "n")
    x = np.array(list(getattr(tree, "x"))[:n])
    y = np.array(list(getattr(tree, "y"))[:n])
    z = np.array(list(getattr(tree, "z"))[:n])
    dE = np.array(list(getattr(tree, "dE"))[:n])

    centroids = np.stack([x, y, z], axis=-1)
    return centroids, dE


def plot_first_10_events(root_file_path, bins=12, save_path="img/debug/first10_events.jpg"):
    """
    Plots the first 10 events in a ROOT file, each in its own subplot.
    """
    os.makedirs("img/debug", exist_ok=True)
    root_file = ROOT.TFile.Open(root_file_path)
    tree = root_file.Get("Events")
    n_events = min(10, tree.GetEntries())

    fig, axes = plt.subplots(2, 5, figsize=(30, 12))
    axes = axes.flatten()

    for i in range(n_events):
        tree.GetEntry(i)
        n = getattr(tree, "n")
        x = np.array(list(getattr(tree, "x"))[:n])
        y = np.array(list(getattr(tree, "y"))[:n])
        z = np.array(list(getattr(tree, "z"))[:n])
        dE = np.array(list(getattr(tree, "dE"))[:n])
        sf = np.array(list(getattr(tree, "signal_fraction"))[:n])

        log_rho = np.log10(dE + 1e-9)

        ax = axes[i]
        h = ax.hist2d(x, z, bins=bins, weights=dE*sf, cmap='cividis',
                      norm=plt.matplotlib.colors.LogNorm())
        ax.set_title(f"Event {i}")
        ax.set_xlabel("X")
        ax.set_ylabel("Z")

    plt.tight_layout()
    plt.savefig(save_path, dpi=250)
    plt.close()


def plot_event(centroids, bib_density, bins=12, event_id=1, save_path="img/debug/generated_vis"):
    """
    Plot projections of BIB energy density across centroids.
    """
    os.makedirs("img/debug", exist_ok=True)

    x, y, z = centroids[:, 0], centroids[:, 1], centroids[:, 2]
    bib_density = bib_density.flatten()
    log_rho = np.log10(bib_density + 1e-9)

    fig = plt.figure(figsize=(24, 6))

    # 3D scatter
    ax = fig.add_subplot(141, projection='3d')
    sc = ax.scatter(x, y, z, c=log_rho, cmap='cividis', s=8, alpha=0.8)
    fig.colorbar(sc, ax=ax, label='log10(BIB density)')
    ax.set_title("BIB Density (3D)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # X-Z projection
    ax2 = fig.add_subplot(142)
    h_xz = ax2.hist2d(x, z, bins=bins, weights=bib_density, cmap='cividis',
                      norm=plt.matplotlib.colors.LogNorm())
    fig.colorbar(h_xz[3], ax=ax2, label='BIB density [X-Z]')
    ax2.set_title("X-Z Projection")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Z")

    # Y-Z projection
    ax3 = fig.add_subplot(143)
    h_yz = ax3.hist2d(y, z, bins=bins, weights=bib_density, cmap='cividis',
                      norm=plt.matplotlib.colors.LogNorm())
    fig.colorbar(h_yz[3], ax=ax3, label='BIB density [Y-Z]')
    ax3.set_title("Y-Z Projection")
    ax3.set_xlabel("Y")
    ax3.set_ylabel("Z")

    # X-Y projection
    ax4 = fig.add_subplot(144)
    h_xy = ax4.hist2d(x, y, bins=bins, weights=bib_density, cmap='cividis',
                      norm=plt.matplotlib.colors.LogNorm())
    fig.colorbar(h_xy[3], ax=ax4, label='BIB density [X-Y]')
    ax4.set_title("X-Y Projection")
    ax4.set_xlabel("X")
    ax4.set_ylabel("Y")

    plt.suptitle(f"BIB Energy Density (Event {event_id})", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{save_path}_{event_id:03d}.jpg", dpi=250)
    plt.close()

if __name__ == "__main__":
    # centroids, bib_density = load_event_from_root("output_root_pyroot/events_002.root")
    # plot_event(centroids, bib_density, event_id=1)
    plot_first_10_events("/media/disk/pyroot_tuples_smear/events_002.root")
    