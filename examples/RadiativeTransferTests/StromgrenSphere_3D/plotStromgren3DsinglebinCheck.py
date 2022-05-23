#!/usr/bin/env python3

# ----------------------------------------------------
# Stromgren 3D with grey approximation (single-frequency bin) and fixed temperature
# The test is identical to Test 1 in Iliev et al. 2006 doi:10.1111/j.1365-2966.2006.10775.x
# Analytic solution is described in Appendix C of SPHM1RT paper (https://arxiv.org/abs/2102.08404)
# Plot comparison of simulated neutral fraction with analytic solution
# ----------------------------------------------------

import swiftsimio
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
import sys
import stromgren_plotting_tools as spt


# Plot parameters
params = {
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "font.size": 14,
    "legend.fontsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.width": 1.5,
    "ytick.major.width": 1.5,
    "axes.linewidth": 1.5,
    "text.usetex": True,
    "figure.figsize": (5, 4),
    "figure.subplot.left": 0.045,
    "figure.subplot.right": 0.99,
    "figure.subplot.bottom": 0.05,
    "figure.subplot.top": 0.99,
    "figure.subplot.wspace": 0.15,
    "figure.subplot.hspace": 0.12,
    "lines.markersize": 1,
    "lines.linewidth": 2.0,
}
mpl.rcParams.update(params)
mpl.rc("font", **{"family": "sans-serif", "sans-serif": ["Times"]})

scatterplot_kwargs = {
    "alpha": 0.6,
    "s": 4,
    "marker": ".",
    "linewidth": 0.0,
    "facecolor": "blue",
}

# Read in cmdline arg: Are we plotting only one snapshot, or all?
plot_all = False
try:
    snapnr = int(sys.argv[1])
except IndexError:
    plot_all = True
    snapnr = -1

snapshot_base = "output_singlebin"


def plot_analytic_compare(filename):
    # Read in data first
    print("working on", filename)
    data = swiftsimio.load(filename)
    meta = data.metadata
    boxsize = meta.boxsize
    scheme = str(meta.subgrid_scheme["RT Scheme"].decode("utf-8"))

    xstar = data.stars.coordinates
    xpart = data.gas.coordinates
    dxp = xpart - xstar
    r = np.sqrt(np.sum(dxp ** 2, axis=1))

    imf = spt.get_imf(scheme, data)
    xHI = imf.HI / (imf.HI + imf.HII)

    r_ana, xn = spt.get_analytic_solution(data)
    plt.scatter(r, xHI, **scatterplot_kwargs)
    plt.plot(r_ana, xn)
    plt.ylabel("Neutral Fraction")
    xlabel_units_str = meta.boxsize.units.latex_representation()
    plt.xlabel("r [$" + xlabel_units_str + "$]")
    plt.yscale("log")
    plt.xlim([0, boxsize[0] / 2.0])
    plt.tight_layout()
    figname = filename[:-5]
    figname += "-Stromgren3Dsinglebin.png"
    plt.savefig(figname, dpi=200)
    plt.close()


if __name__ == "__main__":
    snaplist = spt.get_snapshot_list(snapshot_base, plot_all, snapnr)
    for f in snaplist:
        plot_analytic_compare(f)
