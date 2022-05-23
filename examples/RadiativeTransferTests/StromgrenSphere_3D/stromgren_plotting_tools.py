import os
import numpy as np
import unyt
import copy
from scipy.integrate import odeint


# analytic solution
def neutralfraction3d(rfunc, nH, sigma, alphaB, dNinj, rini):
    def fn(xn, rn):
        """this is the rhs of the ODE to integrate, i.e. dx/drn=fn(x,r)=x*(1-x)/(1+x)*(2/rn+x)"""
        return xn * (1.0 - xn) / (1.0 + xn) * (2.0 / rn + xn)

    xn0 = nH * alphaB * 4.0 * np.pi / sigma / dNinj * rini * rini
    rnounit = rfunc * nH * sigma
    xn = odeint(fn, xn0, rnounit)
    return xn


def get_analytic_solution(data):
    meta = data.metadata
    rho = data.gas.densities
    rini_value = 0.1
    r_ana = np.linspace(rini_value, 10.0, 100) * unyt.kpc
    rini = rini_value * unyt.kpc
    nH = np.mean(rho.to("g/cm**3") / unyt.proton_mass)
    sigma_cross = trim_paramstr(
        meta.parameters["SPHM1RT:sigma_cross"].decode("utf-8")
    ) * unyt.unyt_array(1.0, "cm**2")
    sigma = sigma_cross[0]
    alphaB = trim_paramstr(
        meta.parameters["SPHM1RT:alphaB"].decode("utf-8")
    ) * unyt.unyt_array(1.0, "cm**3/s")
    units = data.units
    unit_l_in_cgs = units.length.in_cgs()
    unit_v_in_cgs = (units.length / units.time).in_cgs()
    unit_m_in_cgs = units.mass.in_cgs()
    star_emission_rates = (
        trim_paramstr(meta.parameters["SPHM1RT:star_emission_rates"].decode("utf-8"))
        * unit_m_in_cgs
        * unit_v_in_cgs ** 3
        / unit_l_in_cgs
    )
    ionizing_photon_energy_erg = (
        trim_paramstr(
            meta.parameters["SPHM1RT:ionizing_photon_energy_erg"].decode("utf-8")
        )
        * unyt.erg
    )
    dNinj = star_emission_rates[1] / ionizing_photon_energy_erg[0]
    xn = neutralfraction3d(r_ana, nH, sigma, alphaB, dNinj, rini)
    return r_ana, xn


def get_TT1Dsolution():
    TT1D_runit = 5.4 * unyt.kpc  # kpc
    data = np.loadtxt("data/xTT1D_Stromgren100Myr.txt", delimiter=",")
    rtt1dlist = data[:, 0] * TT1D_runit
    xtt1dlist = 10 ** data[:, 1]

    data = np.loadtxt("data/TTT1D_Stromgren100Myr.txt", delimiter=",")
    rTtt1dlist = data[:, 0] * TT1D_runit
    Ttt1dlist = 10 ** data[:, 1] * unyt.K

    outdict = {}
    outdict["rtt1dlist"] = rtt1dlist
    outdict["xtt1dlist"] = xtt1dlist
    outdict["rTtt1dlist"] = rTtt1dlist
    outdict["Ttt1dlist"] = Ttt1dlist
    return outdict


def get_TT1Dsolution_HHe():
    TT1D_runit = 5.4 * unyt.kpc  # kpc
    data = np.loadtxt("data/xHITT1D_Stromgren100Myr_HHe.txt", delimiter=",")
    rHItt1dlist = data[:, 0] * TT1D_runit
    xHItt1dlist = 10 ** data[:, 1]

    data = np.loadtxt("data/xHIITT1D_Stromgren100Myr_HHe.txt", delimiter=",")
    rHIItt1dlist = data[:, 0] * TT1D_runit
    xHIItt1dlist = 10 ** data[:, 1]

    data = np.loadtxt("data/xHeITT1D_Stromgren100Myr_HHe.txt", delimiter=",")
    rHeItt1dlist = data[:, 0] * TT1D_runit
    xHeItt1dlist = 10 ** data[:, 1]

    data = np.loadtxt("data/xHeIITT1D_Stromgren100Myr_HHe.txt", delimiter=",")
    rHeIItt1dlist = data[:, 0] * TT1D_runit
    xHeIItt1dlist = 10 ** data[:, 1]

    data = np.loadtxt("data/xHeIIITT1D_Stromgren100Myr_HHe.txt", delimiter=",")
    rHeIIItt1dlist = data[:, 0] * TT1D_runit
    xHeIIItt1dlist = 10 ** data[:, 1]

    data = np.loadtxt("data/TTT1D_Stromgren100Myr_HHe.txt", delimiter=",")
    rTtt1dlist = data[:, 0] * TT1D_runit
    Ttt1dlist = 10 ** data[:, 1] * unyt.K

    outdict = {}
    outdict["rHItt1dlist"] = rHItt1dlist
    outdict["xHItt1dlist"] = xHItt1dlist
    outdict["rHIItt1dlist"] = rHIItt1dlist
    outdict["xHIItt1dlist"] = xHIItt1dlist
    outdict["rHeItt1dlist"] = rHeItt1dlist
    outdict["xHeItt1dlist"] = xHeItt1dlist
    outdict["rHeIItt1dlist"] = rHeIItt1dlist
    outdict["xHeIItt1dlist"] = xHeIItt1dlist
    outdict["rHeIIItt1dlist"] = rHeIIItt1dlist
    outdict["xHeIIItt1dlist"] = xHeIIItt1dlist
    outdict["rTtt1dlist"] = rTtt1dlist
    outdict["Ttt1dlist"] = Ttt1dlist
    return outdict


def mean_molecular_weight(XH0, XHp, XHe0, XHep, XHepp):
    """
    Determines the mean molecular weight for given 
    mass fractions of
        hydrogen:   XH0
        H+:         XHp
        He:         XHe0
        He+:        XHep
        He++:       XHepp

    returns:
        mu: mean molecular weight [in atomic mass units]
        NOTE: to get the actual mean mass, you still need
        to multiply it by m_u, as is tradition in the formulae
    """

    # 1/mu = sum_j X_j / A_j * (1 + E_j)
    # A_H    = 1, E_H    = 0
    # A_Hp   = 1, E_Hp   = 1
    # A_He   = 4, E_He   = 0
    # A_Hep  = 4, E_Hep  = 1
    # A_Hepp = 4, E_Hepp = 2
    one_over_mu = XH0 + 2 * XHp + 0.25 * XHe0 + 0.5 * XHep + 0.75 * XHepp

    return 1.0 / one_over_mu


def gas_temperature(u, mu, gamma):
    """
    Compute the gas temperature given the specific internal 
    energy u and the mean molecular weight mu
    """

    # Using u = 1 / (gamma - 1) * p / rho
    #   and p = N/V * kT = rho / (mu * m_u) * kT

    T = u * (gamma - 1) * mu * unyt.atomic_mass_unit / unyt.boltzmann_constant

    return T.to("K")


def get_snapshot_list(snapshot_basename="output", plot_all=True, snapnr=0):
    """
    Find the snapshot(s) that are to be plotted 
    and return their names as list
    """

    snaplist = []

    if plot_all:
        dirlist = os.listdir()
        for f in dirlist:
            if f.startswith(snapshot_basename) and f.endswith("hdf5"):
                snaplist.append(f)

        snaplist = sorted(snaplist)

    else:
        fname = snapshot_basename + "_" + str(snapnr).zfill(4) + ".hdf5"
        if not os.path.exists(fname):
            print("Didn't find file", fname)
            quit(1)
        snaplist.append(fname)

    return snaplist


def get_imf(scheme, data):
    """
    Get the ion mass fraction (imf) according to the scheme.
    return a class with ion mass function for species X, 
    including HI, HII, HeI, HeII, HeIII:
    The ion mass function can be accessed through: imf.X
    The unit is in m_X/m_tot, where m_X is the mass in species X
    and m_tot is the total gas mass.
    """
    if scheme.startswith("GEAR M1closure"):
        imf = data.gas.ion_mass_fractions
    elif scheme.startswith("SPH M1closure"):
        # atomic mass
        mamu = {"e": 0.0, "HI": 1.0, "HII": 1.0, "HeI": 4.0, "HeII": 4.0, "HeIII": 4.0}
        mass_fraction_hydrogen = data.gas.rt_element_mass_fractions.hydrogen
        imf = copy.deepcopy(data.gas.rt_species_abundances)
        named_columns = data.gas.rt_species_abundances.named_columns
        for column in named_columns:
            # abundance is in n_X/n_H unit. We convert it to mass fraction by multipling mass fraction of H
            mass_fraction = (
                getattr(data.gas.rt_species_abundances, column)
                * mass_fraction_hydrogen
                * mamu[column]
            )
            setattr(imf, column, mass_fraction)
    return imf


def get_abundances(scheme, data):
    """
    Get the species abundance according to the scheme
    return a class with normalized number densities for abunance X, 
    including HI, HII, HeI, HeII, HeIII:
    The ion mass function can be accessed through: sA.X
    The unit is in n_X/n_H, where n_X is the number density of species X
    and n_H is the number density of hydrogen.
    """
    if scheme.startswith("GEAR M1closure"):
        # atomic mass
        mamu = {"e": 0.0, "HI": 1.0, "HII": 1.0, "HeI": 4.0, "HeII": 4.0, "HeIII": 4.0}
        sA = copy.deepcopy(data.gas.ion_mass_fractions)
        mass_fraction_hydrogen = (
            data.gas.ion_mass_fractions.HI + data.gas.ion_mass_fractions.HII
        )
        # abundance is in n_X/n_H unit. We convert mass fraction to abundance by dividing mass fraction of H
        abundance = data.gas.ion_mass_fractions.HI / mass_fraction_hydrogen / mamu["HI"]
        setattr(sA, "HI", abundance)
        abundance = (
            data.gas.ion_mass_fractions.HII / mass_fraction_hydrogen / mamu["HII"]
        )
        setattr(sA, "HII", abundance)
        abundance = (
            data.gas.ion_mass_fractions.HeI / mass_fraction_hydrogen / mamu["HeI"]
        )
        setattr(sA, "HeI", abundance)
        abundance = (
            data.gas.ion_mass_fractions.HeII / mass_fraction_hydrogen / mamu["HeII"]
        )
        setattr(sA, "HeII", abundance)
        abundance = (
            data.gas.ion_mass_fractions.HeIII / mass_fraction_hydrogen / mamu["HeIII"]
        )
        setattr(sA, "HeIII", abundance)
    elif scheme.startswith("SPH M1closure"):
        sA = data.gas.rt_species_abundances
    return sA


def trim_paramstr(paramstr):
    # clean string up
    paramstr = paramstr.strip()
    if paramstr.startswith("["):
        paramstr = paramstr[1:]
    if paramstr.endswith("]"):
        paramstr = paramstr[:-1]

    # transform string values to floats with unyts
    params = paramstr.split(",")
    paramtrimmed = []
    for er in params:
        paramtrimmed.append(float(er))
    return paramtrimmed
