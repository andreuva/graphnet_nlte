import numpy as np

# Physical Constants in CGS units
kB_CGS = 1.380649e-16       # Boltzmann constant in erg/K
h_CGS = 6.62607015e-27      # Planck constant in erg*s
m_e_CGS = 9.10938356e-28    # Electron mass in g
c_CGS = 2.99792458e10       # Speed of light cm/s
m_u_CGS = 1.66053892e-24    # Atomic mass unit g
q_e_CGS = 4.80320425e-10    # electron charge
eV_CGS = 1.60217663e-12     # 1 eV in erg

def get_partition_functions(Z, T):
    """
    Returns partition functions (UI, UII, UIII) for element Z at temperature T.
    Exactly mirrors the partition function definitions in chemeq.py but is fully
    vectorized to handle NumPy arrays of any shape as well as scalars.

    Supports:
        Z = 1 (H), Z = 11 (Na), Z = 12 (Mg), Z = 20 (Ca)
    """
    is_scalar = np.isscalar(T)
    T = np.atleast_1d(T)
    
    x = np.log(5040.0 / T)
    y = 1e-3 * T
    
    if Z == 1: # H
        UI = np.ones_like(T)
        UII = np.ones_like(T) * 2.0
        UII = np.where(T > 1.3e4, 1.51 + 3.8e-5 * T, UII)
        UII = np.where(T > 1.62e4, 11.41 + T * (-1.1428e-3 + T * 3.52e-8), UII)
        UIII = np.ones_like(T)
    elif Z == 11: # Na
        UI = np.maximum(2.0, 1.72 + 9.3e-5 * T)
        UI = np.where(T > 5400.0, -0.83 + 5.66e-4 * T, UI)
        UI = np.where(T > 8.5e3, 4.5568 + T * (-1.2415e-3 + T * 1.3861e-7), UI)
        UII = np.ones_like(T)
        UIII = 5.69 + 5.69e-6 * T
    elif Z == 12: # Mg
        UI = 1.0 + np.exp(-4.027262 - x * (6.173172 + x * (2.889176 + x * (2.393895 + 0.784131 * x))))
        UI = np.where(T > 8e3, 2.757 + T * (-7.8909e-4 + T * 7.4531e-8), UI)
        UII = 2.0 + np.exp(-7.721172 - x * (7.600678 + x * (1.966097 + 0.212417 * x)))
        UII = np.where(T > 2e4, 7.1041 + T * (-1.0817e-3 + T * 4.7841e-8), UII)
        UIII = np.ones_like(T)
    elif Z == 20: # Ca
        UI = 1.0 + np.exp(-1.731273 - x * (5.004556 + x * (1.645456 + x * (1.326861 + 0.508553 * x))))
        UII = 2.0 + np.exp(-1.582112 - x * (3.996089 + x * (1.890737 + 0.539672 * x)))
        UIII = np.ones_like(T)
    else:
        raise ValueError(f"Partition functions for Z={Z} not implemented in lte_utils.py")
        
    if is_scalar:
        return UI.item(), UII.item(), UIII.item()
    else:
        return UI, UII, UIII

def compute_lte_populations(atom, temp, ne_bg, nh):
    """
    Calculates LTE level populations for a multi-level atom using the Saha-Boltzmann 
    equations and partition functions. Fully vectorized to support NumPy arrays 
    of any shape for temperature, electron density, and hydrogen density.

    Args:
        atom: The atomic model object (e.g., from atoms.py) containing:
              - Z (int): atomic number
              - abundance (float): elemental abundance
              - levels (list): atomic levels with .energy, .g, .ionization
        temp (np.ndarray): Temperature array (K) of arbitrary shape.
        ne_bg (np.ndarray): Electron density array (cm^-3) of the same shape.
        nh (np.ndarray): Hydrogen density array (cm^-3) of the same shape.

    Returns:
        np.ndarray: An array of shape (temp.shape) + (num_levels,) containing
                    the population of each level at each atmospheric point.
    """
    levels = atom.levels
    num_levels = len(levels)
    energies = np.array([l.energy for l in levels])     # In ergs
    gs = np.array([l.g for l in levels])                # Statistical weights
    stages = np.array([l.ionization for l in levels])   # Ionization stage (0-indexed or 1-indexed offset)

    T = np.asarray(temp, dtype=np.float64)
    ne = np.asarray(ne_bg, dtype=np.float64)
    nh = np.asarray(nh, dtype=np.float64)

    original_shape = T.shape

    # Handle negative / zero temperature and electron densities safely
    if np.any(T <= 0.0) or np.any(ne <= 0.0):
        print("WARNING: Zero or negative temperature/electron density in compute_lte_populations. Clipping.")
        T = np.clip(T, 1e-10, None)
        ne = np.clip(ne, 1e-10, None)

    kT = kB_CGS * T
    saha_const = ((2.0 * np.pi * m_e_CGS * kB_CGS) / (h_CGS**2))**1.5

    # Total number density of this element
    N_total = atom.abundance * nh

    # Saha factor: (2 / N_e) * (2 * pi * m_e * k_B * T / h^2)^1.5
    Phi = (2.0 / ne) * saha_const * (T ** 1.5)

    # Find unique ionizations and their ground state energies
    stages_unique = np.unique(stages)
    E_ground = {}
    for s in stages_unique:
        E_ground[s] = np.min(energies[stages == s])

    s_ref = np.min(stages_unique)
    E_ref = E_ground[s_ref]

    # Calculate partition functions
    UI, UII, UIII = get_partition_functions(atom.Z, T)
    if atom.Z == 1:
        # For Hydrogen, UI is H-, UII is H I (stage 0), UIII is H II (stage 1)
        U_t = {0: UII, 1: UIII}
    else:
        # For all other atoms, UI is stage 0, UII is stage 1, UIII is stage 2
        U_t = {0: UI, 1: UII, 2: UIII}

    # Calculate fractional abundance of each stage relative to the reference stage (s_ref)
    f = {}
    for s in stages_unique:
        dE = E_ground[s] - E_ref
        f[s] = (U_t[s] / U_t[s_ref]) * (Phi ** (s - s_ref)) * np.exp(-dE / kT)

    sum_f = sum(f.values())

    # Total absolute population of reference stage
    N_ref = N_total / sum_f

    # Total absolute population of each stage
    N_stage = {s: N_ref * f[s] for s in stages_unique}

    # Initialize the output array for populations
    populations = np.zeros(original_shape + (num_levels,))

    # Calculate the LTE population for each individual level (Boltzmann equation)
    for i in range(num_levels):
        s = stages[i]
        E_level_rel = energies[i] - E_ground[s]
        populations[..., i] = N_stage[s] * (gs[i] / U_t[s]) * np.exp(-E_level_rel / kT)

    return populations
