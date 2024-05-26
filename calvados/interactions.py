import numpy as np
from openmm import openmm, unit

def genParamsDH(temp,ionic):
    """ Debye-Huckel parameters. """

    kT = 8.3145*temp*1e-3
    # Calculate the prefactor for the Yukawa potential
    fepsw = lambda T : 5321/T+233.76-0.9297*T+0.1417*1e-2*T*T-0.8292*1e-6*T**3
    epsw = fepsw(temp)
    lB = 1.6021766**2/(4*np.pi*8.854188*epsw)*6.022*1000/kT
    eps_yu = np.sqrt(lB*kT)
    # Calculate the inverse of the Debye length
    k_yu = np.sqrt(8*np.pi*lB*ionic*6.022/10)
    return eps_yu, k_yu

def init_bonded_interactions():
    """ Define bonded interactions. """

    # harmonic bonds
    hb = openmm.HarmonicBondForce()
    hb.setUsesPeriodicBoundaryConditions(True)

    return hb

def init_ah_interactions(eps_lj,cutoff_lj):
    """ Define Ashbaugh-Hatch interactions. """

    # intermolecular interactions
    energy_expression = 'select(step(r-2^(1/6)*s),4*eps*l*((s/r)^12-(s/r)^6-shift),4*eps*((s/r)^12-(s/r)^6-l*shift)+eps*(1-l))'
    ah = openmm.CustomNonbondedForce(energy_expression+'; s=0.5*(s1+s2); l=0.5*(l1+l2); shift=(0.5*(s1+s2)/rc)^12-(0.5*(s1+s2)/rc)^6')

    ah.addGlobalParameter('eps',eps_lj*unit.kilojoules_per_mole)
    ah.addGlobalParameter('rc',float(cutoff_lj)*unit.nanometer)
    ah.addPerParticleParameter('s')
    ah.addPerParticleParameter('l')

    ah.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
    ah.setCutoffDistance(cutoff_lj*unit.nanometer)
    ah.setForceGroup(0)

    print('rc',cutoff_lj*unit.nanometer)
    # print('pbc ah: ', ah.usesPeriodicBoundaryConditions())
    return ah

def init_yu_interactions(k_yu):
    """ Define Yukawa interactions. """

    yu = openmm.CustomNonbondedForce('q*(exp(-kappa*r)/r-shift); q=q1*q2')
    yu.addGlobalParameter('kappa',k_yu/unit.nanometer)
    yu.addGlobalParameter('shift',np.exp(-k_yu*4.0)/4.0/unit.nanometer)
    yu.addPerParticleParameter('q')
    
    yu.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
    yu.setCutoffDistance(4.*unit.nanometer)
    yu.setForceGroup(1)

    return yu

    # print('pbc yu: ', yu.usesPeriodicBoundaryConditions())

def init_interactions(eps_lj,cutoff_lj,k_yu):
    """ Define protein interaction expressions (without restraints). """
    
    hb = init_bonded_interactions()
    ah = init_ah_interactions(eps_lj, cutoff_lj)
    yu = init_yu_interactions(k_yu)

    return hb, ah, yu

def init_restraints(restraint_type):
    """ Initialize restraints. """

    if restraint_type == 'harmonic':
        cs = openmm.HarmonicBondForce()
    if restraint_type == 'go':
        go_expr = 'k*(5*(s/r)^12-6*(s/r)^10)'
        cs = openmm.CustomBondForce(go_expr+'; s=s; k=k')#; shift=(0.5*(s)/rc)^12-(0.5*(s)/rc)^6')
        cs.addPerBondParameter('s')
        cs.addPerBondParameter('k')
    cs.setUsesPeriodicBoundaryConditions(True)
    return cs

def init_scaled_LJ(eps_lj,cutoff_lj):
    """ Initialize restraints. """

    energy_expression = 'select(step(r-2^(1/6)*s),n*4*eps*l*((s/r)^12-(s/r)^6-shift),n*4*eps*((s/r)^12-(s/r)^6-l*shift)+n*eps*(1-l))'
    scLJ = openmm.CustomBondForce(energy_expression+'; shift=(s/rc)^12-(s/rc)^6')
    scLJ.addGlobalParameter('eps',eps_lj*unit.kilojoules_per_mole)
    scLJ.addGlobalParameter('rc',float(cutoff_lj)*unit.nanometer)
    scLJ.addPerBondParameter('s')
    scLJ.addPerBondParameter('l')
    scLJ.addPerBondParameter('n')
    scLJ.setUsesPeriodicBoundaryConditions(True)
    return scLJ

def init_scaled_YU(k_yu):
    """ Initialize restraints. """

    scYU = openmm.CustomBondForce('n*q*(exp(-kappa*r)/r-shift)')
    scYU.addGlobalParameter('kappa',k_yu/unit.nanometer)
    scYU.addGlobalParameter('shift',np.exp(-k_yu*4.0)/4.0/unit.nanometer)
    scYU.addPerBondParameter('q')
    scYU.addPerBondParameter('n')
    scYU.setUsesPeriodicBoundaryConditions(True)
    return scYU

def init_eq_restraints(box,k):
    """ Define restraints towards box center in z direction. """

    mindim = np.amin(box)
    rcent_expr = 'k*abs(periodicdistance(x,y,z,x,y,z0))'
    rcent = openmm.CustomExternalForce(rcent_expr)
    rcent.addGlobalParameter('k',k*unit.kilojoules_per_mole/unit.nanometer)
    rcent.addGlobalParameter('z0',box[2]/2.*unit.nanometer) # center of box in z
    # rcent.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
    # rcent.setCutoffDistance(mindim/2.*unit.nanometer)
    return rcent

def add_single_restraint(
        cs, restraint_type: str,
        dij: float, k: float,
        i: int, j: int):
    """ Add single harmonic or Go restraint. """

    if restraint_type == 'harmonic':
        cs.addBond(
                i,j, dij*unit.nanometer,
                k*unit.kilojoules_per_mole/(unit.nanometer**2))
    elif restraint_type == 'go':
        cs.addBond(
                i,j, [dij*unit.nanometer,
                k*unit.kilojoules_per_mole])
    restr_pair = [i+1, j+1, dij, k] # 1-based
    return cs, restr_pair

def add_scaled_lj(scLJ, i, j, offset, comp):
    """ Add downscaled LJ interaction. """

    s = 0.5 * (comp.sigmas[i] + comp.sigmas[j])
    l = 0.5 * (comp.lambdas[i] + comp.lambdas[j])
    scLJ.addBond(i+offset,j+offset, [s*unit.nanometer, l*unit.dimensionless, comp.bondscale[i,j]*unit.dimensionless])
    scaled_pair = [i+offset+1, j+offset+1, s, l, comp.bondscale[i,j]] # 1-based
    return scLJ, scaled_pair

def add_scaled_yu(scYU, i, j, offset, comp, eps_yu):
    """ Add downsscaled YU interaction. """

    qij = comp.qs[i] * comp.qs[j] * eps_yu * eps_yu * unit.nanometer*unit.kilojoules_per_mole *unit.nanometer * unit.kilojoules_per_mole
    scYU.addBond(i+offset, j+offset, [qij, comp.bondscale[i,j]*unit.dimensionless])
    scaled_pair = [i+offset+1, j+offset+1, comp.bondscale[i,j]] # 1-based
    return scYU, scaled_pair

def add_exclusion(force, i: int, j: int):
    """ Add exclusions to a list of openMM forces """
    force.addExclusion(i,j)
    return force