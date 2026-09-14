import numpy as np
import scipy.linalg as spla

# import EVC matrices
datadir = 'evcData'
He4_const = np.loadtxt(f'./{datadir}/He4_Nmax18_hw36_NNLOsat_Nsub10/const')
He4_cD    = np.loadtxt(f'./{datadir}/He4_Nmax18_hw36_NNLOsat_Nsub10/cD')
He4_cE    = np.loadtxt(f'./{datadir}/He4_Nmax18_hw36_NNLOsat_Nsub10/cE')
He4_norm  = np.loadtxt(f'./{datadir}/He4_Nmax18_hw36_NNLOsat_Nsub10/norm_matrix')
He4_r2    = np.loadtxt(f'./{datadir}/He4_Nmax18_hw36_NNLOsat_Nsub10/rp2_He4_Nmax18_hw36')

He3_const = np.loadtxt(f'./{datadir}/He3_Nmax40_hw36_NNLOsat_Nsub10/const')
He3_cD    = np.loadtxt(f'./{datadir}/He3_Nmax40_hw36_NNLOsat_Nsub10/cD')
He3_cE    = np.loadtxt(f'./{datadir}/He3_Nmax40_hw36_NNLOsat_Nsub10/cE')
He3_norm  = np.loadtxt(f'./{datadir}/He3_Nmax40_hw36_NNLOsat_Nsub10/norm_matrix')

H3_const = np.loadtxt(f'./{datadir}/H3_Nmax40_hw36_NNLOsat_Nsub10/const')
H3_cD    = np.loadtxt(f'./{datadir}/H3_Nmax40_hw36_NNLOsat_Nsub10/cD')
H3_cE    = np.loadtxt(f'./{datadir}/H3_Nmax40_hw36_NNLOsat_Nsub10/cE')
H3_norm  = np.loadtxt(f'./{datadir}/H3_Nmax40_hw36_NNLOsat_Nsub10/norm_matrix')

E1A_const = np.loadtxt(f'./{datadir}/E1A_Nmax40_hw36_Nsub10/const')
E1A_cD    = np.loadtxt(f'./{datadir}/E1A_Nmax40_hw36_Nsub10/cD')

#NNLOsat LECs:

cD_NNLOsat = +0.81680589148271 
cE_NNLOsat = -0.03957471270351

def fewnucleonEmulator(cD,cE):
    """
    Computes observables for A=3,4 nuclear systems.
    
    This method uses reduced-order models of the full solutions to the Schrodinger equation.
    These models are based on eigenvector continuation and correspond to the solution
    of generalized eigenvalue problems for small matrices.
    
    Parameters
    ----------
    cD : float
        Low-energy constant for the two-nucleon contact plus one-pion exchange diagram.
    cE : float
        Low-energy constant for the three-nucleon contact diagram.
 
    Returns
    -------
    tuple of 7 floats
        (E4He, Rp4He, Rch4He, E3He, E3H, E1A3H, fT3H)
        in units of
        MeV    fm      fm       MeV    MeV   1    sec
        where the observables are:
        E = ground-state energy
        Rp = point-proton radius
        Rch = Charge radius
        E1A = GT matrix element
        fT = comparative halflife
    """
    
    H = He4_const + cD*He4_cD + cE*He4_cE
    N = He4_norm
    
    eigvals, eigvec = spla.eigh(H,N)
    E_He4 = eigvals[0]  
    
    r2 = eigvec[:,0].T@He4_r2@eigvec[:,0]
    r2_ch = r2 + 0.8775**2 -0.1149 +0.033
    Rp_He4 = np.sqrt(r2)
    Rch_He4 = np.sqrt(r2_ch)
    
    H = He3_const + cD*He3_cD + cE*He3_cE
    N = He3_norm
    
    eigvals, eigvec_bra = spla.eigh(H,N)
    E_He3 = eigvals[0]
    
    H = H3_const + cD*H3_cD + cE*H3_cE
    N = H3_norm
    
    eigvals, eigvec_ket = spla.eigh(H,N)
    E_H3 = eigvals[0]
    
    E1A = np.abs(eigvec_bra[:,0].T@(E1A_const + cD*E1A_cD)@eigvec_ket[:,0])
    
    # fTexp = 1129.6 +- 3 s
    # => E1A_emp = 0.6848(11)
    fT = 6146.6/( (1-0.0013) + 3*np.pi*1.00529*E1A**2)
    
    #units MeV    fm      fm       MeV    MeV   1    sec
    return (E_He4, Rp_He4, Rch_He4, E_He3, E_H3, E1A, fT)

if __name__=="__main__":
    print('Few-nucleon observables with NNLOsat computed with EVC emulator:')
    print(fewnucleonEmulator(cD_NNLOsat,cE_NNLOsat))
