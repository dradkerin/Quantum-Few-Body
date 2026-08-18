"""
Two-body stuff. Thus should contain the wavefunction, normalisation,
energy spectrum etc.
"""

import numpy as np
import mpmath as mpmath
import scipy.special as special
import scipy.integrate as integrate
import scipy.stats as stats
from scipy.optimize import fsolve
from scipy.optimize import root
import matplotlib.pyplot as plt
import time as ti
from matplotlib import rc
from multiprocessing import Pool
rc('font',**{'family':'DejaVu Sans','serif':['Computer Modern']})
rc('text',usetex=True)

import sys
import os

"""
This is the code associated with the two-body
quantum quench dynamics calculations of two
zero-range intaracting particles in a 
spherical harmonic trap.

We have implicitely set the SHO lengthscale
to be 1 (a_rel=1).

"""

def IntWaveFuncNoNorm(v,r):
    """
    Gives the non-normalised interacting two-body wavefunciton
    for some energy and at some separation 
    
    Parameters
    -----------------
    v : a real float
        v is the energy pseudo quantum number (\nu) make sure this is never a
        positive integer because gamma(-integer) is infinite    
    r : a real positive float
        r is the separation between the two particles ((r_1-r_2)/2)

    Returns
    -----------------
    X : a real float
        the value of the non-normalised wavefunction for the specific input
    """

    X=special.gamma(-v)*np.exp(-0.5*r**2)*special.hyperu(-v,1.5,r**2)
    return X

def IntNorm(v):
    """
    Gives the normalisation of the interacting two-body wavefunction 

    note: there would be an factor of a_rel**-1.5  but we have set 
    a_rel=1 implicitly everywhere


    Parameters
    -----------------
    v : a real float
        v is the energy pseudo quantum number (\nu) make sure this is never a
        positive integer because gamma(-int) is infinite    
    u : a real float
        u is the reduced mass m_1*m_2/(m_1+m_2)
    r : a real float
        r is the separation between the two particles ((r_1-r_2)/2)

    Returns
    -----------------
    X : a real float
        the value of the normalisation constant
    """

    Z=np.pi*special.gamma(1-v)*(special.digamma(-v-0.5)-special.digamma(-v))/(v*special.gamma(-v-0.5))
    X=(2*np.pi*Z)**(-0.5)

    return X

def IntWaveFunc(v,r):
    """
    Gives the normalised two-body wavefunciton for some energy
    (v) and at some separation (r)

    Parameters
    -----------------
    v : a real float
        v is the energy pseudo quantum number (\nu) make sure this is never a
        positive integer because gamma(-int) is infinite    
    u : a real float
        u is the reduced mass (m_{1}*m_{2}/(m_{1}+m_{2}())
    r : a real positive float
        r is the separation between the two particles ((r_{1}-r_{2})/2)

    Returns
    -----------------
    X : a float
        the value of the normalised wavefunction
    """

    #Supresses any warnings
    old_stdout = sys.stdout # backup current stdout
    sys.stdout = open(os.devnull, "w")

    X=IntNorm(v)*IntWaveFuncNoNorm(v,r)

    sys.stdout = old_stdout # reset old stdout





    

    return X

def NIWaveFunc(n,r):
    """
    The normalised noninteracting SHO wavefunction for l=0.

    We only care about the l=0 case for physical reasons.

    The energy associated with this wavefunction is 
    (2n+1.5) hbar omega
    
    Parameters
    ---------------
    n : a non-negative int
        the principle quantum number. 
        primary index of the associated laguerre polynomial
    r: a real positive float
        the radial input. Note that this code is unitless so this function is a
        function of r/a where a=1.

    Norm : a float
        the normalisation of the radial part. doesn't include a 4*pi because
        that's handled by the spherical harmonic parts that we don't care about

    Returns
    ---------------
    X : a float
        The value of the function
    """
    Norm=np.sqrt((1/(4*np.pi)**1.5)*(2**(n+3))*special.factorial(n)/special.factorial2(2*n+1))

    X=np.exp(-0.5*r**2)*special.assoc_laguerre(r**2,n,0.5)
    X=Norm*X

    return X

def Energies(Nmax,a):
    """
    the lowest "Nmax" energies of the interacting two body wavefunction for some
    s-wave scattering length a. In units of a=a_s/a_rel.

    we're implicitly doing l=0 here. l!=0 means the wavefunction is non-interacting
    and we don't care about that here

    Parameters
    ---------------------
    Nmax: a positive int
        the number of energy levels we calculate up to
    a: a float
        the s-wave scattering length in units a=a_s/a_rel

    Returns
    ---------------------
    Energies: a 1 by Nmax array of real floats
        an array of the energies of the two body system. First entry is the
        lowest energy

    """

    En=np.zeros(Nmax)

    #interacting case
    if a != 0:
        funcBinom = lambda v : (2/np.sqrt(np.pi))*special.binom(-v-3/2,-v-1)**-1-(a)**-1


        for j in range(Nmax):
            #Energies[j]=fsolve(funcBinom,j-0.45)#decent at j~171 but get some misses at low

            #print(fsolve(funcBinom,j-0.01))
            En[j]=fsolve(funcBinom,j-0.01)[0]#good for low energy but some dodginess at j=171

            #I think the binomial function just has some dodginess around that j=171
            #not sure how to get around it
            

        En=2*En+3/2
    
    if a==0:
        for j in range(Nmax):
            En[j]=2*j+3/2

   

    return En


def WavefFuncOverlaps(Variables):
    """
    This calculates the square overlap of two interacting wavefunctions.
    It is designed to be a part of parallelised code.
    
    This can't handle unitary limits, only for finite non-zero scattering
    lengths

    Parameters
    ---------------------
    Variables: a 1 x 3 array of real floats
        Variables[0] is Vf, the energy pseudo-quantum number for the 
        final state 
        Variables[1] is V0, the energy pseudo-quantum number for the 
        initial state
        Variables[2] is Z0, is a part of the normalisation of the
        initial state it is faster to calculate it once before passing
        it to this function rather than have this function calculate it
        each time it runs 

    Parameters
    ---------------------
    Square overlap: a positive real float
        the square overlap of the two interacting wavefunction terms.

    """
    Vf=Variables[0]
    V0=Variables[1]
    Z0=Variables[2]

    Zj=np.pi*(special.digamma(-Vf-0.5)-special.digamma(-Vf))*special.poch(-Vf-1/2,3/2)/Vf

    Overlap=np.sqrt(np.pi)*(2*V0*Vf*np.sqrt(Z0*Zj))**(-1)\
    *mpmath.hyp3f2(1.5,-Vf,-V0,1-Vf,1-V0,1)

    #SquareOverlap=abs(Overlap)**2

    #print("Vf=",Vf)

    return Overlap


#post-quench energy expectation
def IntIntExpectE(Nmax,InitialA,InitialJ,FinalA):
    """
    Plots the expectation of the post quench energy of the system as a function
    of the number of terms in the expansion. Quench is finite non-zero a_s to
    finite non-zero a_s

    Parameters
    ---------------------
    Nmax: an int
        the number of terms we go up to in the expansion
    InitialA: a float
        the s-wave scattering length of the pre-quench system
    InitialJ: an int
        the principle quantum number of the initial system. 0 ground, 1 first
        excited etc.
    FinalA: a float
        the s-wave scattering length of the post-quench system


    Returns
    ---------------------
    SumSize: a 1 x Nmax array of positive ints
        an array of [1,2,3,...,Nmax] 
    CumulEnergies: a 1 x Nmax array of real floats
        <E> as a function of number of terms
    """

    V0=0.5*(Energies(InitialJ+1,InitialA)[InitialJ]-1.5)
    Z0=np.pi*(special.digamma(-V0-0.5)-special.digamma(-V0))*special.poch(-V0-1/2,3/2)/V0
    Vf=0.5*(Energies(Nmax,FinalA)-1.5)

    Variables=np.zeros([Nmax,3])
    Variables[:,0]=Vf
    Variables[:,1]=V0
    Variables[:,2]=Z0

    
    Terms=np.zeros(Nmax)
    p=Pool()
    Terms=p.map(WavefFuncOverlaps,Variables)
    Terms=np.array(Terms)**2

    for j in range(Nmax):
        Terms[j]=(2*Vf[j]+3/2)*Terms[j]


    CumulEnergies=np.cumsum(Terms)
    SumSize=np.linspace(1,Nmax,Nmax)


    return SumSize,CumulEnergies

def IntUnitExpectE(Nmax,InitialA,InitialJ):
    """
    Plots the expectation of the post quench energy of the system as a function
    of the number of terms in the expansion. Quench is from finite non-zero a_{s}
    to unitarity.
    
    Parameters
    ---------------------
    Nmax: an int
        the number of terms we go up to in the expansion
    InitialA: a float
        the s-wave scattering length of the pre-quench system
    InitialJ: an int
        the principle quantum number of the initial system. 0 ground, 1 first
        excited etc.

    Returns
    ---------------------
    SumSize: a 1 x Nmax array of positive ints
        an array of [1,2,3,...,Nmax] 
    CumulEnergies: a 1 x Nmax array of real floats
        <E> as a function of number of terms


    """

    Terms=np.zeros(Nmax)
    NormCheck=np.zeros(Nmax)

    V0=0.5*(Energies(InitialJ+1,InitialA)-1.5)[InitialJ]
    Z0=np.pi*special.gamma(1-V0)*(special.digamma(-V0-0.5)-special.digamma(-V0))/(V0*special.gamma(-V0-0.5))

    for j in range(Nmax):
        Zj=(np.pi**1.5)*special.binom(j-1/2,j)**-1
        OverlapJ=np.sqrt(np.pi)*(2*V0*(j-1/2)*np.sqrt(Z0*Zj))**(-1)\
        *mpmath.hyp3f2(1.5,1/2-j,-V0,3/2-j,1-V0,1)

        Terms[j]=(2*j+1/2)*OverlapJ**2
        NormCheck[j]=OverlapJ**2

        #print(j,"/",Nmax)
        #print("j=",j)
        #print("Zj=",Zj)
        #print("Overlap=",OverlapJ)

    CumulEnergies=np.cumsum(Terms)
    SumSize=np.linspace(1,Nmax,Nmax)

    return SumSize, CumulEnergies

def IntNIExpectE(Nmax,InitialA,InitialJ):
    """
    Plots the expectation of the post quench energy of the system as a function
    of the number of terms in the expansion. Quench is finite non-zero a_s to
    non-interacting

    THIS IS DIVERGENT AS EXPECTED

    Parameters
    ---------------------
    Nmax: an int
        the number of terms we go up to in the expansion
    InitialA: a float
        the s-wave scattering length of the pre-quench system
    InitialJ: an int
        the principle quantum number of the initial system. 0 ground, 1 first
        excited etc.

    Returns
    ---------------------
    SumSize: a 1 x Nmax array of positive ints
        an array of [1,2,3,...,Nmax] 
    CumulEnergies: a 1 x Nmax array of real floats
        <E> as a function of number of terms

    """

    Terms=np.zeros(Nmax)
    NormCheck=np.zeros(Nmax)

    V0=0.5*(Energies(InitialJ+1,InitialA)[InitialJ]-1.5)
    Z0=np.pi*special.gamma(1-V0)*(special.digamma(-V0-0.5)-special.digamma(-V0))/(V0*special.gamma(-V0-0.5))

    for n in range(Nmax):
        OverlapN=(np.sqrt(n+1/2)/((n-V0)*np.sqrt(Z0)))*np.sqrt(np.sqrt(np.pi)*special.binom(n-1/2,n))

        C=OverlapN**2
        Terms[n]=(2*n+3/2)*C
        NormCheck[n]=C

    if Terms[Nmax-1]==Terms[Nmax-2]:
        print("WARNING: ENERGIES NOT CHANGING EVEN THOUGH <E> IS DIVERGENT TO TOWARDS NI QUENCH")


    CumulEnergies=np.cumsum(Terms)
    SumSize=np.linspace(1,Nmax,Nmax)


    return SumSize, CumulEnergies

def NIUnitExpectE(Nmax,InitialN):
    """
    Plots the expectation of the post quench energy of the system as a function
    of the number of terms in the expansion. Quench is non-itneracring
    to unitarity

    Parameters
    ---------------------
    Nmax: an int
        the number of terms we go up to in the expansion
    InitialN: an int
        the principle quantum number of the initial system. 0 ground, 1 first
        excited etc.

    Returns
    ---------------------
    SumSize: a 1 x Nmax array of positive ints
        an array of [1,2,3,...,Nmax] 
    CumulEnergies: a 1 x Nmax array of real floats
        <E> as a function of number of terms
    """

    Terms=np.zeros(Nmax)
    NormCheck=np.zeros(Nmax)

    for j in range(Nmax):
        C=np.sqrt((InitialN+1/2)*special.binom(InitialN-1/2,InitialN)*special.binom(j-1/2,j))/(np.sqrt(np.pi)*(InitialN+1/2-j))
        Terms[j]=(2*j+1/2)*C**2
        NormCheck[j]=C**2

    CumulEnergies=np.cumsum(Terms)
    SumSize=np.linspace(1,Nmax,Nmax)


    return SumSize, CumulEnergies

def NIIntExpectE(Nmax,InitialN,FinalA):
    """
    Plots the expectation of the post quench energy of the system as a function
    of the number of terms in the expansion. Quench is non-interacting to
    finite non-zero a_s

    Parameters
    ---------------------
    Nmax: an int
        the number of terms we go up to in the expansion
    InitialN: an int
        the principle quantum number of the initial system. 0 ground, 1 first
        excited etc.
    Final A: a float
        the s-wave scattering length of the final system

    Returns
    ---------------------
    SumSize: a 1 x Nmax array of positive ints
        an array of [1,2,3,...,Nmax] 
    CumulEnergies: a 1 x Nmax array of real floats
        <E> as a function of number of terms

    """

    Terms=np.zeros(Nmax)
    NormCheck=np.zeros(Nmax)

    Vf=0.5*(Energies(Nmax,FinalA)-1.5)

    for j in range(Nmax):
        Zj=np.pi*(special.digamma(-Vf[j]-0.5)-special.digamma(-Vf[j]))*special.poch(-Vf[j]-1/2,3/2)/Vf[j]

        OverlapJ=np.sqrt(InitialN+1/2)*np.sqrt(np.sqrt(np.pi)*special.binom(InitialN-1/2,InitialN))/((InitialN-Vf[j])*np.sqrt(Zj))
        Terms[j]=(2*Vf[j]+3/2)*OverlapJ**2
        NormCheck[j]=OverlapJ**2

        #if j%100==0:
        #    print(j,Vf[j])

    CumulEnergies=np.cumsum(Terms)
    SumSize=np.linspace(1,Nmax,Nmax)

    return SumSize, CumulEnergies

def UnitNIExpectE(Nmax,InitialJ):
    """
    Plots the expectation of the post quench energy of the system as a function
    of the number of terms in the expansion. Quench is from unitarity to 
    non-interacting.

    THIS IS DIVERGENT AS EXPECTED


    Parameters
    -------------------------------
    Nmax: an int
        the number of terms we go up to in the expansion
    InitialJ: an int
        the principle quantum number of the initial system. 0 ground, 1 first
        excited etc.

    Returns:
    -------------------------------
    SumSize: 1xNmax array of ints
        a list of numbers 1 to Nmax. To be used to plot <E> against
    CumulEnergies: a 1xNmax array of floats
        The <E> evaluated for a number of terms equal to it's index in the array
        plus one. i.e. CumulEnergies[N] is <E> evaluated with N+1 terms
    """

    Terms=np.zeros(Nmax)
    
    #the commented out code has can't handle Nmax>~179 because of the gamma
    #functions. The new code is the same thing just expressed in a diff form
    for n in range(Nmax):
        #C=np.sqrt((n+0.5)*special.gamma(n+0.5)*special.gamma(InitialJ+0.5))/(np.pi*(n-InitialJ+0.5)*np.sqrt(special.gamma(InitialJ+1)*special.gamma(n+1)))
        #Terms[n]=(2*n+3/2)*C**2
        #NormCheck[n]=C**2
        C=np.sqrt((n+1/2)*special.binom(n-1/2,n)*special.binom(InitialJ-1/2,InitialJ))/(np.sqrt(np.pi)*(n-InitialJ+1/2))
        Terms[n]=(2*n+3/2)*C**2
        

    CumulEnergies=np.cumsum(Terms)
    SumSize=np.linspace(1,Nmax,Nmax)

    



    return SumSize, CumulEnergies

def UnitIntExpectE(Nmax,InitialJ,FinalA):
    """
    Plots the expectation of the post quench energy of the system as a function
    of the number of terms in the expansion. Quench is unitarity to finite 
    non-zero a_


    Parameters
    -------------------------------
    Nmax: an int
        the number of terms we go up to in the expansion
    InitialJ: an int
        the principle quantum number of the initial system. 0 ground, 1 first
        excited etc.
    FinalA: a float
        the s-wave scattering length of the post-quench system

    Returns:
    -------------------------------
    SumSize: 1xNmax array of ints
        a list of numbers 1 to Nmax. To be used to plot <E> against
    CumulEnergies: a 1xNmax array of floats
        The <E> evaluated for a number of terms equal to it's index in the array
        plus one. i.e. CumulEnergies[N] is <E> evaluated with N+1 terms
    """

    Terms=np.zeros(Nmax)
    NormCheck=np.zeros(Nmax)

    V0=InitialJ-1/2
    Vf=0.5*(Energies(Nmax,FinalA)-1.5)


    Z0=(np.pi**1.5)*special.binom(InitialJ-1/2,InitialJ)**(-1)

    for j in range(Nmax):

        Zj=np.pi*(special.digamma(-Vf[j]-0.5)-special.digamma(-Vf[j]))*special.poch(-Vf[j]-1/2,3/2)/Vf[j]

        OverlapJ=np.sqrt(np.pi)*(2*V0*Vf[j]*np.sqrt(Z0*Zj))**(-1)\
        *mpmath.hyp3f2(1.5,-Vf[j],-V0,1-Vf[j],1-V0,1)

        Terms[j]=(2*Vf[j]+3/2)*OverlapJ**2
        NormCheck[j]=OverlapJ**2

        #if j%100==0:
            #print(j,Vf[j])

    if Terms[Nmax-1]==Terms[Nmax-2]:
        print("WARNING: ENERGIES NOT CHANGING EVEN THOUGH <E> IS DIVERGENT TO TOWARDS NI QUENCH")
    CumulEnergies=np.cumsum(Terms)
    SumSize=np.linspace(1,Nmax,Nmax)

  
    return SumSize,CumulEnergies

#a single function we can call for general <E> calculations
def QuenchExpectE(Nmax,InitialA,InitialJ,FinalA):
    """
    This is a single function we can call for all our
    <E> quenching needs. Depending on what Initial and FinalA
    are the function chooses the appropriate function to call
    
    
    Parameters
    ---------------------
    Nmax: a positive int
        the number of terms included in the calculation
    InitialA: a real float or a string
        initial interaction strength
        if it is a string it should be "inf" for infinty, i.e.
        the unitary limit
    InitialJ: a non-negative int
        exictation of the initial state. 0 is ground state,
        1 is first excited etc.
    FinalA: a real float or a string
        final interaction strength
        if it is a string it should be "inf" for infinty, i.e.
        the unitary limit

    Returns
    ---------------------
    SumSize: a 1 x Nmax array of positive ints
        an array of [1,2,3,...,Nmax] 
    
    CumulEnergies: a 1 x Nmax array of real floats
        <E> as a function of number of terms
    """


    #UnitNI
    if InitialA=="inf" and FinalA==0:
        [SumSize,CumulEnergies]=UnitNIExpectE(Nmax,InitialJ)
        
    #NIUnit
    if InitialA==0 and FinalA=="inf":
        [SumSize,CumulEnergies]=NIUnitExpectE(Nmax,InitialJ)
    
    #IntInt
    if InitialA!=0 and InitialA!="inf" and FinalA!=0 and FinalA!="inf" and InitialA!=FinalA:
        [SumSize,CumulEnergies]=IntIntExpectE(Nmax,InitialA,InitialJ,FinalA)
    
    #IntNI
    if InitialA!=0 and InitialA!="inf" and FinalA==0:
        [SumSize,CumulEnergies]=IntNIExpectE(Nmax,InitialA,InitialJ)

    #NIInt
    if InitialA==0 and FinalA!=0 and FinalA!="inf":
        [SumSize,CumulEnergies]=NIIntExpectE(Nmax,InitialJ,FinalA)
       
    #IntUnit
    if InitialA!=0 and InitialA!="inf" and FinalA=="inf":
        [SumSize,CumulEnergies]=IntUnitExpectE(Nmax,InitialA,InitialJ)
    
    #UnitInt
    if InitialA=="inf" and FinalA!=0 and FinalA!="inf":
        [SumSize,CumulEnergies]=UnitIntExpectE(Nmax,InitialJ,FinalA)

    #UnitUnit
    if InitialA=="inf" and FinalA=="inf":
        print("Unitary to Unitary quench makes no sense")
        SumSize=np.linspace(1,Nmax,Nmax)
        CumulEnergies=np.zeros(Nmax)
        CumulEnergies[:]=2*InitialJ+0.5

    #NINI
    if InitialA==0 and FinalA==0:
        print("NI to NI quench makes no sense")
        SumSize=np.linspace(1,Nmax,Nmax)
        CumulEnergies=np.zeros(Nmax)
        CumulEnergies[:]=2*InitialJ+1.5

    #Int to same Int
    if InitialA==FinalA and FinalA!=0 and FinalA!="inf":
        print(str(InitialA)+" to "+str(FinalA)+" quench makes no sense")
        SumSize=np.linspace(1,Nmax,Nmax)
        CumulEnergies=np.zeros(Nmax)
        CumulEnergies[:]=Energies(InitialJ+1,InitialA)[InitialJ]
      
        

    return SumSize,CumulEnergies


#Ramsey signal
def IntIntRamsey(Nmax,InitialA,InitialJ,FinalA,Tmax,Spacing):
    """
    Gives the two-body Ramsey signal over time 
    for a quench between two non-zero finite scattering lengths     


    Parameters
    ------------------
    Nmax : an integer
        the number of terms in the sum we calculate up to
    InitialA: a real float 
        initial interaction strength
    InitialJ: a non-negative int
        exictation of the initial state. 0 is ground state,
        1 is first excited etc.
    FinalA: a real float
        final interaction strength
    Tmax : a positive float
        calculate Ramsey Signal from time=0 until time=Tmax
    spacing : a small positive float
        the timestep size
        

    Returns
    -------------------
    S : a 1 x int(Tmax/spacing) array
        the Ramsey Signal as a function of time
    Tomega : a 1 x int(Tmax/spacing) array
        the time
    """

    Tomega=np.linspace(0,Tmax,int(Tmax/Spacing))
    S=np.zeros([int(Tmax/Spacing)],dtype=complex)

    Exponentials=np.zeros(Nmax,dtype=complex)

    V0=0.5*(Energies(InitialJ+1,InitialA)[InitialJ]-1.5)
    Z0=np.pi*(special.digamma(-V0-0.5)-special.digamma(-V0))*special.poch(-V0-1/2,3/2)/V0
    Vf=0.5*(Energies(Nmax,FinalA)-1.5)

    Variables=np.zeros([Nmax,3])
    Variables[:,0]=Vf
    Variables[:,1]=V0
    Variables[:,2]=Z0

    
    Coeffs=np.zeros(Nmax)
    p=Pool()
    Coeffs=p.map(WavefFuncOverlaps,Variables)
    Coeffs=np.array(Coeffs)**2

    Exponentials=np.zeros(Nmax,dtype=complex)
    for counter1 in range(int(Tmax/Spacing)):
        for counter2 in range(Nmax):
            Exponentials[counter2]=np.exp(-2j*(Vf[counter2]-V0)*np.pi*Tomega[counter1])


        S[counter1]=np.matmul(Coeffs,Exponentials)
    

    return Tomega, S

def IntUnitRamsey(Nmax,InitialA,InitialJ,Tmax,Spacing):
    """
    Gives the two-body Ramsey signal over time for a quench
    from some finite non-zero a_s to unitarity 

    Parameters
    ------------------
    Nmax : an integer
        the number of terms in the sum we calculate up to
    InitialA: a real float 
        initial interaction strength
    InitialJ: a non-negative int
        exictation of the initial state. 0 is ground state,
        1 is first excited etc.
    Tmax : a positive float
        calculate Ramsey Signal from time=0 until time=Tmax
    Spacing : a small positive float
        the timestep size
        

    Returns
    -------------------
    S : a 1 x int(Tmax/spacing) array
        the Ramsey Signal as a function of time
    Tomega : a 1 x int(Tmax/spacing) array
        the time


    """
    Tomega=np.linspace(0,Tmax,int(Tmax/Spacing))
    S=np.zeros([int(Tmax/Spacing)],dtype=complex)

    Coeffs=np.zeros(Nmax)
    Exponentials=np.zeros(Nmax,dtype=complex)

    VInitial=0.5*(Energies(InitialJ+1,InitialA)[InitialJ]-1.5)

    Z0=np.pi*special.gamma(1-VInitial)*(special.digamma(-VInitial-0.5)-special.digamma(-VInitial))\
        /(VInitial*special.gamma(-VInitial-0.5))
    for j in range(Nmax):
        Zj=(np.pi**1.5)*special.binom(j-1/2,j)**-1
        Overlap=np.sqrt(np.pi)*(2*VInitial*(j-1/2)*np.sqrt(Z0*Zj))**(-1)\
        *mpmath.hyp3f2(1.5,1/2-j,-VInitial,3/2-j,1-VInitial,1)
        Coeffs[j]=Overlap**2

    for counter1 in range(int(Tmax/Spacing)):
        for counter2 in range(Nmax):
            Exponentials[counter2]=np.exp(-2j*(counter2-0.5-VInitial)*np.pi*Tomega[counter1])


        S[counter1]=np.matmul(Coeffs,Exponentials)

    return Tomega, S

def IntNIRamsey(Nmax,InitialA,InitialJ,Tmax,Spacing):

    """
    Gives the two-body Ramsey signal over time for a quench
    from some finite non-zero a_s to unitarity 

    Parameters
    ------------------
    Nmax : an integer
        the number of terms in the sum we calculate up to
    InitialA: a real float 
        initial interaction strength
    InitialJ: a non-negative int
        exictation of the initial state. 0 is ground state,
        1 is first excited etc.
    Tmax : a positive float
        calculate Ramsey Signal from time=0 until time=Tmax
    Spacing : a small positive float
        the timestep size
        

    Returns
    -------------------
    S : a 1 x int(Tmax/spacing) array
        the Ramsey Signal as a function of time
    Tomega : a 1 x int(Tmax/spacing) array
        the time


    """
    Tomega=np.linspace(0,Tmax,int(Tmax/Spacing))
    S=np.zeros([int(Tmax/Spacing)],dtype=complex)

    Coeffs=np.zeros(Nmax)


    VInitial=0.5*(Energies(InitialJ+1,InitialA)[InitialJ]-1.5)

    V0=0.5*(Energies(InitialJ+1,InitialA)[InitialJ]-1.5)
    Z0=np.pi*special.gamma(1-V0)*(special.digamma(-V0-0.5)-special.digamma(-V0))/(V0*special.gamma(-V0-0.5))

    for n in range(Nmax):
        Overlap=(np.sqrt(n+1/2)/((n-V0)*np.sqrt(Z0)))*np.sqrt(np.sqrt(np.pi)*special.binom(n-1/2,n))

        Coeffs[n]=Overlap**2

    Exponentials=np.zeros(Nmax,dtype=complex)
    for counter1 in range(int(Tmax/Spacing)):
        for counter2 in range(Nmax):
            Exponentials[counter2]=np.exp(-2j*(counter2-0.5-VInitial)*np.pi*Tomega[counter1])


        S[counter1]=np.matmul(Coeffs,Exponentials)

    return Tomega, S

def NIUnitRamsey(Nmax,InitialJ,Tmax,Spacing):
    """
    Gives the two-body Ramsey signal over time 
    for a quench between from the non-interacting
    limit to the unitary limit

    Parameters
    ------------------
    Nmax : an integer
        the number of terms in the sum we calculate up to
    InitialJ: a non-negative int
        exictation of the initial state. 0 is ground state,
        1 is first excited etc.
    Tmax : a positive float
        calculate Ramsey Signal from time=0 until time=Tmax
    spacing : a small positive float
        the timestep size
        

    Returns
    -------------------
    S : a 1 x int(Tmax/spacing) array
        the Ramsey Signal as a function of time
    Tomega : a 1 x int(Tmax/spacing) array
        the time
    """

    Tomega=np.linspace(0,Tmax,int(Tmax/Spacing))

    S=np.zeros([int(Tmax/Spacing)],dtype=complex)

    Coeffs=np.zeros(Nmax)

    
    for counter2 in range(Nmax):
        C=np.sqrt((InitialJ+0.5)*special.gamma(InitialJ+0.5)\
        *special.gamma(counter2+0.5))/(np.pi*(InitialJ-counter2+0.5)\
        *np.sqrt(special.gamma(counter2+1)*special.gamma(InitialJ+1)))
        Coeffs[counter2]=C**2
  
    Exponentials=np.zeros(Nmax,dtype=complex)
    for counter1 in range(int(Tmax/Spacing)):
        for counter2 in range(Nmax):
            Exponentials[counter2]=np.exp(-2j*(counter2-0.5-InitialJ)*np.pi*Tomega[counter1])

        S[counter1]=np.matmul(Coeffs,Exponentials)

    return Tomega, S

def NIIntRamsey(Nmax,InitialJ,FinalA,Tmax,Spacing):
    """
    Gives the two-body Ramsey signal over time 
    for a quench between from the non-interacting
    limit to some finite non-zero scattering length

    Parameters
    ------------------
    Nmax : an integer
        the number of terms in the sum we calculate up to
    InitialJ: a non-negative int
        exictation of the initial state. 0 is ground state,
        1 is first excited etc.
    FinalA: a real float
        final interaction strength
    Tmax : a positive float
        calculate Ramsey Signal from time=0 until time=Tmax
    Spacing : a small positive float
        the timestep size

    Returns
    -------------------
    S : a 1 x int(Tmax/spacing) array
        the Ramsey Signal as a function of time
    Tomega : a 1 x int(Tmax/spacing) array
        the time
    """

    Tomega=np.linspace(0,Tmax,int(Tmax/Spacing))
    S=np.zeros([int(Tmax/Spacing)],dtype=complex)

    Coeffs=np.zeros(Nmax)
    Exponentials=np.zeros(Nmax,dtype=complex)
    
    v=0.5*(Energies(Nmax,FinalA)-1.5)

    for i in range(Nmax):
        Z=np.pi*special.gamma(1-v[i])*(special.digamma(-v[i]-0.5)-special.digamma(-v[i]))/(v[i]*special.gamma(-v[i]-0.5))
        Coeffs[i]=((4/np.pi)**0.5)*(special.gamma(1.5)**2)/((v[i]**2)*Z)

    for counter1 in range(int(Tmax/Spacing)):
        for counter2 in range(Nmax):
            Exponentials[counter2]=np.exp(-2j*(v[counter2]-InitialJ)*np.pi*Tomega[counter1])
            
        S[counter1]=np.matmul(Coeffs,Exponentials)



    return Tomega, S

def UnitIntRamsey(Nmax,InitialJ,FinalA,Tmax,Spacing):
    """
    Gives the two-body Ramsey signal over time 
    for a quench between from the unitary limit
    to some finite non-zero scattering length

    Parameters
    ------------------
    Nmax : an integer
        the number of terms in the sum we calculate up to
    InitialJ: a non-negative int
        exictation of the initial state. 0 is ground state,
        1 is first excited etc.
    FinalA: a real float
        final interaction strength
    Tmax : a positive float
        calculate Ramsey Signal from time=0 until time=Tmax
    Spacing : a small positive float
        the timestep size

    Returns
    -------------------
    S : a 1 x int(Tmax/spacing) array
        the Ramsey Signal as a function of time
    Tomega : a 1 x int(Tmax/spacing) array
        the time
    """
    
    Tomega=np.linspace(0,Tmax,int(Tmax/Spacing))
    S=np.zeros([int(Tmax/Spacing)],dtype=complex)

    Coeffs=np.zeros(Nmax)
    Exponentials=np.zeros(Nmax,dtype=complex)
    
    V0=InitialJ-1/2
    Vf=0.5*(Energies(Nmax,FinalA)-1.5)


    Z0=(np.pi**1.5)*special.binom(InitialJ-1/2,InitialJ)**(-1)

    for j in range(Nmax):
        Zj=np.pi*(special.digamma(-Vf[j]-0.5)-special.digamma(-Vf[j]))*special.poch(-Vf[j]-1/2,3/2)/Vf[j]

        Overlap=np.sqrt(np.pi)*(2*V0*Vf[j]*np.sqrt(Z0*Zj))**(-1)\
        *mpmath.hyp3f2(1.5,-Vf[j],-V0,1-Vf[j],1-V0,1)

        Coeffs[j]=Overlap**2

    Exponentials=np.zeros(Nmax,dtype=complex)
    for counter1 in range(int(Tmax/Spacing)):
        for counter2 in range(Nmax):
            Exponentials[counter2]=np.exp(-2j*(Vf[counter2]-InitialJ)*np.pi*Tomega[counter1])
            
        S[counter1]=np.matmul(Coeffs,Exponentials)

    return Tomega, S

def UnitNIRamsey(Nmax,InitialJ,Tmax,Spacing):
    """
    Gives the two-body Ramsey signal over time 
    for a quench between from the unitary limit
    to the non-interacting limit

    Parameters
    ------------------
    Nmax : an integer
        the number of terms in the sum we calculate up to
    InitialJ: a non-negative int
        exictation of the initial state. 0 is ground state,
        1 is first excited etc.
    Tmax : a positive float
        calculate Ramsey Signal from time=0 until time=Tmax
    spacing : a small positive float
        the timestep size
        

    Returns
    -------------------
    S : a 1 x int(Tmax/spacing) array
        the Ramsey Signal as a function of time
    Tomega : a 1 x int(Tmax/spacing) array
        the time
    """

    Tomega=np.linspace(0,Tmax,int(Tmax/Spacing))

    S=np.zeros([int(Tmax/Spacing)],dtype=complex)

    Coeffs=np.zeros(Nmax)


    for counter2 in range(Nmax):
        C=np.sqrt((counter2+0.5)*special.gamma(counter2+0.5)\
        *special.gamma(InitialJ+0.5))/(np.pi*(counter2-InitialJ+0.5)\
        *np.sqrt(special.gamma(InitialJ+1)*special.gamma(counter2+1)))
        Coeffs[counter2]=C**2

    Exponentials=np.zeros(Nmax,dtype=complex)
    for counter1 in range(int(Tmax/Spacing)):
        for counter2 in range(Nmax):
            Exponentials[counter2]=np.exp(-2j*(counter2+0.5-InitialJ)*np.pi*Tomega[counter1])
            
        S[counter1]=np.matmul(Coeffs,Exponentials)

    return Tomega, S


#a single function we can call for general Ramsey calculations
def RamseySignal(Nmax,InitialA,InitialJ,FinalA,Tmax,Spacing):
    """
    Gives the two-body Ramsey signal over time 
    for an arbitrary quench


    Parameters
    ------------------
    Nmax: a positive int
        the number of terms included in the calculation
    InitialA: a real float or a string
        initial interaction strength
        if it is a string it should be "inf" for infinty, i.e.
        the unitary limit
    InitialJ: a non-negative int
        exictation of the initial state. 0 is ground state,
        1 is first excited etc.
    FinalA: a real float or a string
        final interaction strength
        if it is a string it should be "inf" for infinty, i.e.
        the unitary limit
    Tmax : a positive float
        calculate Ramsey Signal from time=0 until time=Tmax
    spacing : a small positive float
        the timestep size
        

    Returns
    -------------------
    S : a 1 x int(Tmax/spacing) array
        the Ramsey Signal as a function of time
    Tomega : a 1 x int(Tmax/spacing) array
        the time
    """

    #UnitNI
    if InitialA=="inf" and FinalA==0:
        [Tomega,S]=UnitNIRamsey(Nmax,InitialJ,Tmax,Spacing)
    
    #NIUnit
    if InitialA==0 and FinalA=="inf":
        [Tomega,S]=NIUnitRamsey(Nmax,InitialJ,Tmax,Spacing)
  
    #IntInt
    if InitialA!=0 and InitialA!="inf" and FinalA!=0 and FinalA!="inf" and InitialA != FinalA:
        [Tomega,S]=IntIntRamsey(Nmax,InitialA,InitialJ,FinalA,Tmax,Spacing)
  
    #IntNI
    if InitialA!=0 and InitialA!="inf" and FinalA==0:
        [Tomega,S]=IntNIRamsey(Nmax,InitialA,InitialJ,Tmax,Spacing)

    #NIInt
    if InitialA==0 and FinalA!=0 and FinalA!="inf":
        [Tomega,S]=NIIntRamsey(Nmax,InitialJ,FinalA,Tmax,Spacing)
       
    #IntUnit
    if InitialA!=0 and InitialA!="inf" and FinalA=="inf":
        [Tomega,S]=IntUnitRamsey(Nmax,InitialA,InitialJ,Tmax,Spacing)
  
    #UnitInt
    if InitialA=="inf" and FinalA!=0 and FinalA!="inf":
        [Tomega,S]=UnitIntRamsey(Nmax,InitialJ,FinalA,Tmax,Spacing)

    #UnitUnit
    if InitialA=="inf" and FinalA=="inf":
        print("Unitary to Unitary quench makes no sense")
        Tomega=np.linspace(0,Tmax,int(Tmax/Spacing))
        S=np.ones([int(Tmax/Spacing)],dtype=complex)

    #NINI
    if InitialA==0 and FinalA==0:
        print("NI to NI quench makes no sense")
        Tomega=np.linspace(0,Tmax,int(Tmax/Spacing))
        S=np.ones([int(Tmax/Spacing)],dtype=complex)

    #Int to same Int
    if InitialA==FinalA and FinalA!=0 and FinalA!="inf":
        print(str(InitialA)+" to "+str(FinalA)+" quench makes no sense")
        Tomega=np.linspace(0,Tmax,int(Tmax/Spacing))
        S=np.ones([int(Tmax/Spacing)],dtype=complex)




    return Tomega, S


#Particle Separation expectation.
def IntIntExpectR(Nmax,InitialA,InitialJ,FinalA,Tmax,Spacing):
    """
    Gives the two-body particle separation after a quench
    between two non-zero finite values of scattering length

    Parameters
    ------------------
    Nmax : an integer
        the number of terms in the sum we calculate up to
    InitialA: a real float 
        initial interaction strength
    InitialJ: a non-negative int
        exictation of the initial state. 0 is ground state,
        1 is first excited etc.
    FinalA: a real float
        final interaction strength
    Tmax : a positive float
        calculate Ramsey Signal from time=0 until time=Tmax
    spacing : a small positive float
        the timestep size
        

    Returns
    -------------------
    R : a 1 x int(Tmax/spacing) array
        the particle separation expectation
        as a function of time
    Tomega : a 1 x int(Tmax/spacing) array
        the time


    """

    Coeffs=np.zeros([Nmax,Nmax])

    R=np.zeros(int(Tmax/Spacing))
    Tomega=np.linspace(0,Tmax,int(Tmax/Spacing))

    V0=0.5*(Energies(InitialJ+1,InitialA)[InitialJ]-1.5)
    Z0=np.pi*(special.digamma(-V0-0.5)-special.digamma(-V0))*special.poch(-V0-1/2,3/2)/V0
    Vf=0.5*(Energies(Nmax,FinalA)-1.5)

    Variables=np.zeros([Nmax,3])
    Variables[:,0]=Vf
    Variables[:,1]=V0
    Variables[:,2]=Z0

    p=Pool()
    Overlaps=p.map(WavefFuncOverlaps,Variables)

    Z=np.zeros(Nmax)
    for j in range(Nmax):
        Z[j]=np.pi*special.gamma(1-Vf[j])*(special.digamma(-Vf[j]-0.5)-special.digamma(-Vf[j]))/(Vf[j]*special.gamma(-Vf[j]-0.5))

    for j in range(Nmax):
            
        for k in range(j,Nmax):

            CrossTerm=0
            #Size=k+10 is reliable for ensuring convergence of this double sum
            Size=k+10
            for n in range(Size):
                for m in range(Size):
                    CrossTerm=CrossTerm+((-1)**(m+n))*special.gamma(m+1.5)*special.gamma(n+1.5)\
                    /((m-Vf[j])*(n-Vf[k])*special.gamma(n+1)*special.gamma(m-n+1.5)*special.gamma(m+1)*special.gamma(n-m+1.5))

            CrossTerm=(1/np.sqrt(Z[k]*Z[j]))*CrossTerm

            Coeffs[j,k]=Overlaps[k]*Overlaps[j]*CrossTerm

            Coeffs[k,j]=Coeffs[j,k]

    ConstContrib=np.trace(Coeffs)
    for t in range(int(Tmax/Spacing)):
        for j in range(Nmax):
            for k in range(j+1,Nmax):
                R[t]=R[t]+np.real(Coeffs[j,k]*2*np.cos(-2*(Vf[j]-Vf[k])*np.pi*Tomega[t]))
    R=R+ConstContrib


    return Tomega, R

def IntUnitExpectR(Nmax,InitialA,InitialJ,Tmax,Spacing):
    """
    This function gives the expectation value of r over time after
    a quench from some finite non-zero interaction strength to 
    unitarity

    Parameters
    ------------------------------
    Nmax: a positive int
        the number of terms we evaluate up to
        typically set this to be fairly small ~5 it converges quickly
    InitialA: a float
        initial s-wave scattering length
    InitialJ: an int
        the prinicple quantum number of the initial interacting state,
        InitialJ=0 is ground
    Tmax: a float
        the maximum time we calculate up to
    Spacing: a float
        the time-step size, the time resolution
        
    Returns
    --------------------------------
    R: a 1 x int(Tmax/spacing) array of real positive floats
        the values of <r(t)>
    Tomega: a 1 x int(Tmax/spacing) array of real positive floats
        the values of t
    """

    Coeffs=np.zeros([Nmax,Nmax])
    R=np.zeros(int(Tmax/Spacing))
    Tomega=np.linspace(0,Tmax,int(Tmax/Spacing))

    V0=0.5*(Energies(InitialJ+1,InitialA)-1.5)[InitialJ]
    Z0=np.pi*special.gamma(1-V0)*(special.digamma(-V0-0.5)-special.digamma(-V0))/(V0*special.gamma(-V0-0.5))

    Z=np.zeros(Nmax)
    Overlaps=np.zeros(Nmax)
    for j in range(Nmax):
        Z[j]=(np.pi**2)*special.gamma(j+1)/special.gamma(j+1/2)
        Overlaps[j]=np.sqrt(np.pi)*(2*V0*(j-1/2)*np.sqrt(Z0*Z[j]))**(-1)\
        *mpmath.hyp3f2(1.5,1/2-j,-V0,3/2-j,1-V0,1)

    for j in range(Nmax):
        for k in range(j,Nmax):


            CrossTerm=0
            #k+10 is reliable for ensuring convergence of this double sum
            for n in range(k+10):
                for m in range(k+10):
                    CrossTerm=CrossTerm+((-1)**(m+n))*special.gamma(m+1.5)*special.gamma(n+1.5)\
                    /((m-j+1/2)*(n-k+1/2)*special.gamma(n+1)*special.gamma(m-n+1.5)*special.gamma(m+1)*special.gamma(n-m+1.5))

            CrossTerm=(1/np.sqrt(Z[k]*Z[j]))*CrossTerm
            Coeffs[j,k]=Overlaps[j]*Overlaps[k]*CrossTerm
            Coeffs[k,j]=Coeffs[j,k]

    ConstContrib=np.trace(Coeffs)
    for t in range(int(Tmax/Spacing)):
        for j in range(Nmax):
            for k in range(j+1,Nmax):
                R[t]=R[t]+np.real(Coeffs[j,k]*2*np.cos(-2*(j-k)*np.pi*Tomega[t]))
    R=R+ConstContrib

    return Tomega, R

def IntNIExpectR(Nmax,InitialA,InitialJ,Tmax,Spacing):
    """
    This function gives the expectation value of r over time after
    a quench from some finite non-zero interaction strength to the 
    non-interacting limit

    THIS IS DIVERGENT AS EXPECTED

    Parameters
    ------------------------------
    Nmax: a positive int
        the number of terms we evaluate up to
        typically set this to be fairly small ~5 it converges quickly
    InitialA: a float
        initial s-wave scattering length
    InitialJ: an int
        the prinicple quantum number of the initial interacting state,
        InitialJ=0 is ground
    Tmax: a float
        the maximum time we calculate up to
    Spacing: a float
        the time-step size, the time resolution
        
    Returns
    --------------------------------
    r: a 1 x int(Tmax/spacing) array of real positive floats
        the values of <r(t)>
    Tomega: a 1 x int(Tmax/spacing) array of real positive floats
        the values of t
    """

    Coeffs=np.zeros([Nmax,Nmax])
    R=np.zeros(int(Tmax/Spacing))
    Tomega=np.linspace(0,Tmax,int(Tmax/Spacing))

    V0=0.5*(Energies(InitialJ+1,InitialA)[InitialJ]-1.5)
    Z0=np.pi*special.gamma(1-V0)*(special.digamma(-V0-0.5)-special.digamma(-V0))/(V0*special.gamma(-V0-0.5))

    Overlaps=np.zeros(Nmax)
    for n in range(Nmax):
        Overlaps[n]=(((n-V0)*np.sqrt(Z0))**(-1))*np.sqrt(special.gamma(n+3/2)/special.gamma(n+1))

    for n in range(Nmax):
        for m in range(n,Nmax):

            CrossTerm=((-1)**(m+n))*np.sqrt(special.gamma(3/2+m)*special.gamma(3/2+n)/(special.gamma(1+m)*special.gamma(1+n)))\
            /(special.gamma(m-n+3/2)*special.gamma(n-m+3/2))

            Coeffs[n,m]=Overlaps[m]*Overlaps[n]*CrossTerm
            Coeffs[m,n]=Coeffs[n,m]

    ConstContrib=np.trace(Coeffs)
    for t in range(int(Tmax/Spacing)):
        for j in range(Nmax):
            for k in range(j+1,Nmax):
                R[t]=R[t]+np.real(Coeffs[j,k]*2*np.cos(-2*(j-k)*np.pi*Tomega[t]))
    R=R+ConstContrib      
  

    return Tomega, R

def UnitIntExpectR(Nmax,InitialJ,FinalA,Tmax,Spacing):
    """
    This function gives the expectation value of r over time in the unitary to
    finite non-zero scattering length

    Parameters
    ------------------------------
    Tmax: a float
        the maximum time we calculate up to
    spacing: a float
        the time-step size, the time resolution
    Nmax: an int
        the number of terms we evaluate <r> to, the j_{max}, k_{max}
    InitialJ: an int
        the prinicple quantum number of the initial interacting state,
        InitialJ=0 is ground
    FinalA: a float
        the s-wave scattering length of the final state

    Returns
    --------------------------------
    AnalyticR: a 1 x int(Tmax/spacing) array of floats
        the values of <r(t)>
    Tomega: a 1 x int(Tmax/spacing) array of floats
        the values of t
    """


    Coeffs=np.zeros([Nmax,Nmax])
    R=np.zeros(int(Tmax/Spacing))
    Tomega=np.linspace(0,Tmax,int(Tmax/Spacing))

    V0=InitialJ-1/2
    Vf=0.5*(Energies(Nmax,FinalA)-1.5)

    Z0=(np.pi**2)*special.gamma(InitialJ+1)/special.gamma(InitialJ+1/2)

    Z=np.zeros(Nmax)
    Overlaps=np.zeros(Nmax)
    for j in range(Nmax):
        Z[j]=np.pi*special.gamma(1-Vf[j])*(special.digamma(-Vf[j]-0.5)-special.digamma(-Vf[j]))/(Vf[j]*special.gamma(-Vf[j]-0.5))
        Overlaps[j]=np.sqrt(np.pi)*(2*V0*Vf[j]*np.sqrt(Z0*Z[j]))**(-1)\
                    *mpmath.hyp3f2(1.5,-Vf[j],-V0,1-Vf[j],1-V0,1)


    for j in range(Nmax):
        for k in range(j,Nmax):

            CrossTerm=0
            #Size=k+10 is reliable for ensuring convergence of this double sum
            Size=k+10
            for n in range(Size):
                for m in range(Size):
                    CrossTerm=CrossTerm+((-1)**(m+n))*special.gamma(m+1.5)*special.gamma(n+1.5)\
                    /((m-Vf[j])*(n-Vf[k])*special.gamma(n+1)*special.gamma(m-n+1.5)*special.gamma(m+1)*special.gamma(n-m+1.5))

            CrossTerm=(1/np.sqrt(Z[k]*Z[j]))*CrossTerm
            Coeffs[j,k]=Overlaps[k]*Overlaps[j]*CrossTerm
            Coeffs[k,j]=Coeffs[j,k]


    #ConstContrib=0
    #for j in range(Nmax):
    #    ConstContrib=ConstContrib+Coeffs[j,j]
    #for t in range(int(Tmax/Spacing)):
    #    for j in range(Nmax):
    #        for k in range(j+1,Nmax):
    #            R[t]=R[t]+np.real(Coeffs[j,k]*2*np.cos(-2*(j-k)*np.pi*Tomega[t]))
    #        #R[t]=R[t]+np.real(Coeffs[j,j])
    #    R[t]=R[t]+ConstContrib
    ConstContrib=np.trace(Coeffs)
    for t in range(int(Tmax/Spacing)):
        for j in range(Nmax):
            for k in range(j+1,Nmax):
                R[t]=R[t]+np.real(Coeffs[j,k]*2*np.cos(-2*(j-k)*np.pi*Tomega[t]))
    R=R+ConstContrib
    
    return Tomega, R

def UnitNIExpectR(Nmax,InitialJ,Tmax,Spacing):
    """
    This function gives the expectation value of r over time in the unitary to
    non interacting quench

    THIS IS DIVERGENT AS EXPECTED


    Parameters
    ------------------------------
    Tmax: a float
        the maximum time we calculate up to
    spacing: a float
        the time-step size, the time resolution
    Nmax: an int
        the number of terms we evaluate <r> to, the j_{max}, k_{max}
    InitialJ: an int
        the prinicple quantum number of the initial interacting state,
        InitialJ=0 is ground

    Returns
    --------------------------------
    AnalyticR: a 1 x int(Tmax/spacing) array of floats
        the values of <r(t)>
    Tomega: a 1 x int(Tmax/spacing) array of floats
        the values of t
    """

    Tomega=np.linspace(0,Tmax,int(Tmax/Spacing))

    Coeffs=np.zeros([Nmax,Nmax])
    R=np.zeros(int(Tmax/Spacing))

    for j in range(Nmax):
        for k in range(j,Nmax):
            Coeffs[j,k]=((-1)**(j+k))*np.sqrt(special.gamma(j+1.5)*special.gamma(k+1.5))/(np.sqrt(special.gamma(1+j)*special.gamma(1+k))*special.gamma(j-k+1.5)*special.gamma(k-j+1.5))\
            *special.gamma(InitialJ+0.5)*np.sqrt(special.gamma(j+1.5)*special.gamma(k+1.5))/((np.pi**2)*(j-InitialJ+0.5)*(k-InitialJ+0.5)*special.gamma(InitialJ+1)*np.sqrt(special.gamma(j+1)*special.gamma(k+1)))
            Coeffs[k,j]=Coeffs[j,k]

    ConstContrib=np.trace(Coeffs)
    for t in range(int(Tmax/Spacing)):
        for j in range(Nmax):
            for k in range(j+1,Nmax):
                R[t]=R[t]+np.real(Coeffs[j,k]*2*np.cos(-2*(j-k)*np.pi*Tomega[t]))
    R=R+ConstContrib

    return Tomega, R

def NIIntExpectR(Nmax,InitialJ,FinalA,Tmax,Spacing):
    """
    This function gives the expectation value of r over time in the unitary to
    general interacting case

    Parameters
    ------------------------------
    Tmax: a float
        the maximum time we calculate up to
    spacing: a float
        the time-step size, the time resolution
    Nmax: an int
        the number of terms we evaluate <r> to, the j_{max}, k_{max}
    InitialJ: an int
        the prinicple quantum number of the initial interacting state,
        InitialJ=0 is ground
    FinalA: a float
        the s-wave scattering length of the final state

    Returns
    --------------------------------
    AnalyticR: a 1 x int(Tmax/spacing) array of floats
        the values of <r(t)>
    Tomega: a 1 x int(Tmax/spacing) array of floats
        the values of t
    """



    Coeffs=np.zeros([Nmax,Nmax])
    R=np.zeros(int(Tmax/Spacing))
    Tomega=np.linspace(0,Tmax,int(Tmax/Spacing))

    Vf=0.5*(Energies(Nmax,FinalA)-1.5)

    Z=np.zeros(Nmax)
    Overlaps=np.zeros(Nmax)
    for j in range(Nmax):
        Z[j]=np.pi*special.gamma(1-Vf[j])*(special.digamma(-Vf[j]-0.5)-special.digamma(-Vf[j]))/(Vf[j]*special.gamma(-Vf[j]-0.5))
        Overlaps[j]=(((InitialJ-Vf[j])*np.sqrt(Z[j]))**(-1))*np.sqrt(special.gamma(InitialJ+3/2)/special.gamma(InitialJ+1))


    for j in range(Nmax):
        for k in range(j,Nmax):
            CrossTerm=0
            #Size=k+10 is reliable for ensuring convergence of this double sum
            Size=k+10
            for n in range(Size):
                for m in range(Size):
                    CrossTerm=CrossTerm+((-1)**(m+n))*special.gamma(m+1.5)*special.gamma(n+1.5)\
                    /((m-Vf[j])*(n-Vf[k])*special.gamma(n+1)*special.gamma(m-n+1.5)*special.gamma(m+1)*special.gamma(n-m+1.5))

            CrossTerm=(1/np.sqrt(Z[k]*Z[j]))*CrossTerm

            Coeffs[j,k]=Overlaps[j]*Overlaps[k]*CrossTerm
            Coeffs[k,j]=Coeffs[j,k]

   

    ConstContrib=np.trace(Coeffs)
    for t in range(int(Tmax/Spacing)):
        for j in range(Nmax):
            for k in range(j+1,Nmax):
                R[t]=R[t]+np.real(Coeffs[j,k]*2*np.cos(-2*(Vf[j]-Vf[k])*np.pi*Tomega[t]))
    R=R+ConstContrib

    return Tomega, R

def NIUnitExpectR(Nmax,InitialJ,Tmax,Spacing):
    """
    This function gives the expectation value of r over time in the non-int to
    unitary case, using the analytic evaluations of the integrals.

    Parameters
    ------------------------------
    Tmax: a float
        the maximum time we calculate up to
    spacing: a float
        the time-step size, the time resolution
    InitialN: an int
        the prinicple quantum number of the initial state, InitialN=0 is ground
    Nmax: an int
        the number of terms we evaluate <r> to, the j_{max}, k_{max}
        typically set this to be fairly small ~5 it converges quickly
    Size: an int
        the number of terms we evaluate the double sum that is for
        <psi_{k}|r|psi_{j}> to. typically set this to be fairly large ~1000
        this doesn't appear in the unitary to non-int quench because in order to
        evaluate the term here we have to expand in terms of laguerres but we
        don't have to do that in the unitary to non-int case

    """
    Coeffs=np.zeros([Nmax,Nmax])
    R=np.zeros(int(Tmax/Spacing))
    Tomega=np.linspace(0,Tmax,int(Tmax/Spacing))


    for j in range(Nmax):
        for k in range(j,Nmax):
            Size=k+10
            for n in range(Size):
                for m in range(Size):
                    Coeffs[j,k]=Coeffs[j,k]+((-1)**(n+m))/((n-j+0.5)*(m-k+0.5))*special.binom(m+0.5,n)*special.binom(n+0.5,m)

            Coeffs[j,k]=Coeffs[j,k]*(special.gamma(InitialJ+1.5)*(special.gamma(j+0.5)*special.gamma(k+0.5)))/(np.pi**4*(InitialJ-j+0.5)*(InitialJ-k+0.5)*((special.gamma(j+1)*special.gamma(k+1))*special.gamma(InitialJ+1)))
            Coeffs[k,j]=Coeffs[j,k]


    ConstContrib=np.trace(Coeffs)
    for t in range(int(Tmax/Spacing)):
        for j in range(Nmax):
            for k in range(j+1,Nmax):
                R[t]=R[t]+np.real(Coeffs[j,k]*2*np.cos(-2*(j-k)*np.pi*Tomega[t]))
    R=R+ConstContrib


    return Tomega, R


#a single function we call for general <r> calculations
def ExpectR(Nmax,InitialA,InitialJ,FinalA,Tmax,Spacing):
    """
    Gives the two-body particle separation expectation
    over time for an arbitrary quench


    Parameters
    ------------------
    Nmax: a positive int
        the number of terms included in the calculation
    InitialA: a real float or a string
        initial interaction strength
        if it is a string it should be "inf" for infinty, i.e.
        the unitary limit
    InitialJ: a non-negative int
        exictation of the initial state. 0 is ground state,
        1 is first excited etc.
    FinalA: a real float or a string
        final interaction strength
        if it is a string it should be "inf" for infinty, i.e.
        the unitary limit
    Tmax : a positive float
        calculate Ramsey Signal from time=0 until time=Tmax
    spacing : a small positive float
        the timestep size
        

    Returns
    -------------------
    R : a 1 x int(Tmax/spacing) array
        the particle separation expectation
        as a function of time
    Tomega : a 1 x int(Tmax/spacing) array
        the time


    """


    #UnitNI
    if InitialA=="inf" and FinalA==0:
        [Tomega,R]=UnitNIExpectR(Nmax,InitialJ,Tmax,Spacing)
    
    #NIUnit
    if InitialA==0 and FinalA=="inf":
        [Tomega,R]=NIUnitExpectR(Nmax,InitialJ,Tmax,Spacing)
  
    #IntInt
    if InitialA!=0 and InitialA!="inf" and FinalA!=0 and FinalA!="inf" and InitialA!=FinalA:
        [Tomega,R]=IntIntExpectR(Nmax,InitialA,InitialJ,FinalA,Tmax,Spacing)
  
    #IntNI
    if InitialA!=0 and InitialA!="inf" and FinalA==0:
        [Tomega,R]=IntNIExpectR(Nmax,InitialA,InitialJ,Tmax,Spacing)

    #NIInt
    if InitialA==0 and FinalA!=0 and FinalA!="inf":
        [Tomega,R]=NIIntExpectR(Nmax,InitialJ,FinalA,Tmax,Spacing)
       
    #IntUnit
    if InitialA!=0 and InitialA!="inf" and FinalA=="inf":
        [Tomega,R]=IntUnitExpectR(Nmax,InitialA,InitialJ,Tmax,Spacing)
  
    #UnitInt
    if InitialA=="inf" and FinalA!=0 and FinalA!="inf":
        [Tomega,R]=UnitIntExpectR(Nmax,InitialJ,FinalA,Tmax,Spacing)

    #UnitUnit
    if InitialA=="inf" and FinalA=="inf":
        print("Unitary to Unitary quench makes no sense")
        Tomega=np.linspace(0,Tmax,int(Tmax/Spacing))
        R=np.zeros([int(Tmax/Spacing)],dtype=complex)

    #NINI
    if InitialA==0 and FinalA==0:
        print("NI to NI quench makes no sense")
        Tomega=np.linspace(0,Tmax,int(Tmax/Spacing))
        R=np.zeros([int(Tmax/Spacing)],dtype=complex)

    #Int to same Int
    if InitialA==FinalA and FinalA!=0 and FinalA!="inf":
        print(str(InitialA)+" to "+str(FinalA)+" quench makes no sense")
        Tomega=np.linspace(0,Tmax,int(Tmax/Spacing))
        R=np.zeros([int(Tmax/Spacing)],dtype=complex)



    return Tomega, R



#particle separation probability distribution
def IntIntRProbDistrib(Nmax,InitialA,InitialJ,FinalA,Tomega,Rmax,Spacing):
    """
    Essentially, if you integrate this distribution from 0 to infinity then
    you get <r(t)> for a qunech between two finite non-zero scattering 
    lengths. 
    

    Parameters
    ----------------------
    Nmax: an int
        the maximum value of the double sum we go up to
    InitialA: a real float 
        initial interaction strength
    InitialJ: a non-negative int
        exictation of the initial state. 0 is ground state,
        1 is first excited etc.
    FinalA: a real float
        final interaction strength
    Tomega: a float
        the time value we evaluate the whole thing for
    Rmax: a float
        calculate all the values of this integrand from 0 to Rmax
    spacing: a float
        the resolution in r


    Returns 
    ----------------------
    Rrange: a 1 x int(Rmax/Spacing) array of real positive floats
        the r range we calculate P(r,t) over
    Distrib: a 1 x int(Rmax/Spacing) array of real positive floats
        P(r,t), the probability density

    """


    Rrange=np.linspace(Spacing**3,Rmax,int(Rmax/Spacing))
    Distrib=np.zeros(int(Rmax/Spacing),dtype=complex)
    
    VInitial=0.5*(Energies(1,InitialA)[InitialJ]-1.5)
    V=0.5*(Energies(Nmax,FinalA)-1.5)

    Z0=np.pi*(special.digamma(-VInitial-0.5)-special.digamma(-VInitial))\
            *special.poch(-VInitial-1/2,3/2)/VInitial
    
    V0=0.5*(Energies(InitialJ+1,InitialA)[InitialJ]-1.5)
    Z0=np.pi*(special.digamma(-V0-0.5)-special.digamma(-V0))*special.poch(-V0-1/2,3/2)/V0
    Vf=0.5*(Energies(Nmax,FinalA)-1.5)

    Variables=np.zeros([Nmax,3])
    Variables[:,0]=Vf
    Variables[:,1]=V0
    Variables[:,2]=Z0

    p=Pool()
    Overlaps=p.map(WavefFuncOverlaps,Variables)
        
  
    for counter in range(int(Rmax/Spacing)):
        for j in range(Nmax):
            for k in range(j+1,Nmax):
                
                Distrib[counter]=Distrib[counter]\
                +4*np.pi*Overlaps[j]*Overlaps[k]*IntWaveFunc(V[j],Rrange[counter])*IntWaveFunc(V[k],Rrange[counter])\
                *(Rrange[counter]**3)*2*np.cos(-2*(V[k]-V[j])*np.pi*Tomega)

            Distrib[counter]=Distrib[counter]\
            +4*np.pi*Overlaps[j]*Overlaps[j]*IntWaveFunc(V[j],Rrange[counter])*IntWaveFunc(V[j],Rrange[counter])\
            *(Rrange[counter]**3)
   
    return Rrange, Distrib

def UnitNIRProbDistrib(Nmax,InitialJ,Tomega,Rmax,Spacing):
    """
    Essentially, if you integrate this distribution from 0 to infinity then
    you get <r(t)> in the unitary to non-interacting case. 
    
    The key result here is that as you increase Nmax 
    (at Tomega not an integer multiple of pi) the area under the graph just
    increases and does not converge.  You should see a power law decay
    that ends in an exponential cut-off. The cut-off kicks in later and
    later depending on how large Nmax is.

    THIS IS DIVERGENT AS EXPECTED

    Parameters
    ----------------------
    Nmax: an int
        the maximum value of the double sum we go up to
    InitialJ: an int
        the excitation state of the initial unitary system
    Tomega: a float
        the time value we evaluate the whole thing for
    Rmax: a float
        calculate all the values of this integrand from 0 to Rmax
    spacing: a float
        the resolution in r

    Returns 
    ----------------------
    Rrange: a 1 x int(Rmax/Spacing) array of real positive floats
        the r range we calculate P(r,t) over
    Distrib: a 1 x int(Rmax/Spacing) array of real positive floats
        P(r,t), the probability density

    """
    Rrange=np.linspace(0,Rmax,int(Rmax/Spacing))
    Distrib=np.zeros(int(Rmax/Spacing),dtype=complex)

    for counter in range(int(Rmax/Spacing)):
        for j in range(Nmax):
            for k in range(j+1,Nmax):
                Distrib[counter]=Distrib[counter]+(special.gamma(InitialJ+0.5)*np.sqrt(special.gamma(j+1.5)*special.gamma(k+1.5))/((np.pi**2)*(j-InitialJ+0.5)*(k-InitialJ+0.5)*special.gamma(InitialJ+1)*np.sqrt(special.gamma(1+j)*special.gamma(1+k))))\
                *4*np.pi*NIWaveFunc(j,Rrange[counter])*NIWaveFunc(k,Rrange[counter])*(Rrange[counter]**3)*2*np.cos(-2*(k-j)*np.pi*Tomega)

            Distrib[counter]=Distrib[counter]+(special.gamma(InitialJ+0.5)*np.sqrt(special.gamma(j+1.5)*special.gamma(j+1.5))/((np.pi**2)*(j-InitialJ+0.5)*(j-InitialJ+0.5)*special.gamma(InitialJ+1)*np.sqrt(special.gamma(1+j)*special.gamma(1+j))))\
            *4*np.pi*NIWaveFunc(j,Rrange[counter])*NIWaveFunc(j,Rrange[counter])*(Rrange[counter]**3)

    return Rrange, Distrib

def NIUnitRProbDistrib(Nmax,InitialJ,Tomega,Rmax,Spacing):
    """
    Essentially, if you integrate this distribution from 0 to infinity then
    you get <r(t)> in the NI to Unitary case. The area under this curve
    should be convergent.

    This gives out at Nmax~8 depending on InitialJ and
    Tomega. The singulariries in the gamma functions are a problem.

    It appears to be convergent but the low Nmax limit is frustrating


    Parameters
    ----------------------
    Nmax: an int
        the maximum value of the double sum we go up to
    InitialJ: an int
        the excitation state of the initial NI system
    Tomega: a float
        the time value we evaluate the whole thing for
    Rmax: a float
        calculate all the values of this integrand from 0 to Rmax
    spacing: a float
        the resolution in r

    Returns 
    ----------------------
    Rrange: a 1 x int(Rmax/Spacing) array of real positive floats
        the r range we calculate P(r,t) over
    Distrib: a 1 x int(Rmax/Spacing) array of real positive floats
        P(r,t), the probability density

    """

    Epsilon=0.001
    Rrange=np.linspace(Epsilon,Rmax,int(Rmax/Spacing))
    Distrib=np.zeros(int(Rmax/Spacing),dtype=complex)

    Overlaps=np.zeros(Nmax)

    for j in range(Nmax):
        Overlaps[j]=np.sqrt((InitialJ+0.5)*special.gamma(InitialJ+0.5)\
                    *special.gamma(j+0.5))/(np.pi*(InitialJ-j+0.5)\
                    *np.sqrt(special.gamma(j+1)*special.gamma(InitialJ+1)))

    for counter in range(int(Rmax/Spacing)):
        for j in range(Nmax):
            for k in range(j+1,Nmax):
                
                Distrib[counter]=Distrib[counter]\
                +4*np.pi*Overlaps[j]*Overlaps[k]*IntWaveFunc(j-0.5+Epsilon,Rrange[counter])*IntWaveFunc(k-0.5+Epsilon,Rrange[counter])\
                *(Rrange[counter]**3)*2*np.cos(-2*(k-j)*np.pi*Tomega)

            Distrib[counter]=Distrib[counter]\
            +4*np.pi*Overlaps[j]*Overlaps[j]*IntWaveFunc(j-0.5+Epsilon,Rrange[counter])*IntWaveFunc(j-0.5+Epsilon,Rrange[counter])\
            *(Rrange[counter]**3)
    

    return Rrange, Distrib

def IntNIRProbDistrib(Nmax,InitialA,InitialJ,Tomega,Rmax,Spacing):
    """
    Essentially, if you integrate this distribution from 0 to infinity then
    you get <r(t)> for a qunech from a finite non-zero scattering length
    to non-interacting.

    The key result here is that as you increase Nmax 
    (at Tomega not an integer multiple of pi) the area under the graph just
    increases and does not converge.  You should see a power law decay
    that ends in an exponential cut-off. The cut-off kicks in later and
    later depending on how large Nmax is.

    THIS IS DIVERGENT AS EXPECTED
    
    

    Parameters
    ----------------------
    Nmax: an int
        the maximum value of the double sum we go up to
    InitialA: a real float 
        initial interaction strength
    InitialJ: a non-negative int
        exictation of the initial state. 0 is ground state,
        1 is first excited etc.
    Tomega: a float
        the time value we evaluate the whole thing for
    Rmax: a float
        calculate all the values of this integrand from 0 to Rmax
    spacing: a float
        the resolution in r

    Returns 
    ----------------------
    Rrange: a 1 x int(Rmax/Spacing) array of real positive floats
        the r range we calculate P(r,t) over
    Distrib: a 1 x int(Rmax/Spacing) array of real positive floats
        P(r,t), the probability density

    """

    Rrange=np.linspace(0,Rmax,int(Rmax/Spacing))
    Distrib=np.zeros(int(Rmax/Spacing),dtype=complex)


    Overlaps=np.zeros(Nmax)

    V0=0.5*(Energies(InitialJ+1,InitialA)[InitialJ]-1.5)
    Z0=np.pi*special.gamma(1-V0)*(special.digamma(-V0-0.5)-special.digamma(-V0))/(V0*special.gamma(-V0-0.5))


    for j in range(Nmax):
        Overlaps[j]=(((j-V0)*np.sqrt(Z0))**(-1))*np.sqrt(special.gamma(j+3/2)/special.gamma(j+1))

    for counter in range(int(Rmax/Spacing)):
        for j in range(Nmax):
            for k in range(j+1,Nmax):
                Distrib[counter]=Distrib[counter]+Overlaps[j]*Overlaps[k]\
                *4*np.pi*NIWaveFunc(j,Rrange[counter])*NIWaveFunc(k,Rrange[counter])*(Rrange[counter]**3)*2*np.cos(-2*(k-j)*np.pi*Tomega)

            Distrib[counter]=Distrib[counter]+Overlaps[j]*Overlaps[j]\
            *4*np.pi*NIWaveFunc(j,Rrange[counter])*NIWaveFunc(j,Rrange[counter])*(Rrange[counter]**3)
        #print(str(counter)+"of"+str(int(Rmax/Spacing)))
    
    
    #ExpectR=Spacing*sum(Distrib)
    #print("<r(t="+str(Tomega)+")>="+str(ExpectR))

    return Rrange, Distrib

def NIIntRProbDistrib(Nmax,InitialJ,FinalA,Tomega,Rmax,Spacing):
    """
    Essentially, if you integrate this distribution from 0 to infinity then
    you get <r(t)> for a qunech from the NI limit to a finite non-zero
    scattering length

    This craps out at Nmax~8 depending on parameters. a

    Parameters
    ----------------------
    Nmax: an int
        the maximum value of the double sum we go up to
    InitialJ: a non-negative int
        exictation of the initial state. 0 is ground state,
        1 is first excited etc.
    FinalA: a real float
        final interaction strength
    Tomega: a float
        the time value we evaluate the whole thing for
    Rmax: a float
        calculate all the values of this integrand from 0 to Rmax
    spacing: a float
        the resolution in r


    Returns 
    ----------------------
    Rrange: a 1 x int(Rmax/Spacing) array of real positive floats
        the r range we calculate P(r,t) over
    Distrib: a 1 x int(Rmax/Spacing) array of real positive floats
        P(r,t), the probability density

    """

    Rrange=np.linspace(Spacing**3,Rmax,int(Rmax/Spacing))
    Distrib=np.zeros(int(Rmax/Spacing),dtype=complex)


    Vf=0.5*(Energies(Nmax,FinalA)-1.5)
    Overlaps=np.zeros(Nmax)

    
    for j in range(Nmax):
        Zj=np.pi*(special.digamma(-Vf[j]-0.5)-special.digamma(-Vf[j]))*special.poch(-Vf[j]-1/2,3/2)/Vf[j]
        Overlaps[j]=np.sqrt(InitialJ+1/2)*np.sqrt(np.sqrt(np.pi)*special.binom(InitialJ-1/2,InitialJ))/((InitialJ-Vf[j])*np.sqrt(Zj))


    for counter in range(int(Rmax/Spacing)):
        for j in range(Nmax):
            for k in range(j+1,Nmax):
                
                Distrib[counter]=Distrib[counter]\
                +4*np.pi*Overlaps[j]*Overlaps[k]*IntWaveFunc(Vf[j],Rrange[counter])*IntWaveFunc(Vf[k],Rrange[counter])\
                *(Rrange[counter]**3)*2*np.cos(-2*(Vf[k]-Vf[j])*np.pi*Tomega)

            Distrib[counter]=Distrib[counter]\
            +4*np.pi*Overlaps[j]*Overlaps[j]*IntWaveFunc(Vf[j],Rrange[counter])*IntWaveFunc(Vf[j],Rrange[counter])\
            *(Rrange[counter]**3)

    return Rrange, Distrib
    
def IntUnitRProbDistrib(Nmax,InitialA,InitialJ,Tomega,Rmax,Spacing):
    """
    Essentially, if you integrate this distribution from 0 to infinity then
    you get <r(t)> for a qunech from a finite non-zero scattering length
    to unitarity

    This craps out at Nmax~8 depending on parameters. 


    Parameters
    ----------------------
    Nmax: an int
        the maximum value of the double sum we go up to
    InitialA: a real float 
        initial interaction strength
    InitialJ: a non-negative int
        exictation of the initial state. 0 is ground state,
        1 is first excited etc.
    Tomega: a float
        the time value we evaluate the whole thing for
    Rmax: a float
        calculate all the values of this integrand from 0 to Rmax
    spacing: a float
        the resolution in r

        
    Returns 
    ----------------------
    Rrange: a 1 x int(Rmax/Spacing) array of real positive floats
        the r range we calculate P(r,t) over
    Distrib: a 1 x int(Rmax/Spacing) array of real positive floats
        P(r,t), the probability density

    """

    Epsilon=0.001
    Rrange=np.linspace(Spacing**3,Rmax,int(Rmax/Spacing))
    Distrib=np.zeros(int(Rmax/Spacing),dtype=complex)

    VInitial=0.5*(Energies(1,InitialA)[InitialJ]-1.5)
    

    Overlaps=np.zeros(Nmax)
    Z0=np.pi*special.gamma(1-VInitial)*(special.digamma(-VInitial-0.5)-special.digamma(-VInitial))\
        /(VInitial*special.gamma(-VInitial-0.5))
    for j in range(Nmax):
        Zj=(np.pi**1.5)*special.binom(j-1/2,j)**-1
        Overlaps[j]=np.sqrt(np.pi)*(2*VInitial*(j-1/2)*np.sqrt(Z0*Zj))**(-1)\
        *mpmath.hyp3f2(1.5,1/2-j,-VInitial,3/2-j,1-VInitial,1)

    for counter in range(int(Rmax/Spacing)):
        for j in range(Nmax):
            for k in range(j+1,Nmax):
                
                Distrib[counter]=Distrib[counter]\
                +4*np.pi*Overlaps[j]*Overlaps[k]*IntWaveFunc(j-0.5-Epsilon,Rrange[counter])*IntWaveFunc(k-0.5-Epsilon,Rrange[counter])\
                *(Rrange[counter]**3)*2*np.cos(-2*(k-j)*np.pi*Tomega)

            Distrib[counter]=Distrib[counter]\
            +4*np.pi*Overlaps[j]*Overlaps[j]*IntWaveFunc(j-0.5-Epsilon,Rrange[counter])*IntWaveFunc(j-0.5-Epsilon,Rrange[counter])\
            *(Rrange[counter]**3)

    return Rrange, Distrib

def UnitIntRProbDistrib(Nmax,InitialJ,FinalA,Tomega,Rmax,Spacing):
    """
    Essentially, if you integrate this distribution from 0 to infinity then
    you get <r(t)> for a qunech from unitarity to a finite non-zero
    scattering length
    
    This craps out at Nmax~8 depending on parameters.


    Parameters
    ----------------------
    Nmax: an int
        the maximum value of the double sum we go up to
    InitialJ: a non-negative int
        exictation of the initial state. 0 is ground state,
        1 is first excited etc.
    FinalA: a real float
        final interaction strength
    Tomega: a float
        the time value we evaluate the whole thing for
    Rmax: a float
        calculate all the values of this integrand from 0 to Rmax
    spacing: a float
        the resolution in r


    Returns 
    ----------------------
    Rrange: a 1 x int(Rmax/Spacing) array of real positive floats
        the r range we calculate P(r,t) over
    Distrib: a 1 x int(Rmax/Spacing) array of real positive floats
        P(r,t), the probability density

    """
    
    
    Rrange=np.linspace(Spacing**3,Rmax,int(Rmax/Spacing))
    Distrib=np.zeros(int(Rmax/Spacing),dtype=complex)

    Overlaps=np.zeros(Nmax)
    
    Vf=0.5*(Energies(Nmax,FinalA)-1.5)

    V0=InitialJ-1/2
    Z0=(np.pi**2)*special.gamma(InitialJ+1)/special.gamma(InitialJ+1/2)

    for j in range(Nmax):

        Zj=np.pi*(special.digamma(-Vf[j]-0.5)-special.digamma(-Vf[j]))*special.poch(-Vf[j]-1/2,3/2)/Vf[j]

        Overlaps[j]=np.sqrt(np.pi)*(2*V0*Vf[j]*np.sqrt(Z0*Zj))**(-1)\
        *mpmath.hyp3f2(1.5,-Vf[j],-V0,1-Vf[j],1-V0,1)

    for counter in range(int(Rmax/Spacing)):
        for j in range(Nmax):
            for k in range(j+1,Nmax):
                
                Distrib[counter]=Distrib[counter]\
                +4*np.pi*Overlaps[j]*Overlaps[k]*IntWaveFunc(Vf[j],Rrange[counter])*IntWaveFunc(Vf[k],Rrange[counter])\
                *(Rrange[counter]**3)*2*np.cos(-2*(Vf[k]-Vf[j])*np.pi*Tomega)

            Distrib[counter]=Distrib[counter]\
            +4*np.pi*Overlaps[j]*Overlaps[j]*IntWaveFunc(Vf[j],Rrange[counter])*IntWaveFunc(Vf[j],Rrange[counter])\
            *(Rrange[counter]**3)
 
    return Rrange, Distrib


#a single function we call for general P(r,t) calculations
def RProbDistrib(Nmax,InitialA,InitialJ,FinalA,Tomega,Rmax,Spacing):
    """
    Gives the interparticle separation probability distribution
    as a function of time following an arbitrary quench.


    Parameters
    ----------------------
    Nmax: an int
        the maximum value of the double sum we go up to
    InitialA: a real float 
        initial interaction strength
    InitialJ: a non-negative int
        exictation of the initial state. 0 is ground state,
        1 is first excited etc.
    FinalA: a real float
        final interaction strength
    Tomega: a float
        the time value we evaluate the whole thing for
    Rmax: a float
        calculate all the values of this integrand from 0 to Rmax
    spacing: a float
        the resolution in r
    """

    #UnitNI
    if InitialA=="inf" and FinalA==0:
        [Rrange, Distrib]=UnitNIRProbDistrib(Nmax,InitialJ,Tomega,Rmax,Spacing)
        title=r"$P(r,\omega t/\pi$="+str(Tomega)+") for a Unitarity to NI quench." + r" $n_{\rm i}$="+str(InitialJ)
    
    #NIUnit
    if InitialA==0 and FinalA=="inf":
        [Rrange, Distrib]=NIUnitRProbDistrib(Nmax,InitialJ,Tomega,Rmax,Spacing)
        title=r"$P(r,\omega t/\pi$="+str(Tomega)+") for a Unitarity to NI quench." + r" $n_{\rm i}$="+str(InitialJ)

    #IntInt
    if InitialA!=0 and InitialA!="inf" and FinalA!=0 and FinalA!="inf" and InitialA != FinalA:
        [Rrange, Distrib]=IntIntRProbDistrib(Nmax,InitialA,InitialJ,FinalA,Tomega,Rmax,Spacing)
        title=r"$P(r,\omega t/\pi$="+str(Tomega)+") for a "+"$a_{s}=$"+str(InitialA)+" to $a_{s}=$" +str(FinalA)+ " quench." + r" $n_{\rm i}$="+str(InitialJ) 

    #IntNI
    if InitialA!=0 and InitialA!="inf" and FinalA==0:
        [Rrange, Distrib]=IntNIRProbDistrib(Nmax,InitialA,InitialJ,Tomega,Rmax,Spacing)
        title=r"$P(r,\omega t/\pi$="+str(Tomega)+") for a "+"$a_{s}=$"+str(InitialA)+" to NI quench." + r" $n_{\rm i}$="+str(InitialJ)

    #NIInt
    if InitialA==0 and FinalA!=0 and FinalA!="inf":
        [Rrange, Distrib]=NIIntRProbDistrib(Nmax,InitialJ,FinalA,Tomega,Rmax,Spacing)
        title=r"$P(r,\omega t/\pi$="+str(Tomega)+") for a NI to $a_{s}=$" +str(FinalA)+ " quench." + r" $n_{\rm i}$="+str(InitialJ) 

    #IntUnit
    if InitialA!=0 and InitialA!="inf" and FinalA=="inf":
        [Rrange, Distrib]=IntUnitRProbDistrib(Nmax,InitialA,InitialJ,Tomega,Rmax,Spacing)
        title=r"$P(r,\omega t/\pi$="+str(Tomega)+") for a "+"$a_{s}=$"+str(InitialA)+" to unitary quench." + r" $n_{\rm i}$="+str(InitialJ) 

    #UnitInt
    if InitialA=="inf" and FinalA!=0 and FinalA!="inf":
        [Rrange, Distrib]=UnitIntRProbDistrib(Nmax,InitialJ,FinalA,Tomega,Rmax,Spacing)
        title=r"$P(r,\omega t/\pi$="+str(Tomega)+") for a unitary to $a_{s}=$" +str(FinalA)+ " quench." + r" $n_{\rm i}$="+str(InitialJ) 

    #UnitUnit
    if InitialA=="inf" and FinalA=="inf":
        print("Unitary to Unitary quench makes no sense")
        Rrange=np.linspace(0,Rmax,int(Rmax/Spacing))
        Distrib=np.zeros(len(Rrange))

    #NINI
    if InitialA==0 and FinalA==0:
        print("NI to NI quench makes no sense")
        Rrange=np.linspace(0,Rmax,int(Rmax/Spacing))
        Distrib=np.zeros(len(Rrange))

    #Int to Same Int
    if InitialA==FinalA and FinalA!=0 and FinalA!="inf":
        print(str(InitialA)+" to "+str(FinalA)+" quench makes no sense")
        Rrange=np.linspace(0,Rmax,int(Rmax/Spacing))
        Distrib=np.zeros(len(Rrange))


    return Rrange, Distrib