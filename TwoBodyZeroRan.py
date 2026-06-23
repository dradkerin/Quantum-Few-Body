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
import matplotlib.pyplot as plt
import time as ti
from matplotlib import rc
from multiprocessing import Pool
rc('font',**{'family':'DejaVu Sans','serif':['Computer Modern']})
rc('text',usetex=True)

"""
README

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
        positive integer because gamma(-int) is infinite    
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
    Gives us the normalisation of the interacting two-body wavefunction 

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
    X=IntNorm(v)*IntWaveFuncNoNorm(v,r)

    return X


def NIWaveFunc(n,r):
    """
    The normalised noninteracting SHO wavefunction for l=0.

    We only care about the l=0 case for physical reasons.

    The energy associated with this wavefunction is (2n+1.5)hw
    
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
    s-wave scattering length a. In units of a=a_s/a_rel

    The output of this function agrees with mathematica calculations.

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
    Energies=np.zeros(Nmax)

    func = lambda v : (2/np.sqrt(np.pi))*special.binom(-v-3/2,-v-1)**-1-(a)**-1

    for j in range(Nmax):
        Energies[j]=fsolve(func,-0.01+j)

    Energies=2*Energies+3/2

    #this is a cludge because i dont understand the ultimate cause of the problem
    # but it gets wonky around those indices 
    if Nmax >= 171:
        Energies[171]=Energies[170]+2
    if Nmax >= 172:
        Energies[172]=Energies[171]+2

    return Energies


#post-quench energy expectation
def IntIntExpectE(Nmax,InitialA,InitialJ,FinalA):
    """
    This calculates the energy expectation value of the post-quench state
    for a quench between any two non-zero non-infinite scattering lengths


    Parameters
    ---------------------
    Nmax: a positive int
        the number of terms included in the calculation
    InitialA: a real float
        initial interaction strength
    InitialJ: a non-negative int
        exictation of the initial state. 0 is ground state,
        1 is first excited etc.
    FinalA: a real float
        final interaction strength

    Returns
    ---------------------
    SumSize: a 1 x Nmax array of positive ints
        an array of [1,2,3,...,Nmax] 
    
    CumulEnergies: a 1 x Nmax array of real floats
        <E> as a function of number of terms


    """
    t1=ti.time()
    Terms=np.zeros(Nmax)
    NormCheck=np.zeros(Nmax)

    V0=0.5*(Energies(InitialJ+1,InitialA)[InitialJ]-1.5)
    Vf=0.5*(Energies(Nmax,FinalA)-1.5)


    Z0=np.pi*(special.digamma(-V0-0.5)-special.digamma(-V0))*special.poch(-V0-1/2,3/2)/V0
    for j in range(Nmax):
        Zj=np.pi*(special.digamma(-Vf[j]-0.5)-special.digamma(-Vf[j]))*special.poch(-Vf[j]-1/2,3/2)/Vf[j]

        OverlapJ=np.sqrt(np.pi)*(2*V0*Vf[j]*np.sqrt(Z0*Zj))**(-1)\
        *mpmath.hyp3f2(1.5,-Vf[j],-V0,1-Vf[j],1-V0,1)

        Terms[j]=(2*Vf[j]+3/2)*abs(OverlapJ)**2
        NormCheck[j]=abs(OverlapJ)**2

        if j%10==0:
            print(j,Vf[j])

    CumulEnergies=np.cumsum(Terms)
    SumSize=np.linspace(1,Nmax,Nmax)

    t2=ti.time()
    print("Nmax=",Nmax)
    print("Initial Energy=",2*V0+3/2," Final Energy=", sum(Terms))
    print("Normalises to ",sum(NormCheck))
    print("Code took "+str(t2-t1)+"seconds. " +str(Nmax/(t2-t1))+" seconds per term")



    return SumSize, CumulEnergies

def IntIntParallel(Variables):
    """

    This calculates the square overlap of two interacting wavefunctions.
    It is designed to be a part of parallelised code

    Parameters
    ---------------------
    Variables: a 1 x 3 array of real floats
        Variables[0] is Vf, the energy pseudo-quantum number for the 
        final state 
        Variables[1] is V0, the energy pseudo-quantum number for the 
        initial state
        Variables[2] is Z0, is a part of the normalisation of the
        initial state it is faster to calculate it onece before passing
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

    OverlapJ=np.sqrt(np.pi)*(2*V0*Vf*np.sqrt(Z0*Zj))**(-1)\
    *mpmath.hyp3f2(1.5,-Vf,-V0,1-Vf,1-V0,1)

    SquareOverlap=abs(OverlapJ)**2

    print("Vf=",Vf)

    return SquareOverlap

def IntIntExpectEParallel(Nmax,InitialA,InitialJ,FinalA):
    """
    Plots the expectation of the post quench energy of the system as a function
    of the number of terms in the expansion. Quench is arbitrary to arbitrary a_{s}

    Same as IntToIntExpectE except parallelised to make it faster

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

    t1=ti.time()

    V0=0.5*(Energies(InitialJ+1,InitialA)[InitialJ]-1.5)
    Z0=np.pi*(special.digamma(-V0-0.5)-special.digamma(-V0))*special.poch(-V0-1/2,3/2)/V0
    Vf=0.5*(Energies(Nmax,FinalA)-1.5)

    Variables=np.zeros([Nmax,3])
    Variables[:,0]=Vf
    Variables[:,1]=V0
    Variables[:,2]=Z0


    Terms=np.zeros(Nmax)
    p=Pool()
    Terms=p.map(IntIntParallel,Variables)

    for j in range(Nmax):
        Terms[j]=(2*Vf[j]+3/2)*Terms[j]


    CumulEnergies=np.cumsum(Terms)
    SumSize=np.linspace(1,Nmax,Nmax)

    t2=ti.time()
    print("Nmax=",Nmax)
    print("Initial Energy=",2*V0+3/2," Final Energy=", sum(Terms))
    print("Normalises to ",sum(Terms))
    print("Code took "+str(t2-t1)+"seconds. " +str(Nmax/(t2-t1))+" terms per second")
    print("Code took "+str(t2-t1)+"seconds. " +str((t2-t1)/Nmax)+" seconds per term")



    return SumSize,CumulEnergies

def IntUnitExpectE(Nmax,InitialA,InitialJ):
    """
    Plots the expectation of the post quench energy of the system as a function
    of the number of terms in the expansion. Quench is from arbitrary a_{s} to
    unitarity.

    This is the slow one. 
    
    Appears to be convergent as expected

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

    #print("v=",V0)
    #print("Z0=",Z0)

    for j in range(Nmax):
        Zj=(np.pi**1.5)*special.binom(j-1/2,j)**-1
        OverlapJ=np.sqrt(np.pi)*(2*V0*(j-1/2)*np.sqrt(Z0*Zj))**(-1)\
        *mpmath.hyp3f2(1.5,1/2-j,-V0,3/2-j,1-V0,1)

        Terms[j]=(2*j+1/2)*OverlapJ**2
        NormCheck[j]=OverlapJ**2

        print(j,"/",Nmax)
        #print("j=",j)
        #print("Zj=",Zj)
        #print("Overlap=",OverlapJ)

    CumulEnergies=np.cumsum(Terms)
    SumSize=np.linspace(1,Nmax,Nmax)

    print("Nmax=",Nmax)
    print("Initial energy=",2*V0+3/2," Final Energy=", sum(Terms))
    print("Normalises to ",sum(NormCheck))
    #print(NormCheck)
    #print(Terms)

    return SumSize, CumulEnergies

def IntNIExpectE(Nmax,InitialA,InitialJ):
    """
    Plots the expectation of the post quench energy of the system as a function
    of the number of terms in the expansion.

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
        print("RED FLAG: ENERGIES NOT CHANGING")


    CumulEnergies=np.cumsum(Terms)
    SumSize=np.linspace(1,Nmax,Nmax)

    print("Nmax=",Nmax)

    print("Initial Energy=",2*V0+1/2," Final Energy=", sum(Terms))
    print("Normalises to ",sum(NormCheck))


    return SumSize, CumulEnergies

def NIUnitExpectE(Nmax,InitialN):
    """
    Plots the expectation of the post quench energy of the system as a function
    of the number of terms in the expansion. Quench is from NonInt to unitarity

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

    print("Nmax=",Nmax)
    print("Initial Energy=",2*InitialN+3/2," Final Energy=", sum(Terms))
    print("Normalises to ",sum(NormCheck))


    return SumSize, CumulEnergies

def NIIntExpectE(Nmax,InitialN,FinalA):
    """
    Plots the expectation of the post quench energy of the system as a function
    of the number of terms in the expansion. Quench is from NonInt to arbitrary
    a_s

    this is convergent

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

    print("Nmax=",Nmax)
    print("Initial Energy=",2*InitialN+3/2," Final Energy=", sum(Terms))
    print("Normalises to ",sum(NormCheck))

    return SumSize, CumulEnergies

def UnitNIExpectE(Nmax,InitialJ):
    """
    Plots the expectation of the post quench energy of the system as a function
    of the number of terms in the expansion. Quench is from unitarity to NonInt
    this is the one that diverges


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
    NormCheck=np.zeros(Nmax)

    #the commented out code has can't handle Nmax>~179 because of the gamma
    #functions. The new code is the same thing just expressed in a diff form
    for n in range(Nmax):
        #C=np.sqrt((n+0.5)*special.gamma(n+0.5)*special.gamma(InitialJ+0.5))/(np.pi*(n-InitialJ+0.5)*np.sqrt(special.gamma(InitialJ+1)*special.gamma(n+1)))
        #Terms[n]=(2*n+3/2)*C**2
        #NormCheck[n]=C**2
        C=np.sqrt((n+1/2)*special.binom(n-1/2,n)*special.binom(InitialJ-1/2,InitialJ))/(np.sqrt(np.pi)*(n-InitialJ+1/2))
        Terms[n]=(2*n+3/2)*C**2
        NormCheck[n]=C**2

    CumulEnergies=np.cumsum(Terms)
    SumSize=np.linspace(1,Nmax,Nmax)

    print("Nmax=",Nmax)
    print("Initial Energy=",2*InitialJ+1/2," Final Energy=", sum(Terms))
    print("Normalises to ",sum(NormCheck))



    return SumSize, CumulEnergies

def UnitIntExpectE(Nmax,InitialJ,FinalA):
    """
    Plots the expectation of the post quench energy of the system as a function
    of the number of terms in the expansion. Quench is arbitrary to arbitrary

    convergent

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
        print("RED FLAG: ENERGIES NOT CHANGING")
    CumulEnergies=np.cumsum(Terms)
    SumSize=np.linspace(1,Nmax,Nmax)

    print("Nmax=",Nmax)
    print("Initial Energy=",2*V0+3/2," Final Energy=", sum(Terms))
    print("Normalises to ",sum(NormCheck))

  
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

    if InitialA == 0:
        if FinalA == 0:
            print("NI to NI quench makes no sense")
            #the first term of the sum (not cumulative sum)
            #will be the initial energy and all others zero
            SumSize=np.linspace(1,Nmax,Nmax)
            CumulEnergies=np.zeros(Nmax)
            CumulEnergies[:]=2*InitialJ+1.5
        elif FinalA != "inf": 
            [SumSize,CumulEnergies]=NIIntExpectE(Nmax,InitialJ,FinalA)
        if FinalA== "inf":
            [SumSize,CumulEnergies]=NIUnitExpectE(Nmax,InitialJ)
        
    elif InitialA != "inf": 
        if FinalA == 0:
            [SumSize,CumulEnergies]=IntNIExpectE(Nmax,InitialA,InitialJ)
        elif FinalA != "inf":
            [SumSize,CumulEnergies]=IntIntExpectEParallel(Nmax,InitialA,InitialJ,FinalA)
        if FinalA == "inf":
            [SumSize,CumulEnergies]=IntUnitExpectE(Nmax,InitialA,InitialJ)


    if InitialA == "inf":
        if FinalA == 0:
            [SumSize,CumulEnergies]=UnitNIExpectE(Nmax,InitialJ)
        elif FinalA != "inf":
            [SumSize,CumulEnergies]=UnitIntExpectE(Nmax,InitialJ,FinalA)
        if FinalA == "inf":
            print("unitary to unitary quench makes no sense")
            #the first term of the sum (not cumulative sum)
            #will be the initial energy and all others zero
            SumSize=np.linspace(1,Nmax,Nmax)
            CumulEnergies=np.zeros(Nmax)
            CumulEnergies[:]=2*InitialJ+0.5
            
        

    return SumSize,CumulEnergies


#Ramsey signal
def IntIntRamsey(Nmax,InitialA,InitialJ,FinalA,Tmax,Spacing):
    """
    Gives the two-body Ramsey signal over time 
    for a quench between two non-zero finite scattering lengths

    I have been having a strange issue where the numeric
    integrator returns NaNs sometimes. I open an
    interactive session and import this module. Sometimes
    the integration in this function would return NaNs 
    sometimes not. It was consistent in a given session but
    not across sessions. 

    even though it's slower I'm switching to the analytic
    calculations
     


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


    Coeffs=np.zeros(Nmax)
    Exponentials=np.zeros(Nmax,dtype=complex)
    VInitial=0.5*(Energies(1,InitialA)[InitialJ]-1.5)
    V=0.5*(Energies(Nmax,FinalA)-1.5)

    Z0=np.pi*(special.digamma(-VInitial-0.5)-special.digamma(-VInitial))\
            *special.poch(-VInitial-1/2,3/2)/VInitial
    
    for counter in range(Nmax):
        #t1=ti.time()
        #Numeric calcualtion, agrees with analytic and is faster
        #but also doesn't work reliably which is odd because it
        #should be deterministic
        #NumericOverlap=4*np.pi*integrate.quad(lambda r: (r**2)*
        #IntWaveFunc(VInitial,r)*IntWaveFunc(V[counter],r),0, np.inf)[0]
        #Coeffs[counter]=NumericOverlap**2
        #t2=ti.time()

        #this is the analytic expression for the wavefunction overlaps
        #agrees with the numeric integration above ~15 decimal places
        #analytic expression takes orders of magnitude longer to evaluate
        #have not seen the analytic expression randomly fail like the numeric
        Zj=np.pi*(special.digamma(-V[counter]-0.5)-special.digamma(-V[counter]))\
            *special.poch(-V[counter]-1/2,3/2)/V[counter]
        AnalyticOverlap=np.sqrt(np.pi)*(2*VInitial*V[counter]*np.sqrt(Z0*Zj))**(-1)\
        *mpmath.hyp3f2(1.5,-V[counter],-VInitial,1-V[counter],1-VInitial,1)
        Coeffs[counter]=AnalyticOverlap**2
        #t3=ti.time()

        #print(Coeffs[counter])
        #print(Overlap**2)
        #print(Coeffs[counter]-SquareOverlap)
        #print("numeric time-",t2-t1)
        #print("analytic time",t3-t2)

    #counter1 is for time counter2 is for coefficients
    for counter1 in range(int(Tmax/Spacing)):
        for counter2 in range(Nmax):
            Exponentials[counter2]=np.exp(-2j*(V[counter2]-VInitial)*Tomega[counter1])


        S[counter1]=np.matmul(Coeffs,Exponentials)

    """
    title=r"Ramsey Signal $a_{s}$=" + str(InitialA) + r" ($\nu$="+str(round(VInitial,3))+") to $a_{s}$=" + str(FinalA)
    plt.figure(1)

    plt.rcParams['xtick.labelsize']=40
    plt.rcParams['ytick.labelsize']=40

    plt.suptitle(title,fontsize=35)

    ax1=plt.subplot(211)
    plt.plot(Tomega/np.pi, abs(S),'bo',markersize=5)
    plt.axis([0, Tmax/np.pi, 0, 1])
    plt.ylabel("$|S(t)|$",fontsize=35*1.5)
    plt.setp(ax1.get_xticklabels(),visible=False)

    ax2=plt.subplot(212)
    plt.plot(Tomega/np.pi, -np.angle(S)/(np.pi),'bo',markersize=5)
    plt.axis([0, Tmax/np.pi, -1, 1 ])
    plt.ylabel("$\phi(t)/\pi$",fontsize=35*1.5)
    plt.xlabel("t$\omega/\pi$",fontsize=35*1.5)

    plt.show()
    """
    return S, Tomega

def IntUnitRamsey(Nmax,InitialA,InitialJ,Tmax,Spacing):
    """
    Gives the two-body Ramsey signal over time for a quench
    from some arbitrary a_s to unitarity 

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

    #hmmm.... this calculation of the coefficients is
    Z0=np.pi*special.gamma(1-VInitial)*(special.digamma(-VInitial-0.5)-special.digamma(-VInitial))\
        /(VInitial*special.gamma(-VInitial-0.5))
    for j in range(Nmax):
        Zj=(np.pi**1.5)*special.binom(j-1/2,j)**-1
        Overlap=np.sqrt(np.pi)*(2*VInitial*(j-1/2)*np.sqrt(Z0*Zj))**(-1)\
        *mpmath.hyp3f2(1.5,1/2-j,-VInitial,3/2-j,1-VInitial,1)
        Coeffs[j]=Overlap**2

    #print(Coeffs)
    for counter1 in range(int(Tmax/Spacing)):
        for counter2 in range(Nmax):
            Exponentials[counter2]=np.exp(-2j*(counter2-0.5-VInitial)*Tomega[counter1])


        S[counter1]=np.matmul(Coeffs,Exponentials)

    """
    title=r"Ramsey Signal $a_{s}$=" + str(InitialA) + r" ($\nu$="+str(round(VInitial,3))+") to unitarity"
    plt.figure(1)

    plt.rcParams['xtick.labelsize']=40
    plt.rcParams['ytick.labelsize']=40

    plt.suptitle(title,fontsize=35)

    ax1=plt.subplot(211)
    plt.plot(Tomega/np.pi, abs(S),'bo',markersize=5)
    plt.axis([0, Tmax/np.pi, 0, 1])
    plt.ylabel("$|S(t)|$",fontsize=35*1.5)
    plt.setp(ax1.get_xticklabels(),visible=False)

    ax2=plt.subplot(212)
    plt.plot(Tomega/np.pi, -np.angle(S)/(np.pi),'bo',markersize=5)
    plt.axis([0, Tmax/np.pi, -1, 1 ])
    plt.ylabel("$\phi(t)/\pi$",fontsize=35*1.5)
    plt.xlabel("t$\omega/\pi$",fontsize=35*1.5)

    plt.show()
    """


    return S, Tomega

def IntNIRamsey(Nmax,InitialA,InitialJ,Tmax,Spacing):

    """
    Gives the two-body Ramsey signal over time for a quench
    from some arbitrary a_s to unitarity 

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

    V0=0.5*(Energies(InitialJ+1,InitialA)[InitialJ]-1.5)
    Z0=np.pi*special.gamma(1-V0)*(special.digamma(-V0-0.5)-special.digamma(-V0))/(V0*special.gamma(-V0-0.5))

    for n in range(Nmax):
        Overlap=(np.sqrt(n+1/2)/((n-V0)*np.sqrt(Z0)))*np.sqrt(np.sqrt(np.pi)*special.binom(n-1/2,n))

        Coeffs[n]=Overlap**2

    #print(Coeffs)
    for counter1 in range(int(Tmax/Spacing)):
        for counter2 in range(Nmax):
            Exponentials[counter2]=np.exp(-2j*(counter2-0.5-VInitial)*Tomega[counter1])


        S[counter1]=np.matmul(Coeffs,Exponentials)

    """
    title=r"Ramsey Signal $a_{s}$=" + str(InitialA) + r" ($\nu$="+str(round(VInitial,3))+") to NonInt"
    plt.figure(1)

    plt.rcParams['xtick.labelsize']=40
    plt.rcParams['ytick.labelsize']=40

    plt.suptitle(title,fontsize=35)

    ax1=plt.subplot(211)
    plt.plot(Tomega/np.pi, abs(S),'bo',markersize=5)
    plt.axis([0, Tmax/np.pi, 0, 1])
    plt.ylabel("$|S(t)|$",fontsize=35*1.5)
    plt.setp(ax1.get_xticklabels(),visible=False)

    ax2=plt.subplot(212)
    plt.plot(Tomega/np.pi, -np.angle(S)/(np.pi),'bo',markersize=5)
    plt.axis([0, Tmax/np.pi, -1, 1 ])
    plt.ylabel("$\phi(t)/\pi$",fontsize=35*1.5)
    plt.xlabel("t$\omega/\pi$",fontsize=35*1.5)

    plt.show()
    """

    return S, Tomega

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

    for counter1 in range(int(Tmax/Spacing)):
        for counter2 in range(Nmax):
            S[counter1]=S[counter1]+Coeffs[counter2]\
            *np.exp(-2j*(counter2-0.5-InitialJ)*np.pi*Tomega[counter1])

    """
    title=r"Ramsey Signal non-interacting ($n_{\rm i}=$"+str(InitialJ)+") to Unitarity"
    plt.figure(1)

    plt.rcParams['xtick.labelsize']=40
    plt.rcParams['ytick.labelsize']=40

    plt.suptitle(title,fontsize=35)

    ax1=plt.subplot(211)
    plt.plot(Tomega/np.pi, abs(S),'bo',markersize=5)
    plt.axis([0, Tmax/np.pi, 0, 1])
    plt.ylabel("$|S(t)|$",fontsize=35*1.5)
    plt.setp(ax1.get_xticklabels(),visible=False)

    ax2=plt.subplot(212)
    plt.plot(Tomega/np.pi, -np.angle(S)/(np.pi),'bo',markersize=5)
    plt.axis([0, Tmax/np.pi, -1, 1 ])
    plt.ylabel("$|\phi(t)|/\pi$",fontsize=35*1.5)
    plt.xlabel("t$\omega/\pi$",fontsize=35*1.5)

    plt.show()
    """

    return S, Tomega

def NIIntRamsey(Nmax,InitialJ,FinalA,Tmax,Spacing):
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

    """
    title=r"Ramsey Signal non-interacting ($n_{\rm i}=$"+str(InitialJ)+") to $a_{s}=$"+str(FinalA)
    plt.figure(1)

    plt.rcParams['xtick.labelsize']=40
    plt.rcParams['ytick.labelsize']=40

    plt.suptitle(title,fontsize=35)

    ax1=plt.subplot(211)
    plt.plot(Tomega/np.pi, abs(S),'bo',markersize=5)
    plt.axis([0, Tmax/np.pi, 0, 1])
    plt.ylabel("$|S(t)|$",fontsize=35*1.5)
    plt.setp(ax1.get_xticklabels(),visible=False)

    ax2=plt.subplot(212)
    plt.plot(Tomega/np.pi, -np.angle(S)/(np.pi),'bo',markersize=5)
    plt.axis([0, Tmax/np.pi, -1, 1 ])
    plt.ylabel("$\phi(t)/\pi$",fontsize=35*1.5)
    plt.xlabel("t$\omega/\pi$",fontsize=35*1.5)

    plt.show()
    """

    return S, Tomega

def UnitIntRamsey(Nmax,InitialJ,FinalA,Tmax,Spacing):
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

    for counter1 in range(int(Tmax/Spacing)):
        for counter2 in range(Nmax):
            Exponentials[counter2]=np.exp(-2j*(Vf[counter2]-InitialJ)*np.pi*Tomega[counter1])
            

        S[counter1]=np.matmul(Coeffs,Exponentials)

    """
    title=r"Ramsey Signal unitary ($n_{\rm i}=$"+str(InitialJ)+") to $a_{s}=$"+str(FinalA)
    plt.figure(1)

    plt.rcParams['xtick.labelsize']=40
    plt.rcParams['ytick.labelsize']=40

    plt.suptitle(title,fontsize=35)

    ax1=plt.subplot(211)
    plt.plot(Tomega/np.pi, abs(S),'bo',markersize=5)
    plt.axis([0, Tmax/np.pi, 0, 1])
    plt.ylabel("$|S(t)|$",fontsize=35*1.5)
    plt.setp(ax1.get_xticklabels(),visible=False)

    ax2=plt.subplot(212)
    plt.plot(Tomega/np.pi, -np.angle(S)/(np.pi),'bo',markersize=5)
    plt.axis([0, Tmax/np.pi, -1, 1 ])
    plt.ylabel("$\phi(t)/\pi$",fontsize=35*1.5)
    plt.xlabel("t$\omega/\pi$",fontsize=35*1.5)

    plt.show()
    """


    return S, Tomega

def UnitNIRamsey(Nmax,InitialJ,Tmax,Spacing):
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
        C=np.sqrt((counter2+0.5)*special.gamma(counter2+0.5)\
        *special.gamma(InitialJ+0.5))/(np.pi*(counter2-InitialJ+0.5)\
        *np.sqrt(special.gamma(InitialJ+1)*special.gamma(counter2+1)))
        Coeffs[counter2]=C**2

    for counter1 in range(int(Tmax/Spacing)):
        for counter2 in range(Nmax):
            S[counter1]=S[counter1]+Coeffs[counter2]\
            *np.exp(-2j*(counter2+0.5-InitialJ)*np.pi*Tomega[counter1])

    """
    title=r"Ramsey Signal Unitarity ($n_{\rm i}=$"+str(InitialJ)+") to non-interacting"
    plt.figure(1)

    plt.rcParams['xtick.labelsize']=40
    plt.rcParams['ytick.labelsize']=40

    plt.suptitle(title,fontsize=35)

    ax1=plt.subplot(211)
    plt.plot(Tomega/np.pi, abs(S),'bo',markersize=5)
    plt.axis([0, Tmax/np.pi, 0, 1])
    plt.ylabel("$|S(t)|$",fontsize=35*1.5)
    plt.setp(ax1.get_xticklabels(),visible=False)

    ax2=plt.subplot(212)
    plt.plot(Tomega/np.pi, -np.angle(S)/(np.pi),'bo',markersize=5)
    plt.axis([0, Tmax/np.pi, -1, 1 ])
    plt.ylabel("$|\phi(t)|/\pi$",fontsize=35*1.5)
    plt.xlabel("t$\omega/\pi$",fontsize=35*1.5)

    plt.show()
    """

    return S, Tomega

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

    Tomega=np.linspace(0,Tmax,int(Tmax/Spacing))

    S=np.zeros([int(Tmax/Spacing)],dtype=complex)

    if InitialA == 0:
        if FinalA == 0:
            print("NI to NI quench makes no sense")
            S[:]=1

        elif FinalA != "inf": 
            [S,Tomega]=NIIntRamsey(Nmax,InitialJ,FinalA,Tmax,Spacing)
        if FinalA== "inf":
            [S,Tomega]=NIUnitRamsey(Nmax,InitialJ,Tmax,Spacing)
        
    elif InitialA != "inf": 
        if FinalA == 0:
            [S,Tomega]=IntNIRamsey(Nmax,InitialA,InitialJ,Tmax,Spacing)
        elif FinalA != "inf":
            [S,Tomega]=IntIntRamsey(Nmax,InitialA,InitialJ,FinalA,Tmax,Spacing)
        if FinalA == "inf":
            [S,Tomega]=IntUnitRamsey(Nmax,InitialA,InitialJ,Tmax,Spacing)


    if InitialA == "inf":
        if FinalA == 0:
            [S,Tomega]=UnitNIRamsey(Nmax,InitialJ,Tmax,Spacing)
        elif FinalA != "inf":
            [S,Tomega]=UnitIntRamsey(Nmax,InitialJ,FinalA,Tmax,Spacing)
        if FinalA == "inf":
            print("unitary to unitary quench makes no sense")
            S[:]=1

    return S, Tomega


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

    V0=0.5*(Energies(InitialJ+1,InitialA)-1.5)[InitialJ]
    Z0=np.pi*special.gamma(1-V0)*(special.digamma(-V0-0.5)-special.digamma(-V0))/(V0*special.gamma(-V0-0.5))


    V0=0.5*(Energies(InitialJ+1,InitialA)[InitialJ]-1.5)
    Vf=0.5*(Energies(Nmax,FinalA)-1.5)

    Z0=np.pi*special.gamma(1-V0)*(special.digamma(-V0-0.5)-special.digamma(-V0))/(V0*special.gamma(-V0-0.5))

    for j in range(Nmax):
        Zj=np.pi*special.gamma(1-Vf[j])*(special.digamma(-Vf[j]-0.5)-special.digamma(-Vf[j]))/(Vf[j]*special.gamma(-Vf[j]-0.5))
        for k in range(j,Nmax):
            Zk=np.pi*special.gamma(1-Vf[k])*(special.digamma(-Vf[k]-0.5)-special.digamma(-Vf[k]))/(Vf[k]*special.gamma(-Vf[k]-0.5))

            OverlapJ=np.sqrt(np.pi)*(2*V0*Vf[j]*np.sqrt(Z0*Zj))**(-1)\
            *mpmath.hyp3f2(1.5,-Vf[j],-V0,1-Vf[j],1-V0,1)
            OverlapK=np.sqrt(np.pi)*(2*V0*Vf[k]*np.sqrt(Z0*Zk))**(-1)\
            *mpmath.hyp3f2(1.5,-Vf[k],-V0,1-Vf[k],1-V0,1)

            CrossTerm=0
            #this Size=k+10 line is a test for efficiency/ensuring convergence
            Size=k+10
            for n in range(Size):
                for m in range(Size):
                    CrossTerm=CrossTerm+((-1)**(m+n))*special.gamma(m+1.5)*special.gamma(n+1.5)\
                    /((m-Vf[j])*(n-Vf[k])*special.gamma(n+1)*special.gamma(m-n+1.5)*special.gamma(m+1)*special.gamma(n-m+1.5))

            CrossTerm=(1/np.sqrt(Zk*Zj))*CrossTerm

            Coeffs[j,k]=OverlapJ*OverlapK*CrossTerm

            Coeffs[k,j]=Coeffs[j,k]
        #print(str(+1)+"/"+str(Nmax) )


    for t in range(int(Tmax/Spacing)):
        for j in range(Nmax):
            for k in range(Nmax):
                R[t]=R[t]+np.real(Coeffs[j,k]*np.exp(-2j*(Vf[k]-Vf[j])*np.pi*Tomega[t]))

    #print(AnalyticCoeffs)
    """
    title=r"$\langle\tilde{r}\rangle$ of $a_{s}=$" + str(InitialA) + " to $a_{s}=$" + str(FinalA)
    plt.figure(1)
    plt.suptitle(title,fontsize=35*1.5)
    plt.rcParams['xtick.labelsize']=30
    plt.rcParams['ytick.labelsize']=30
    plt.plot(Tomega/np.pi, R,'bo',markersize=5)
    plt.axis([0, Tmax/np.pi, 0, 1.1*max(R)])
    plt.ylabel(r"$\langle\tilde{r}\rangle$",fontsize=35*1.5)
    plt.xlabel(r"t$\omega/\pi$",fontsize=35*1.5)
    plt.show()
    """
    return R, Tomega

def IntUnitExpectR(Nmax,InitialA,InitialJ,Tmax,Spacing):
    """
    This function gives the expectation value of r over time after
    a quench from some non-zero interaction strength to unitarity

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

    for j in range(Nmax):
        Zj=(np.pi**2)*special.gamma(j+1)/special.gamma(j+1/2)
        OverlapJ=np.sqrt(np.pi)*(2*V0*(j-1/2)*np.sqrt(Z0*Zj))**(-1)\
        *mpmath.hyp3f2(1.5,1/2-j,-V0,3/2-j,1-V0,1)
        for k in range(j,Nmax):
            Zk=(np.pi**2)*special.gamma(k+1)/special.gamma(k+1/2)
            OverlapK=np.sqrt(np.pi)*(2*V0*(k-1/2)*np.sqrt(Z0*Zk))**(-1)\
            *mpmath.hyp3f2(1.5,1/2-k,-V0,3/2-k,1-V0,1)

            CrossTerm=0
            #The cross term itself has to be evaluated as a double sum.
            #testing shows the double sum converges by a cut off of k+10
            for n in range(k+10):
                for m in range(k+10):
                    CrossTerm=CrossTerm+((-1)**(m+n))*special.gamma(m+1.5)*special.gamma(n+1.5)\
                    /((m-j+1/2)*(n-k+1/2)*special.gamma(n+1)*special.gamma(m-n+1.5)*special.gamma(m+1)*special.gamma(n-m+1.5))

            CrossTerm=(1/np.sqrt(Zk*Zj))*CrossTerm
            Coeffs[j,k]=OverlapJ*OverlapK*CrossTerm
            Coeffs[k,j]=Coeffs[j,k]


    for t in range(int(Tmax/Spacing)):
        for j in range(Nmax):
            for k in range(Nmax):
                R[t]=R[t]+np.real(Coeffs[j,k]*np.exp(-2j*(k-j)*np.pi*Tomega[t]))
    """
    title=r"$\langle\tilde{r}\rangle$ of $a_{s}=$" + str(InitialA) + " to unitarity"
    plt.figure(1)
    plt.suptitle(title,fontsize=35*1.5)

    plt.rcParams['xtick.labelsize']=30
    plt.rcParams['ytick.labelsize']=30
    plt.plot(Tomega/np.pi, R,'bo',markersize=5)
    plt.axis([0, Tmax/np.pi, 0, 1.1*max(R)])
    plt.ylabel(r"$\langle\tilde{r}\rangle$",fontsize=35*1.5)
    plt.xlabel(r"t$\omega/\pi$",fontsize=35*1.5)

    plt.show()
    """

    return R, Tomega

def IntNIExpectR(Nmax,InitialA,InitialJ,Tmax,Spacing):
    """
    This function gives the expectation value of r over time after
    a quench from some non-zero interaction strength to the 
    non-interacting limit

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

    for n in range(Nmax):
        OverlapN=(((n-V0)*np.sqrt(Z0))**(-1))*np.sqrt(special.gamma(n+3/2)/special.gamma(n+1))
        for m in range(n,Nmax):
            OverlapM=(((m-V0)*np.sqrt(Z0))**(-1))*np.sqrt(special.gamma(m+3/2)/special.gamma(m+1))
            CrossTerm=((-1)**(m+n))*np.sqrt(special.gamma(3/2+m)*special.gamma(3/2+n)/(special.gamma(1+m)*special.gamma(1+n)))\
            /(special.gamma(m-n+3/2)*special.gamma(n-m+3/2))

            Coeffs[n,m]=OverlapM*OverlapN*CrossTerm
            Coeffs[m,n]=Coeffs[n,m]

    for t in range(int(Tmax/Spacing)):
        for n in range(Nmax):
            for m in range(Nmax):
                R[t]=R[t]+np.real(Coeffs[n,m]*np.exp(-2j*(n-m)*np.pi*Tomega[t]))
    """
    title=r"$\langle\tilde{r}\rangle$ of $a_{s}=$" + str(InitialA) + " to non-interacting"
    plt.figure(1)
    plt.suptitle(title,fontsize=35*1.5)
    plt.rcParams['xtick.labelsize']=30
    plt.rcParams['ytick.labelsize']=30
    plt.plot(Tomega/np.pi, R,'r',linewidth=10)
    plt.axis([0, Tmax/np.pi, 0, 1.1*max(R)])
    plt.ylabel(r"$\langle\tilde{r}\rangle$",fontsize=35*1.5)
    plt.xlabel(r"t$\omega/\pi$",fontsize=35*1.5)
    plt.show()
    """

    return None

def UnitIntExpectR(Nmax,InitialJ,FinalA,Tmax,Spacing):
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

    V0=InitialJ-1/2
    Vf=0.5*(Energies(Nmax,FinalA)-1.5)

    Z0=(np.pi**2)*special.gamma(InitialJ+1)/special.gamma(InitialJ+1/2)

    for j in range(Nmax):
        Zj=np.pi*special.gamma(1-Vf[j])*(special.digamma(-Vf[j]-0.5)-special.digamma(-Vf[j]))/(Vf[j]*special.gamma(-Vf[j]-0.5))
        for k in range(j,Nmax):
            Zk=np.pi*special.gamma(1-Vf[k])*(special.digamma(-Vf[k]-0.5)-special.digamma(-Vf[k]))/(Vf[k]*special.gamma(-Vf[k]-0.5))

            OverlapJ=np.sqrt(np.pi)*(2*V0*Vf[j]*np.sqrt(Z0*Zj))**(-1)\
            *mpmath.hyp3f2(1.5,-Vf[j],-V0,1-Vf[j],1-V0,1)
            OverlapK=np.sqrt(np.pi)*(2*V0*Vf[k]*np.sqrt(Z0*Zk))**(-1)\
            *mpmath.hyp3f2(1.5,-Vf[k],-V0,1-Vf[k],1-V0,1)

            CrossTerm=0
            #this Size=k+10 line is a test for efficiency/ensuring convergence
            Size=k+10
            for n in range(Size):
                for m in range(Size):
                    CrossTerm=CrossTerm+((-1)**(m+n))*special.gamma(m+1.5)*special.gamma(n+1.5)\
                    /((m-Vf[j])*(n-Vf[k])*special.gamma(n+1)*special.gamma(m-n+1.5)*special.gamma(m+1)*special.gamma(n-m+1.5))

            CrossTerm=(1/np.sqrt(Zk*Zj))*CrossTerm
            Coeffs[j,k]=OverlapJ*OverlapK*CrossTerm
            Coeffs[k,j]=Coeffs[j,k]


    for t in range(int(Tmax/Spacing)):
        for j in range(Nmax):
            for k in range(Nmax):
                R[t]=R[t]+np.real(Coeffs[j,k]*np.exp(-2j*(k-j)*np.pi*Tomega[t]))

    """
    title=r"$\langle\tilde{r}\rangle$ of unitarity to $a_{s}=$" + str(FinalA)
    plt.figure(1)
    plt.suptitle(title,fontsize=35*1.5)

    plt.rcParams['xtick.labelsize']=30
    plt.rcParams['ytick.labelsize']=30
    plt.plot(Tomega/np.pi, R,'bo',markersize=5)
    plt.axis([0, Tmax/np.pi, 0, 1.1*max(R)])
    plt.ylabel(r"$\langle\tilde{r}\rangle$",fontsize=35*1.5)
    plt.xlabel(r"t$\omega/\pi$",fontsize=35*1.5)

    plt.show()
    """

    return R, Tomega

def UnitNIExpectR(Nmax,InitialJ,Tmax,Spacing):
    """
    This function gives the expectation value of r over time in the unitary to
    non interacting case


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

    #here InitialN is the number determining the interacting pseudo quantum number
    #and j and k are the principle quantum numbers for the non-interacting wavefunc

    for j in range(Nmax):
        for k in range(j,Nmax):
            Coeffs[j,k]=((-1)**(j+k))*np.sqrt(special.gamma(j+1.5)*special.gamma(k+1.5))/(np.sqrt(special.gamma(1+j)*special.gamma(1+k))*special.gamma(j-k+1.5)*special.gamma(k-j+1.5))\
            *special.gamma(InitialJ+0.5)*np.sqrt(special.gamma(j+1.5)*special.gamma(k+1.5))/((np.pi**2)*(j-InitialJ+0.5)*(k-InitialJ+0.5)*special.gamma(InitialJ+1)*np.sqrt(special.gamma(j+1)*special.gamma(k+1)))
            Coeffs[k,j]=Coeffs[j,k]

    for counter3 in range(int(Tmax/Spacing)):
        for counter1 in range(Nmax):
            for counter2 in range(Nmax):
                R[counter3]=R[counter3]+Coeffs[counter1,counter2]*np.exp(-2j*(counter2-counter1)*np.pi*Tomega[counter3])

    """
    #print(AnalyticCoeffs)
    fig=plt.figure()
    ax=plt.axes()
    ax.tick_params(axis='x', labelsize=40)
    ax.tick_params(axis='y', labelsize=40)


    plt.plot(Tomega, R,'k',markersize=5)
    plt.axis([0, Tmax, 0, 1.1*max(R)])
    plt.ylabel(r"$\langle$r/$a_{\mu}$ $\rangle$",fontsize=52.5)
    plt.xlabel(r"$\omega$ t/$\pi$",fontsize=52.5)

    plt.show()
    """

    return R, Tomega

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

    for j in range(Nmax):
        Zj=np.pi*special.gamma(1-Vf[j])*(special.digamma(-Vf[j]-0.5)-special.digamma(-Vf[j]))/(Vf[j]*special.gamma(-Vf[j]-0.5))
        OverlapJ=(((InitialJ-Vf[j])*np.sqrt(Zj))**(-1))*np.sqrt(special.gamma(InitialJ+3/2)/special.gamma(InitialJ+1))
        for k in range(j,Nmax):
            Zk=np.pi*special.gamma(1-Vf[k])*(special.digamma(-Vf[k]-0.5)-special.digamma(-Vf[k]))/(Vf[k]*special.gamma(-Vf[k]-0.5))
            OverlapK=(((InitialJ-Vf[k])*np.sqrt(Zk))**(-1))*np.sqrt(special.gamma(InitialJ+3/2)/special.gamma(InitialJ+1))


            CrossTerm=0
            #this Size=k+10 line is a test for efficiency/ensuring convergence
            Size=k+10
            for n in range(Size):
                for m in range(Size):
                    CrossTerm=CrossTerm+((-1)**(m+n))*special.gamma(m+1.5)*special.gamma(n+1.5)\
                    /((m-Vf[j])*(n-Vf[k])*special.gamma(n+1)*special.gamma(m-n+1.5)*special.gamma(m+1)*special.gamma(n-m+1.5))

            CrossTerm=(1/np.sqrt(Zk*Zj))*CrossTerm

            Coeffs[j,k]=OverlapJ*OverlapK*CrossTerm
            Coeffs[k,j]=Coeffs[j,k]

    for t in range(int(Tmax/Spacing)):
        for j in range(Nmax):
            for k in range(Nmax):
                R[t]=R[t]+np.real(Coeffs[j,k]*np.exp(-2j*(Vf[k]-Vf[j])*np.pi*Tomega[t]))

    """
    title=r"$\langle\tilde{r}\rangle$ of non-interacting to $a_{s}=$" + str(FinalA)
    plt.figure(1)
    plt.suptitle(title,fontsize=35*1.5)
    plt.rcParams['xtick.labelsize']=30
    plt.rcParams['ytick.labelsize']=30
    plt.plot(Tomega, R,'bo',markersize=5)
    plt.axis([0, Tmax, 0, 1.1*max(R)])
    plt.ylabel(r"$\langle\tilde{r}\rangle$",fontsize=35*1.5)
    plt.xlabel(r"t$\omega/\pi$",fontsize=35*1.5)
    plt.show()
    """

    return R, Tomega

def NIUnitExpectR(Nmax,InitialJ,Tmax,Spacing):
    """
    This function gives the expectation value of r over time in the non-int to
    unitary case, using the analytic evaluations of the integrals. This is the
    one that converges no problem

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


    for counter3 in range(int(Tmax/Spacing)):
        for counter1 in range(Nmax):
            for counter2 in range(Nmax):
                R[counter3]=R[counter3]+Coeffs[counter1,counter2]*np.exp(-2j*(counter1-counter2)*np.pi*Tomega[counter3])


    """
    fig=plt.figure()
    ax=plt.axes()
    
    ax.tick_params(axis='x', labelsize=40)
    ax.tick_params(axis='y', labelsize=40)

    plt.title(r"Non-Interacting ($n_{i}=$"+str(InitialJ)+") to Unitarity",fontsize=52.5)
    plt.plot(Tomega, R,'k',linewidth=10)
    plt.axis([0, Tmax, 0, 1.1*max(R)])
    plt.ylabel(r"$\langle r(t)/a_{\mu}\rangle$",fontsize=52.5)
    plt.xlabel(r"$\omega $t/$\pi$",fontsize=52.5)
    plt.show()
    """

    return R, Tomega

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

    Tomega=np.linspace(0,Tmax,int(Tmax/Spacing))
    R=np.zeros([int(Tmax/Spacing)],dtype=complex)

    if InitialA == 0:
        if FinalA == 0:
            print("NI to NI quench makes no sense")
            R[:]=0

        elif FinalA != "inf": 
            [R,Tomega]=NIIntExpectR(Nmax,InitialJ,FinalA,Tmax,Spacing)
        if FinalA== "inf":
            [R,Tomega]=NIUnitExpectR(Nmax,InitialJ,Tmax,Spacing)
        
    elif InitialA != "inf": 
        if FinalA == 0:
            [R,Tomega]=IntNIExpectR(Nmax,InitialA,InitialJ,Tmax,Spacing)
        elif FinalA != "inf":
            [R,Tomega]=IntIntExpectR(Nmax,InitialA,InitialJ,FinalA,Tmax,Spacing)
        if FinalA == "inf":
            [R,Tomega]=IntUnitExpectR(Nmax,InitialA,InitialJ,Tmax,Spacing)


    if InitialA == "inf":
        if FinalA == 0:
            [R,Tomega]=UnitNIExpectR(Nmax,InitialJ,Tmax,Spacing)
        elif FinalA != "inf":
            [R,Tomega]=UnitIntExpectR(Nmax,InitialJ,FinalA,Tmax,Spacing)
        if FinalA == "inf":
            print("unitary to unitary quench makes no sense")
            R[:]=0


    return R, Tomega


"""
Functions to make plots
"""

def PlotEnergySpectrum(Levels,Hmax,Spacing):
    """
    Gives a graph of the two-body energy spectrum, 
    
    energy (in \hbar\omega) on the y-axis and
    interaction strength (a_s/a_rel) on the x-axis

    Parameters
    ----------------
    Levels : a positive integer
        the number of energy levels returned (there are infinite, this
        function returns the first "Levels" of them)
    Hmax : a float
        Hmax is the horizontal limits
    spacing : a float
        spacing is the horizontal spacing between points



    """

    #arrays to hold the x-values
    #convenient to split into positive and negative
    #because of the positive energy bound state
    Apos=np.linspace(Spacing**3,Hmax,int(Hmax/Spacing))
    Aneg=np.linspace(-1*Spacing**3,-1*Hmax,int(Hmax/Spacing))

    #arrays to hold the y-values
    Epos=np.zeros([int(Hmax/Spacing),Levels])
    Eneg=np.zeros([int(Hmax/Spacing),Levels-1])

    for counter in range(int(Hmax/Spacing)):
        Epos[counter,:]=Energies(Levels,Apos[counter])

    for counter in range(int(Hmax/Spacing)):
        Eneg[counter,:]=Energies(Levels-1,Aneg[counter])

    fig= plt.figure()
    ax= plt.axes()

    #fig.patch.set_facecolor('xkcd:light grey')
    #ax.patch.set_facecolor('xkcd:light grey')
    #ax.tick_params(axis='x', labelsize=60)
    #ax.tick_params(axis='y', labelsize=60)

    plt.plot(Apos, Epos,'b',markersize=5)
    plt.plot(Aneg, Eneg,'b',markersize=5)
    for i in range(Levels):
        plt.axhline(2*(i-0.5)+1.5, color='r')

    plt.axis([-1*Hmax, Hmax, -5, 2*Levels-1.5])
    title="Two-Body Energy Spectrum"
    fontsize=15
    plt.title(title, fontsize=1.5*fontsize)
    plt.ylabel(r"$E_{\rm rel} \: (\hbar \omega)$",fontsize=fontsize)
    plt.xlabel(r"$a_{\rm s} \: (a_{\rm \mu})$",fontsize=fontsize)
    plt.show()

    return None

def PlotEnergyExpect(Nmax,InitialA,InitialJ,FinalA):
    """
    Gives a graph of the energy expectation of the post-quench
    state for a general quench in the zero range two-body 
    interacting

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

    """

    [SumSize,CumulEnergies]=QuenchExpectE(Nmax,InitialA,InitialJ,FinalA)

    if InitialA == 0:
        if FinalA == 0:
            title="NI to NI quench." + r" $n_{\rm i}$="+str(InitialJ)
        elif FinalA != "inf": #i.e. if FinalA if a non-zero finite number
            title=r"NI to $a_{s}=$" +str(FinalA)+ " quench." + r" $n_{\rm i}$="+str(InitialJ) 
        if FinalA== "inf":
            title=r"NI to unitarity quench." + r" $n_{\rm i}$="+str(InitialJ)

        
    elif InitialA != "inf": #i.e. if InitialA if a non-zero finite number
        if FinalA == 0:
            title=r"$a_{s}=$"+str(InitialA)+" to NI quench." + r" $n_{\rm i}$="+str(InitialJ)
        elif FinalA != "inf": #i.e. if FinalA if a non-zero finite number
            title=r"$a_{s}=$"+str(InitialA)+" to $a_{s}=$" +str(FinalA)+ " quench." + r" $n_{\rm i}$="+str(InitialJ) 
        if FinalA== "inf":
            title=r"$a_{s}=$"+str(InitialA)+" to unitarity quench." + r" $n_{\rm i}$="+str(InitialJ) 



    if InitialA == "inf":
        if FinalA == 0:
            title="Unitarity to NI quench." + r" $n_{\rm i}$="+str(InitialJ) 
        elif FinalA != "inf": #i.e. if FinalA if a non-zero finite number
            title=r"Unitarity to $a_{s}=$" +str(FinalA)+ " quench." + r" $n_{\rm i}$="+str(InitialJ) 
        if FinalA== "inf":
            title=r"Unitarity to unitarity quench." + r" $n_{\rm i}$="+str(InitialJ) 
            
    plt.figure(1)
    plt.suptitle(title,fontsize=35*1.5)

    plt.rcParams['xtick.labelsize']=30
    plt.rcParams['ytick.labelsize']=30
    plt.plot(SumSize, CumulEnergies,'bo',markersize=5)
    plt.axhline(CumulEnergies[Nmax-1],color='b')


    if CumulEnergies[Nmax-1]>=0:
        plt.axis([0, Nmax, 0, max(1.1*float(CumulEnergies[0]),1.1*float(CumulEnergies[Nmax-1]))])
    if CumulEnergies[Nmax-1]<0:
        plt.axis([0, Nmax, min(1.1*float(CumulEnergies[0]),1.1*float(CumulEnergies[Nmax-1])), 0])

    plt.ylabel(r"$\langle E \rangle $",fontsize=35*1.5)
    plt.xlabel(r"$N_{\rm max}$",fontsize=35*1.5)

    plt.show()

    

    return None

def PlotRamseySignal(Nmax,InitialA,InitialJ,FinalA,Tmax,Spacing):
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

    [S,Tomega]=RamseySignal(Nmax,InitialA,InitialJ,FinalA,Tmax,Spacing)

    if InitialA == 0:
        if FinalA == 0:
            title="NI to NI quench." + r" $n_{\rm i}$="+str(InitialJ)
        elif FinalA != "inf": #i.e. if FinalA if a non-zero finite number
            title=r"NI to $a_{s}=$" +str(FinalA)+ " quench." + r" $n_{\rm i}$="+str(InitialJ) 
        if FinalA== "inf":
            title=r"NI to unitarity quench." + r" $n_{\rm i}$="+str(InitialJ)

        
    elif InitialA != "inf": #i.e. if InitialA if a non-zero finite number
        if FinalA == 0:
            title=r"$a_{s}=$"+str(InitialA)+" to NI quench." + r" $n_{\rm i}$="+str(InitialJ)
        elif FinalA != "inf": #i.e. if FinalA if a non-zero finite number
            title=r"$a_{s}=$"+str(InitialA)+" to $a_{s}=$" +str(FinalA)+ " quench." + r" $n_{\rm i}$="+str(InitialJ) 
        if FinalA== "inf":
            title=r"$a_{s}=$"+str(InitialA)+" to unitarity quench." + r" $n_{\rm i}$="+str(InitialJ) 



    if InitialA == "inf":
        if FinalA == 0:
            title="Unitarity to NI quench." + r" $n_{\rm i}$="+str(InitialJ) 
        elif FinalA != "inf": #i.e. if FinalA if a non-zero finite number
            title=r"Unitarity to $a_{s}=$" +str(FinalA)+ " quench." + r" $n_{\rm i}$="+str(InitialJ) 
        if FinalA== "inf":
            title=r"Unitarity to unitarity quench." + r" $n_{\rm i}$="+str(InitialJ) 
            
    plt.figure(1)

    plt.rcParams['xtick.labelsize']=40
    plt.rcParams['ytick.labelsize']=40

    plt.suptitle(title,fontsize=35)

    ax1=plt.subplot(211)
    plt.plot(Tomega/np.pi, abs(S),'bo',markersize=5)
    plt.axis([0, Tmax/np.pi, 0, 1])
    plt.ylabel("$|S(t)|$",fontsize=35*1.5)
    plt.setp(ax1.get_xticklabels(),visible=False)

    ax2=plt.subplot(212)
    plt.plot(Tomega/np.pi, -np.angle(S)/(np.pi),'bo',markersize=5)
    plt.axis([0, Tmax/np.pi, -1, 1 ])
    plt.ylabel("$\phi(t)/\pi$",fontsize=35*1.5)
    plt.xlabel("t$\omega/\pi$",fontsize=35*1.5)

    plt.show()


    return None

def PlotExpectR(Nmax,InitialA,InitialJ,FinalA,Tmax,Spacing):
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
    R : a 1 x int(Tmax/spacing) array
        the particle separation expectation
        as a function of time
    Tomega : a 1 x int(Tmax/spacing) array
        the time
    """

    [R,Tomega]=ExpectR(Nmax,InitialA,InitialJ,FinalA,Tmax,Spacing)

    if InitialA == 0:
        if FinalA == 0:
            title="NI to NI quench." + r" $n_{\rm i}$="+str(InitialJ)
        elif FinalA != "inf": #i.e. if FinalA if a non-zero finite number
            title=r"NI to $a_{s}=$" +str(FinalA)+ " quench." + r" $n_{\rm i}$="+str(InitialJ) 
        if FinalA== "inf":
            title=r"NI to unitarity quench." + r" $n_{\rm i}$="+str(InitialJ)

        
    elif InitialA != "inf": #i.e. if InitialA if a non-zero finite number
        if FinalA == 0:
            title=r"$a_{s}=$"+str(InitialA)+" to NI quench." + r" $n_{\rm i}$="+str(InitialJ)
        elif FinalA != "inf": #i.e. if FinalA if a non-zero finite number
            title=r"$a_{s}=$"+str(InitialA)+" to $a_{s}=$" +str(FinalA)+ " quench." + r" $n_{\rm i}$="+str(InitialJ) 
        if FinalA== "inf":
            title=r"$a_{s}=$"+str(InitialA)+" to unitarity quench." + r" $n_{\rm i}$="+str(InitialJ) 


    if InitialA == "inf":
        if FinalA == 0:
            title="Unitarity to NI quench." + r" $n_{\rm i}$="+str(InitialJ) 
        elif FinalA != "inf": #i.e. if FinalA if a non-zero finite number
            title=r"Unitarity to $a_{s}=$" +str(FinalA)+ " quench." + r" $n_{\rm i}$="+str(InitialJ) 
        if FinalA== "inf":
            title=r"Unitarity to unitarity quench." + r" $n_{\rm i}$="+str(InitialJ) 
            
    plt.figure(1)

    fig=plt.figure()
    ax=plt.axes()
    
    ax.tick_params(axis='x', labelsize=40)
    ax.tick_params(axis='y', labelsize=40)

    plt.title(r"Non-Interacting ($n_{i}=$"+str(InitialJ)+") to Unitarity",fontsize=52.5)
    plt.plot(Tomega, R,'k',linewidth=10)
    plt.axis([0, Tmax, 0, 1.1*max(R)])
    plt.ylabel(r"$\langle r(t)/a_{\mu}\rangle$",fontsize=52.5)
    plt.xlabel(r"$\omega $t/$\pi$",fontsize=52.5)
    plt.show()


    plt.show()


    return None

