import numpy as np
import mpmath as mpmath
import scipy.special as special
import scipy.integrate as integrate
import scipy.stats as stats
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import time as ti
from matplotlib import rc

import TwoBodyZeroRan as TBZR



def IntWaveFuncNormCheck(Min,Max,Spac,Tol=10e-6):
    """
    This tests TBZR.IntWaveFunc across a range of 
    eigenvalues to see if the normalisation is
    correct

    Parameters
    -----------------
    Min: a real float
        the minimim value of v we sweep across

    Max: a real float
        the minimim value of v we sweep across
        must be larger than Min

    Spac: a real positive float
        the spacing in the sweep.     

    Tol: a positive float
        desired tolerance for normalisation accuracy

    """

    if Min>=Max:
        print("ERROR: Min is equal to or greater than Max")
        return None

    v=np.linspace(Min-min(Spac**3,Spac**-3),Max+min(Spac**1.5,Spac**-1.5),int((Max-Min)/Spac**1.1))
    #shift things around to try and ensure we don't land on any positive integers

    Error=0
    for counter in range(len(v)):
        x=4*np.pi*integrate.quad(lambda r: (r**2)*TBZR.IntWaveFunc(v[counter],r)**2,0, np.inf)[0]
        if abs(1-x)>Tol:
            print("PROBLEM")
            print("for v="+str(v[counter])+" Tol="+str(Tol)+" is exceeded")
            print("normalises to "+str(x))
            Error=1
        if mpmath.isnan(x):
            print("PROBLEM")
            print("for v="+str(v[counter])+" we get a nan")
            Error=1
    if Error==0:
        print("between v="+str(v[0])+" and"+" v="+str(v[-1])+" the interacting wavefunction is properly normalised to within a tolerance of "+str(Tol))

    return None

def NIWaveNormCheck(Min,Max,Tol=10e-6):
    """
    This tests TBZR.NIWaveFunc across a range of 
    eigenvalues to see if the normalisation is
    correct

    Parameters
    -----------------
    Min: a nonnegative int
        the minimim value of n we sweep from

    Max: a positive int
        the minimim value of n we sweep to
        must be larger than Min

    Tol: a positive float
        desired tolerance for normalisation accuracy

    
    """

    if Min>=Max:
        print("ERROR: Min is equal to or greater than Max")
        return None
    if Min<0:
        print("ERROR: Min<0")
        return None

    Error=0
    for counter in range(Min,Max+1):
        x=4*np.pi*integrate.quad(lambda r: (r**2)*TBZR.NIWaveFunc(counter,r)**2,0, np.inf)[0]
        if abs(1-x)>Tol:
            print("PROBLEM")
            print("for n="+str(counter)+" Tol="+str(Tol)+" is exceeded")
            print("normalises to "+str(x))
            Error=1
        if mpmath.isnan(x):
            print("PROBLEM")
            print("for n="+str(counter)+" we get a nan")
            Error=1
    if Error==0:
        print("between n="+str(Min)+" and"+" v="+str(Max)+" the non-interacting wavefunction is properly normalised to within a tolerance of "+str(Tol))


    return None



def PlotEnergySpectrum(Levels,Hmax,Spacing):
    """
    Gives a graph of the two-body energy spectrum, 
    
    energy (in \hbar\omega) on the y-axis and
    interaction strength (a_s/a_rel) on the x-axis

    This is intended to make an illustrative plot rather than
    to make something that goes into a paper

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
        Epos[counter,:]=TBZR.Energies(Levels,Apos[counter])

    for counter in range(int(Hmax/Spacing)):
        Eneg[counter,:]=TBZR.Energies(Levels-1,Aneg[counter])

    fig= plt.figure()
    ax= plt.axes()

    #fig.patch.set_facecolor('xkcd:light grey')
    #ax.patch.set_facecolor('xkcd:light grey')
    #ax.tick_params(axis='x', labelsize=60)
    #ax.tick_params(axis='y', labelsize=60)

    plt.plot(Apos, Epos,'bo',markersize=5)
    plt.plot(Aneg, Eneg,'bo',markersize=5)
    for i in range(Levels):
        plt.axhline(2*(i-0.5)+1.5, color='r')

    plt.axis([-1*Hmax, Hmax, -5, 2*Levels-1.5])
    title="Two-Body Energy Spectrum"
    fontsize=35
    plt.title(title, fontsize=1.5*fontsize)
    ax.tick_params(axis='both',labelsize=15)
    plt.ylabel(r"$E_{\rm rel}/\hbar \omega$",fontsize=fontsize)
    plt.xlabel(r"$a_{\rm s}/a_{\rm \mu}$",fontsize=fontsize)
    plt.show()

    return None

def PlotEnergyExpect(Nmax,InitialA,InitialJ,FinalA):
    """
    Gives a graph of the energy expectation of the post-quench
    state for an arbitrary quench 

    Y-axis is <E>, X-axis is the number of terms in sum
    
    This is intended to make an illustrative plot rather than
    to make something that goes into a paper

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


    [SumSize,CumulEnergies]=TBZR.QuenchExpectE(Nmax,InitialA,InitialJ,FinalA)

    #UnitNI
    if InitialA=="inf" and FinalA==0:
        title=r"$\langle E \rangle$ for a Unitarity to NI quench." + r" $n_{\rm i}$="+str(InitialJ)
    
    #NIUnit
    if InitialA==0 and FinalA=="inf":
        title=r"$\langle E \rangle$ for a NI to Unitary quench." + r" $n_{\rm i}$="+str(InitialJ)

    #IntInt
    if InitialA!=0 and InitialA!="inf" and FinalA!=0 and FinalA!="inf":
        title=r"$\langle E \rangle$ for a "+"$a_{s}=$"+str(InitialA)+" to $a_{s}=$" +str(FinalA)+ " quench." + r" $n_{\rm i}$="+str(InitialJ) 

    #IntNI
    if InitialA!=0 and InitialA!="inf" and FinalA==0:
        title=r"$\langle E \rangle$ for a "+"$a_{s}=$"+str(InitialA)+" to NI quench." + r" $n_{\rm i}$="+str(InitialJ)

    #NIInt
    if InitialA==0 and FinalA!=0 and FinalA!="inf":
        title=r"$\langle E \rangle$ for a NI to $a_{s}=$" +str(FinalA)+ " quench." + r" $n_{\rm i}$="+str(InitialJ) 

    #IntUnit
    if InitialA!=0 and InitialA!="inf" and FinalA=="inf":
        title=r"$\langle E \rangle$ for a "+"$a_{s}=$"+str(InitialA)+" to unitary quench." + r" $n_{\rm i}$="+str(InitialJ) 

    #UnitInt
    if InitialA=="inf" and FinalA!=0 and FinalA!="inf":
        title=r"$\langle E \rangle$ for a unitary to $a_{s}=$" +str(FinalA)+ " quench." + r" $n_{\rm i}$="+str(InitialJ) 

    #UnitUnit
    if InitialA=="inf" and FinalA=="inf":
        title=r"$\langle E \rangle$ for a Unitary to Unitary quench. (Unphysical)"

    #NINI
    if InitialA==0 and FinalA==0:
        title=r"$\langle E \rangle$ for a NI to NI quench. (Unphysical)"

    #Int to Same Int
    if InitialA==FinalA and InitialA!=0 and InitialA!="inf":
        title=r"$\langle E \rangle$ for a "+"$a_{s}=$"+str(InitialA)+" to $a_{s}=$" +str(FinalA)+ " quench. (Unphysical)" + r" $n_{\rm i}$="+str(InitialJ)
    
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
    Gives the Ramsey signal over time for an arbitrary quench

    Y-axis is Ramsey Signal, X-axis is time in units of 
    \pi/\omega

    This is intended to make an illustrative plot rather than
    to make something that goes into a paper

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
        in units of \pi/\omega
    spacing : a small positive float
        the timestep size
        

    Returns
    -------------------
    S : a 1 x int(Tmax/spacing) array
        the Ramsey Signal as a function of time
    Tomega : a 1 x int(Tmax/spacing) array
        the time
    """

    [Tomega,S]=TBZR.RamseySignal(Nmax,InitialA,InitialJ,FinalA,Tmax,Spacing)

    #UnitNI
    if InitialA=="inf" and FinalA==0:
        title=r"Ramsey Signal for a Unitarity to NI quench." + r" $n_{\rm i}$="+str(InitialJ)
    
    #NIUnit
    if InitialA==0 and FinalA=="inf":
        title=r"Ramsey Signal for a NI to Unitary quench." + r" $n_{\rm i}$="+str(InitialJ)

    #IntInt
    if InitialA!=0 and InitialA!="inf" and FinalA!=0 and FinalA!="inf":
        title=r"Ramsey Signal for a "+"$a_{s}=$"+str(InitialA)+" to $a_{s}=$" +str(FinalA)+ " quench." + r" $n_{\rm i}$="+str(InitialJ) 

    #IntNI
    if InitialA!=0 and InitialA!="inf" and FinalA==0:
        title=r"Ramsey Signal for a "+"$a_{s}=$"+str(InitialA)+" to NI quench." + r" $n_{\rm i}$="+str(InitialJ)

    #NIInt
    if InitialA==0 and FinalA!=0 and FinalA!="inf":
        title=r"Ramsey Signal for a NI to $a_{s}=$" +str(FinalA)+ " quench." + r" $n_{\rm i}$="+str(InitialJ) 

    #IntUnit
    if InitialA!=0 and InitialA!="inf" and FinalA=="inf":
        title=r"Ramsey Signal for a "+"$a_{s}=$"+str(InitialA)+" to unitary quench." + r" $n_{\rm i}$="+str(InitialJ) 

    #UnitInt
    if InitialA=="inf" and FinalA!=0 and FinalA!="inf":
        title=r"Ramsey Signal for a unitary to $a_{s}=$" +str(FinalA)+ " quench." + r" $n_{\rm i}$="+str(InitialJ) 

    #UnitUnit
    if InitialA=="inf" and FinalA=="inf":
        title=r"Ramsey Signal for a Unitary to Unitary quench. (Unphysical)"

    #NINI
    if InitialA==0 and FinalA==0:
        title=r"Ramsey Signal for a NI to NI quench. (Unphysical)"

    #Int to Same Int
    if InitialA==FinalA and InitialA!=0 and InitialA!="inf":
        title=r"Ramsey Signal for a "+"$a_{s}=$"+str(InitialA)+" to $a_{s}=$" +str(FinalA)+ " quench. (Unphysical)" + r" $n_{\rm i}$="+str(InitialJ)

  
    plt.figure(1)

    plt.rcParams['xtick.labelsize']=40
    plt.rcParams['ytick.labelsize']=40

    plt.suptitle(title,fontsize=35)

    ax1=plt.subplot(211)
    plt.plot(Tomega, abs(S),'bo',markersize=5)
    plt.axis([0, Tmax, 0, 1])
    plt.ylabel("$|S(t)|$",fontsize=35*1.5)
    plt.setp(ax1.get_xticklabels(),visible=False)

    ax2=plt.subplot(212)
    plt.plot(Tomega, -np.angle(S)/(np.pi),'bo',markersize=5)
    plt.axis([0, Tmax, -1, 1 ])
    plt.ylabel("$\phi(t)/\pi$",fontsize=35*1.5)
    plt.xlabel("t$\omega/\pi$",fontsize=35*1.5)

    plt.show()


    return None

def PlotExpectR(Nmax,InitialA,InitialJ,FinalA,Tmax,Spacing):
    """
    Gives the particle separation expectationover time for
    an arbitrary quench

    Y-axis is <r(t)>, X-axis is time in units of 
    \pi/\omega

    This is intended to make an illustrative plot rather than
    to make something that goes into a paper

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
        in units of \pi/\omega
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

    [Tomega,R]=TBZR.ExpectR(Nmax,InitialA,InitialJ,FinalA,Tmax,Spacing)
    #print(R[int(len(Tomega)/2)])

    #UnitNI
    if InitialA=="inf" and FinalA==0:
        title=r"$\langle r(t) \rangle$ for a Unitarity to NI quench." + r" $n_{\rm i}$="+str(InitialJ)
    
    #NIUnit
    if InitialA==0 and FinalA=="inf":
        title=r"$\langle r(t) \rangle$ for a NI to Unitary quench." + r" $n_{\rm i}$="+str(InitialJ)

    #IntInt
    if InitialA!=0 and InitialA!="inf" and FinalA!=0 and FinalA!="inf":
        title=r"$\langle r(t) \rangle$ for a "+"$a_{s}=$"+str(InitialA)+" to $a_{s}=$" +str(FinalA)+ " quench." + r" $n_{\rm i}$="+str(InitialJ) 

    #IntNI
    if InitialA!=0 and InitialA!="inf" and FinalA==0:
        title=r"$\langle r(t) \rangle$ for a "+"$a_{s}=$"+str(InitialA)+" to NI quench." + r" $n_{\rm i}$="+str(InitialJ)

    #NIInt
    if InitialA==0 and FinalA!=0 and FinalA!="inf":
        title=r"$\langle r(t) \rangle$ for a NI to $a_{s}=$" +str(FinalA)+ " quench." + r" $n_{\rm i}$="+str(InitialJ) 

    #IntUnit
    if InitialA!=0 and InitialA!="inf" and FinalA=="inf":
        title=r"$\langle r(t) \rangle$ for a "+"$a_{s}=$"+str(InitialA)+" to unitary quench." + r" $n_{\rm i}$="+str(InitialJ) 

    #UnitInt
    if InitialA=="inf" and FinalA!=0 and FinalA!="inf":
        title=r"$\langle r(t) \rangle$ for a unitary to $a_{s}=$" +str(FinalA)+ " quench." + r" $n_{\rm i}$="+str(InitialJ) 

    #UnitUnit
    if InitialA=="inf" and FinalA=="inf":
        title=r"$\langle r(t) \rangle$ for a Unitary to Unitary quench. (Unphysical)"

    #NINI
    if InitialA==0 and FinalA==0:
        title=r"$\langle r(t) \rangle$ for a NI to NI quench. (Unphysical)"

    #Int to Same Int
    if InitialA==FinalA and InitialA!=0 and InitialA!="inf":
        title=r"$\langle r(t) \rangle$ for a "+"$a_{s}=$"+str(InitialA)+" to $a_{s}=$" +str(FinalA)+ " quench. (Unphysical)" + r" $n_{\rm i}$="+str(InitialJ)

            
    fig=plt.figure()
    ax=plt.axes()
    
    ax.tick_params(axis='x', labelsize=40)
    ax.tick_params(axis='y', labelsize=40)

    plt.title(title,fontsize=52.5)
    plt.plot(Tomega, R,'k',linewidth=10)
    plt.axis([0, Tmax, 0, 1.1*max(R)])
    plt.ylabel(r"$\langle r(t)/a_{\mu}\rangle$",fontsize=52.5)
    plt.xlabel(r"$\omega $t/$\pi$",fontsize=52.5)
    plt.show()

    return None

def PlotRDistrib(Nmax,InitialA,InitialJ,FinalA,Tomega,Rmax,Spacing):
    """
    Gives P(r,t) at a specified time following an arbitrary
    quench

    This is intended to make an illustrative plot rather than
    to make something that goes into a paper

    Y-axis is Ramsey Signal, X-axis is time in units of 
    \pi/\omega

    Parameters
    ----------------------
    Nmax: an int
        the maximum value of the double sum we go up to
    InitialJ: an int
        the excitation state of the initial unitary system
    Tomega: a float
        the time value we evaluate for (units of \omega/\pi)
    Rmax: a float
        calculate all the values of this integrand from 0 to Rmax
    spacing: a float
        the resolution in r
    """

    [Rrange,Distrib]=TBZR.RProbDistrib(Nmax,InitialA,InitialJ,FinalA,Tomega,Rmax,Spacing)
    #ExpectR=Spacing*sum(Distrib)
    #print("<r(t="+str(Tomega)+")>="+str(ExpectR))

    if InitialA=="inf" and FinalA==0:
        title=r"$P(r,\omega t/\pi$="+str(Tomega)+") for a Unitarity to NI quench." + r" $n_{\rm i}$="+str(InitialJ)
    
    #NIUnit
    if InitialA==0 and FinalA=="inf":
        title=r"$P(r,\omega t/\pi$="+str(Tomega)+") for a NI to Unitary quench." + r" $n_{\rm i}$="+str(InitialJ)

    #IntInt
    if InitialA!=0 and InitialA!="inf" and FinalA!=0 and FinalA!="inf":
        title=r"$P(r,\omega t/\pi$="+str(Tomega)+") for a "+"$a_{s}=$"+str(InitialA)+" to $a_{s}=$" +str(FinalA)+ " quench." + r" $n_{\rm i}$="+str(InitialJ) 

    #IntNI
    if InitialA!=0 and InitialA!="inf" and FinalA==0:
        title=r"$P(r,\omega t/\pi$="+str(Tomega)+") for a "+"$a_{s}=$"+str(InitialA)+" to NI quench." + r" $n_{\rm i}$="+str(InitialJ)

    #NIInt
    if InitialA==0 and FinalA!=0 and FinalA!="inf":
        title=r"$P(r,\omega t/\pi$="+str(Tomega)+") for a NI to $a_{s}=$" +str(FinalA)+ " quench." + r" $n_{\rm i}$="+str(InitialJ) 

    #IntUnit
    if InitialA!=0 and InitialA!="inf" and FinalA=="inf":
        title=r"$P(r,\omega t/\pi$="+str(Tomega)+") for a "+"$a_{s}=$"+str(InitialA)+" to unitary quench." + r" $n_{\rm i}$="+str(InitialJ) 

    #UnitInt
    if InitialA=="inf" and FinalA!=0 and FinalA!="inf":
        title=r"$P(r,\omega t/\pi$="+str(Tomega)+") for a unitary to $a_{s}=$" +str(FinalA)+ " quench." + r" $n_{\rm i}$="+str(InitialJ) 

    #UnitUnit
    if InitialA=="inf" and FinalA=="inf":
        title=r"$P(r,\omega t/\pi$="+str(Tomega)+") for a Unitary to Unitary quench. (Unphysical)"

    #NINI
    if InitialA==0 and FinalA==0:
        title=r"$P(r,\omega t/\pi$="+str(Tomega)+") for a NI to NI quench. (Unphysical)"

    #Int to Same Int
    if InitialA==FinalA and InitialA!=0 and InitialA!="inf":
        title=r"$P(r,\omega t/\pi$="+str(Tomega)+") for a "+"$a_{s}=$"+str(InitialA)+" to $a_{s}=$" +str(FinalA)+ " quench. (Unphysical)" + r" $n_{\rm i}$="+str(InitialJ)

    
    fig=plt.figure()
    ax=plt.axes()
    
    ax.tick_params(axis='x', labelsize=40)
    ax.tick_params(axis='y', labelsize=40)

    plt.title(title,fontsize=52.5)
    plt.plot( Rrange,Distrib,'k',linewidth=10)
    plt.axis([0, Rmax, 0, 1.1*max(Distrib)])
    plt.ylabel(r"Prob. Density",fontsize=52.5)
    plt.xlabel(r"$r/a_{\mu}$",fontsize=52.5)
    plt.show()


    return None


def DemoOfFunctions():
    """
    This runs an interactive session which lets the user
    make various plots to demonstrate the functions of 
    the package are working as correctly
    """

    print("This is a demonstration that the functions of the TwoBodyZeroRange package work as desired")

    #checking wavefunction normalisations 
    #IntWaveFuncNormCheck(-3,3,0.1,10e-6)
    #NIWaveNormCheck(0,30,Tol=10e-6)


    print("Want to make a plot of the energy spectrum Y/N?")
    YesNo=input()
    if YesNo=="Y" or YesNo=="y":
        print("How many energy levels?")
        Levels=int(input())
        print("Horizontal limits?")
        Hmax=float(input())
        print("Horizontal distance between points?")
        Spacing=float(input())
        print("calculating...")
        PlotEnergySpectrum(Levels,Hmax,Spacing)
    elif YesNo=="N" or YesNo=="n":
        print("Skipping energy spectrum")
    else:
        print("not a Y or N and skipping energy spectrum")


    #Energy Excitation
    print("Want to make a plot of post quench <E> Y/N?")
    YesNo=input()
    if YesNo=="Y" or YesNo=="y":
        print("Initial a_s? (infinite is \"inf\")")
        InitialA=float(input())
        print("Final a_s? (infinite is \"inf\")")
        FinalA=float(input())
        print("Initial excitation?")
        InitialJ=int(input())
        print("How many terms in the expansion?")
        Nmax=int(input())
        print("calculating...")
        PlotEnergyExpect(Nmax,InitialA,InitialJ,FinalA)
    elif YesNo=="N" or YesNo=="n":
        print("Skipping <E>")
    else:
        print("not a Y or N and skipping <E>")

    #Ramsey Signal
    print("Want to make a plot of the Ramsey Signal Y/N?")
    YesNo=input()
    if YesNo=="Y" or YesNo=="y":
        print("Initial a_s? (infinite is \"inf\")")
        InitialA=float(input())
        print("Final a_s? (infinite is \"inf\")")
        FinalA=float(input())
        print("Initial excitation?")
        InitialJ=int(input())
        print("How many terms in the expansion?")
        Nmax=int(input())
        print("Maximum time (units of omega/pi)?")
        Tmax=float(input())
        print("Timestep size?")
        Spacing=float(input())
        print("calculating...")
        PlotRamseySignal(Nmax,InitialA,InitialJ,FinalA,Tmax,Spacing)
    elif YesNo=="N" or YesNo=="n":
        print("Skipping Ramsey signal")
    else:
        print("not a Y or N and skipping Ramsey signal")

    #Particle separation
    print("Want to make a plot of average interparticle separation as a function of time Y/N?")
    YesNo=input()
    if YesNo=="Y" or YesNo=="y":
        print("Initial a_s? (infinite is \"inf\")")
        InitialA=float(input())
        print("Final a_s? (infinite is \"inf\")")
        FinalA=float(input())
        print("Initial excitation?")
        InitialJ=int(input())
        print("How many terms in the expansion?")
        Nmax=int(input())
        print("Maximum time (units of omega/pi)?")
        Tmax=float(input())
        print("Timestep size?")
        Spacing=float(input())
        print("calculating...")
        PlotExpectR(Nmax,InitialA,InitialJ,FinalA,Tmax,Spacing)
    elif YesNo=="N" or YesNo=="n":
        print("Skipping <r>")
    else:
        print("not a Y or N and skipping <r>")

    #P(r,t)
    print("Want to make a plot of the interparticle probability distribution at some time Y/N?")
    YesNo=input()   
    if YesNo=="Y" or YesNo=="y":
        print("Initial a_s? (infinite is \"inf\")")
        InitialA=float(input())
        print("Final a_s? (infinite is \"inf\")")
        FinalA=float(input())
        print("Initial excitation?")
        InitialJ=int(input())
        print("How many terms in the expansion?")
        Nmax=int(input())
        print("Time (units of omega/pi)?")
        Tomega=float(input())
        print("Maximum distance (in unis of SHO length)")
        Rmax=float(input())
        print("Step size?")
        Spacing=float(input())
        print("calculating...")
        PlotRDistrib(Nmax,InitialA,InitialJ,FinalA,Tomega,Rmax,Spacing)
    elif YesNo=="N" or YesNo=="n":
        print("Skipping P(r,t)")
    else:
        print("not a Y or N and skipping <r>")
    

    return None

#########################

DemoOfFunctions()
