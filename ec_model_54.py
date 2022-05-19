import numpy as np
from numpy import sin as sin
from numpy import cos as cos
import warnings, traceback, sys

def ec_create_data(delta,
    phi,
    theta,
    psi,
    R,
    y0,
    h,
    num,
    n,
    m,
    us,
    Cnm = 1,
    By0 = 10,
    tau = 1.3,
    p_0 = 0,
    norm_poloidal=True,
    noise_type='none',
    epsilon = 0.05):
    """
    The Elliptic-cylindrical Magnetic Flux Rope Model

    A python implementation of the model by T. Nieves-Chinchilla et al. from
    https://doi.org/10.3847/1538-4357/aac951

    Parameters
    ----------
    All angles are expected in units of degrees.

    Coordinate system described are with the assumption that the 
    spacecraft trajectory is along the GSEx direction.
    (TODO: generalize these descriptions to reflect a relationship
    with the S/C RTN system)

    delta : float, A measure of the ellipticity of the flux rope.  
            The ratio between the length of the minor and major axes
            of the elliptical cross section. Valid range (0,1]
    
    phi : float, Angle of rotations of the flux rope around the Z axis, in 
          the XY plane. Valid range [0,2*pi]
    
    theta : float, Angle of rotation of the flux rope out of the XY plane. 
            Valid range [-pi/2, pi/2]
    
    psi : float, Angle of rotation about the central axis of the flux rope.
          Valid range [0,180]
    
    y0 : float, Spacecraft distance from the central axis of the flux rope.
         Valid range (-100, 100)
    
    h :  int, {-1, 1} Helicty of flux rope.

    Cnm : int, optional, default: 1
         Valid only if > 0

    By0 : int, optional, default: 10

    tau : float, optional, default: 1.3

    us : float, optional, default: 450.0
         Spacecraft velocity in km/s

    R : float, optional, default: 0.07
        Radius of the semi-major axis in Astronomical Units (AU)

    n : float, optional, default: 1.0

    m : float, optional, default: 0.0

    norm_poloidal : boolean, optional, default: True
        If True, divide poloidal components by Bmax

    noise_type : {'none,'gaussian','uniform'}, optional, default: 'none'
        The type of noise to add to each component of the magnetic field 
        result.

    epsilon : float > 0, optional, default: 0.05
        Size of noise modifier.  Ignored if noise_type is 'none'.
        
        For noise_type 'gaussian', epsilon is the standard deviation of the 
        normal distrubution centered on 0.  Values in the distribution
        Normal(mu=0,sigma=epsilon) are added to the magnetic field components.

        For noise_type 'uniform', epsilon defines the +/- bounds of the 
        distribution.  Values in [-epsilon,epsilon] are added to the magnetic
        field components.

    num : int,
        Length of return data time series in points.

    Returns
    ----------
    np.array size=(5, num) holding time,BB,BxGSE,ByGSE,BzGSE
    """

    # Store values for warning messages
    phi_in = phi
    th_in  = theta
    psi_in = psi
    y0_in  = y0

    assert delta <= 1 and delta > 0, "DELTA must be in (0,1]"
    assert phi >= 0 and phi < 360, "PHI must be in [0,360)"
    #assert phi != 90 and phi != 270, "PHI cannot be (2k+1)pi/2"
    assert theta >= -90 and theta <= 90, "THETA must be in [-90,90]"
    assert psi >= 0 and psi <= 180, "PSI must be in [0,180]"
    assert y0 > -100 and y0 < 100, "Y0 must be in (-100,100)"
    assert h == 1 or h == -1, "H must be in {-1, 1}"
    assert Cnm > 0, "Cnm must be > 0"
    # TODO: error checks for By0, tau and R ranges

    # Constants
    AU_to_km = 1.496e8
    AU_to_m  = 1.496e11

    # Derived parameters
    if norm_poloidal:
        Bmax = delta * By0 * tau
    else:
        Bmax = 1.

    # Adjust arguments for internal coordinate system and convert to radians
    phi = np.radians(phi-90.)
    if (phi < 0):
        phi += 2 * np.pi
    theta = np.radians(theta)
    psi = np.radians(psi)
    y0 = y0*R/100.


    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            #TODO: Give BGSE a more generic name.  
            # My understanding is it's essentially a rotation of the RTN system.

            HH_factor = np.sqrt(1 - (sin(phi) * cos(theta))**2)
            z0 = y0 * HH_factor / (cos(theta) * cos(phi))

            fctr1 = cos(psi)**2 + (delta * sin(psi))**2 # b-term in the notes
            fctr2 = sin(psi)**2 + (delta * cos(psi))**2 # a-term in the notes
            fctr3 = delta**2 - 1

            F2 = fctr1 * cos(phi)**2 \
                + fctr2 * (sin(phi) * sin(theta))**2 \
                + 2 * fctr3 * cos(phi)  * sin(phi) * sin(theta) * cos(psi) * sin(psi)

            F1 = fctr2 * sin(phi) * sin(theta) * cos(theta) \
                + fctr3 * cos(phi) * cos(theta) * sin(psi) * cos(psi)

            time_final = AU_to_km * (2./us) * (1./F2) \
                        * np.sqrt(F2 * R**2 - (HH_factor*y0)**2)
            
            tt = np.linspace(0,time_final,num=num)

            xc = (500 * time_final * us / AU_to_m) - (z0 * F1 / F2)
            AA = -xc * (cos(phi) * cos(psi) - sin(phi) * sin(theta) * sin(psi)) \
                + cos(theta) * sin(psi) * z0

            BB = -xc * (cos(phi) * sin(psi) + sin(phi) * sin(theta) * cos(psi)) \
                - cos(theta) * cos(psi) * z0

            RR = np.sqrt((AA/delta)**2 + BB**2)

            xsat = (1.0e3 * us * tt / AU_to_m) - xc #in AU

            XL = xsat * (cos(phi) * cos(psi) - sin(phi) * sin(theta) * sin(psi)) \
                + cos(theta)*sin(psi)*z0
            YL = -sin(phi) * cos(theta) * xsat + sin(theta) * z0
            ZL =  xsat * (cos(phi) * sin(psi) + sin(phi) * sin(theta) * cos(psi)) \
                - cos(theta) * cos(psi) * z0
            rnm = np.sqrt(XL**2 + ZL**2) # r_sat
            PHIsat = np.arctan2(ZL,XL)
            HH_f = np.sqrt((delta * sin(phi))**2 + cos(phi)**2) # h (factor de escala)
            
            HR = np.sqrt(sin(phi)**2 + (delta * cos(phi))**2)

            By = delta * By0 * (tau - (rnm/RR)**(n+1)) / Bmax
            fctr = (n+1) / (delta**2 + m + 1)
            Bpol = -h * delta * HH_f * fctr * (By0 / Cnm) * (rnm/RR)**(m+1) / Bmax
            
            mu0 = 4*np.pi*(10**(-7))
            
            jy = (n+1)*By0*(rnm/RR)**m/(Bmax*mu0*h*Cnm*RR*AU_to_m)  #normalized equations for CC-model
            jpol = -(n+1)*By0*(rnm/RR)**n/(Bmax*mu0*RR*AU_to_m)    

            BxL = -Bpol * sin(PHIsat)
            ByL = By
            BzL = Bpol * cos(PHIsat)
            
            jxL = -jpol * sin(PHIsat)
            jyL = jy
            jzL = jpol * cos(PHIsat)

            BxGSE = (BxL * (cos(phi) * cos(psi) - sin(psi) * sin(phi) *sin(theta)) 
                - ByL * sin(phi) * cos(theta) 
                + BzL * (cos(phi) * sin(psi) + cos(psi) * sin(phi) * sin(theta)))

            ByGSE = BxL* (sin(phi) * cos(psi) + sin(psi) * cos(phi) *sin(theta)) \
                + ByL * cos(phi)*cos(theta) \
                + BzL * (sin(phi) * sin(psi) - cos(psi) * cos(phi) * sin(theta))

            BzGSE = -BxL * sin(psi) * cos(theta) \
                + ByL * sin(theta) \
                + BzL * cos(psi) * cos(theta)
            
            jxGSE = (jxL * (cos(phi) * cos(psi) - sin(psi) * sin(phi) *sin(theta)) 
                - jyL * sin(phi) * cos(theta) 
                + jzL * (cos(phi) * sin(psi) + cos(psi) * sin(phi) * sin(theta)))

            jyGSE = jxL* (sin(phi) * cos(psi) + sin(psi) * cos(phi) *sin(theta)) \
                + jyL * cos(phi)*cos(theta) \
                + jzL * (sin(phi) * sin(psi) - cos(psi) * cos(phi) * sin(theta))

            jzGSE = -jxL * sin(psi) * cos(theta) \
                + jyL * sin(theta) \
                + jzL * cos(psi) * cos(theta)
            
            alpha_n = By0*(n+1) / (mu0*delta*tau*(RR*AU_to_m)**(n+1))
            beta_m = By0*(n+1) / (mu0*delta*Cnm*tau*(RR*AU_to_m)**(m+1))
            
            pCC = alpha_n*By0*(rnm*AU_to_m)**(n+1)  / (n+1) \
                -mu0*(alpha_n**2)*((rnm*AU_to_m)**(2*n+2)) / ((n+1)*(2*n+2)) \
                - mu0*(beta_m**2)*((rnm*AU_to_m)**(2*m+2)) / ((m+2)*(2*m+2))
            
            pCC = p_0 + pCC - np.min(pCC)
            
            chi = (delta**2 + 1)/(HH_f**2)
            
            
            pEC = alpha_n*By0*(rnm*AU_to_m)**(n+1)  / (n+1) \
                -mu0*(alpha_n**2)*((rnm*AU_to_m)**(2*n+2)) / ((n+1)*(2*n+2)) \
                - mu0*(HH_f**2)*(beta_m**2)*(chi+m)*((rnm*AU_to_m)**(2*m+2)) / (((delta**2 + m + 1)**2)*(2*m+2))
            
            pEC = HH_f*delta*pEC 
            pEC = p_0 + pEC - np.min(pEC)

            # Add the noise
            if noise_type == 'uniform':
                rng = np.random.default_rng()   #Random number Generator
                BxGSE = 2 * epsilon * rng.random(num) - epsilon
                ByGSE = 2 * epsilon * rng.random(num) - epsilon
                BzGSE = 2 * epsilon * rng.random(num) - epsilon

            elif noise_type == 'gaussian':
                rng = np.random.default_rng()   #Random number Generator
                BxGSE += rng.normal(0,epsilon,num)
                ByGSE += rng.normal(0,epsilon,num)
                BzGSE += rng.normal(0,epsilon,num)

            BB = np.sqrt(BxGSE**2 +ByGSE**2 + BzGSE**2)
            
            jj = np.sqrt(jxGSE**2 +jyGSE**2 + jzGSE**2)

        except Warning:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            # traceback.print_tb(exc_traceback, limit=1, file=sys.stdout)
            traceback.print_exc(limit=1, file=sys.stdout)
            print("Above- delta:{d}; phi:{ph}; theta:{th}; psi:{ps}; y0:{y0}; h:{h}".format(d=delta,ph=phi_in,th=th_in,ps=psi_in,y0=y0_in,h=h))
            return(np.full((5,num),np.nan))

    return(np.stack((tt,BB,BxGSE,ByGSE,BzGSE,jj,jxGSE,jyGSE,jzGSE,pCC,pEC)))


    
