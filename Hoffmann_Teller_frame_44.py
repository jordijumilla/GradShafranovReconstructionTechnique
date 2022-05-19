from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt

# Minimization process
def minimizeD(initialGuess, Brtn, Vrtn, logPath):
    numData = len(Brtn[:, 0])
    
    # Definition of the convection electric field function to be minimized
    def D(v):
        norm = 0
        for i in range(numData):
            norm = norm + np.linalg.norm(np.cross(Vrtn[i,:]/1e5 - v, Brtn[i,:]*1e6))**2
        return norm/numData
    
    result = minimize(D, initialGuess)

    if result.success:
        vHT = result.x*1e5 # Hoffmann-Teller velocity
        D = result.fun # Minimum value of the function
                
        D0 = 0
        for i in range(numData):
            D0 = D0 + np.linalg.norm(np.cross(Vrtn[i,:]/1e5, Brtn[i,:]*1e6))**2
        D0 = D0/numData
        
        return [vHT, D, D0]
    else:
        with open(logPath, 'a') as logFile:
            logFile.writelines("Minimization process not successful.\n")
        print("Minimization process not successful.\n")
        return result.x*1e5
    
def plotRegression(Ec, EHT, Ec_reg, EHT_reg, r_sq, coef, intercept, path):
    fig, ax = plt.subplots(figsize=(8, 8), dpi = 200, tight_layout = True)
    plt.plot(EHT[:, 0]*1e3, Ec[:, 0]*1e3, '.r', markersize = 7)
    plt.plot(EHT[:, 1]*1e3, Ec[:, 1]*1e3, '.g', markersize = 7)
    plt.plot(EHT[:, 2]*1e3, Ec[:, 2]*1e3, '.b', markersize = 7)
    plt.plot(EHT_reg*1e3, Ec_reg*1e3, '-k', linewidth = 3, alpha = 0.4)
    plt.axvline(x = 0, linestyle = '--', alpha = 0.4)
    plt.axhline(y = 0, linestyle = '--', alpha = 0.4)
    plt.title('Scatter plot and regression of convection E')
    plt.legend(['r components', 't components', 'n components'], facecolor='white')
    plt.xlabel('$E_{HT}$ (mV/m)')
    plt.ylabel('$E_c$ (mV/m)')
    ax.set_aspect('equal')
    pos_y = -np.max(Ec)/2
    stringText = '$R^2$ = ' + str(round(r_sq, 6)) + '\nSlope: ' + str(round(coef, 5)) + '\nIntercept = ' + str(round(intercept, 9))
    plt.text(0, pos_y*1e3, stringText, bbox = dict(facecolor = 'gray', alpha = 0.5))
    plt.savefig(path + '/ConvectionEregression.png')
    plt.show()