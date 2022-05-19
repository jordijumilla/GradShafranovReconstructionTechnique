import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import math
from ReconstructionUtils import*

# Magnetic field or velocity components plot
def plotVectorMagnitude(timeSec, Ax, Ay, Az, mag, labels, unit, unitFactor, title, path):
    timeHour = timeSec/3600
    modA = np.sqrt(Ax**2 + Ay**2 + Az**2)
    
    plt.figure(figsize=(10,6), dpi = 200, tight_layout=True)
    plt.plot(timeHour, Ax*unitFactor, '.-r', markersize = 10)
    plt.plot(timeHour, Ay*unitFactor, '.-g', markersize = 10)
    plt.plot(timeHour, Az*unitFactor, '.-b', markersize = 10)
    plt.plot(timeHour, modA*unitFactor, '.-k', markersize = 10)
    plt.axhline(y = 0, linestyle = '--', alpha = 0.4)
    plt.xlim(np.min(timeSec)/3600, np.max(timeSec)/3600)
    plt.legend(['$' + mag + '_' + labels[0] + '$', '$' + mag + '_' + labels[1] + '$', '$' + mag + '_' + labels[2] + '$', '$|' + mag + '|$'], facecolor='white')
    plt.xlabel('Time (h)')
    plt.ylabel(mag + ' components (' + unit + ')')
    plt.title(title)
    plt.savefig(path + '/' + title + '.png')
    plt.show()

# Plasma velocity angle plot
def plotVrtnAngle(df, Vrtn, path):
    alpha = [0]
    V_0 = Vrtn[0,:]
    V_0_norm = np.linalg.norm(V_0)
    for i in range(1,len(df)):
        V_i = Vrtn[i,:]
        V_i_norm = np.linalg.norm(V_i)
        dot_product = np.dot(V_0, V_i)
        angle = math.acos(dot_product/(V_0_norm*V_i_norm))*(180/math.pi) 
        alpha = np.append(alpha, angle)

    figure(figsize=(8, 6), dpi=80)
    plt.plot(df['seconds']/3600, alpha, '.-r')
    plt.axhline(y = 0, linestyle = '--', alpha = 0.3)
    plt.xlabel('Time (h)')
    plt.ylabel('Angle (degrees)')
    plt.title('Angle between the plasma velocity and initial plasma velocity')
    plt.savefig(path + '/' + 'Angle between the plasma velocity and initial plasma velocity.png')
    plt.show()
    
def plotResidualVelocity(timeSec, Vrel, vHT, title, labels, path):
    modV = np.sqrt(Vrel[:, 0]**2 + Vrel[:, 1]**2 + Vrel[:, 2]**2)
    
    timeHour = timeSec/3600
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi = 200, tight_layout=True)
    ax2 = ax1.twinx()

    ax1.plot(timeHour, Vrel[:, 0]/1e3, '.-r')
    ax2.plot(timeHour, (Vrel[:, 0]/np.linalg.norm(vHT))*100, '.-r')

    ax1.plot(timeHour, Vrel[:, 1]/1e3, '.-g')
    ax2.plot(timeHour, (Vrel[:, 1]/np.linalg.norm(vHT))*100, '.-g')

    ax1.plot(timeHour, Vrel[:, 2]/1e3, '.-b')
    ax2.plot(timeHour, (Vrel[:, 2]/np.linalg.norm(vHT))*100, '.-b')

    ax1.plot(timeHour, modV/1e3, '.-k')
    ax2.plot(timeHour, (modV/np.linalg.norm(vHT))*100, '.-k')
    
    plt.axhline(y = 0, linestyle = '--', alpha = 0.4)

    plt.xlim(np.min(timeHour), np.max(timeHour))
    plt.legend(['$v^{res}_' + labels[0] + '$', '$v^{res}_' + labels[1] + '$', '$v^{res}_' + labels[2] + '$', '$|v^{res}|$'], facecolor='white')
    
    ax1.set_title(title)
    ax1.set_xlabel('Time (h)')
    ax1.set_ylabel('$v_{res}$ (km/s)')
    ax2.set_ylabel('$v_{res}/|v_{HT}|$ (%)')
    ax2.grid(False)
    
    plt.savefig(path + '/ResidualVelocity.png')
    plt.show()
    
def plotVectorMagnitudeDist(x, Ax, Ay, Az, mag, labels, unit, unitFactor, title, path):
    x_AU = x/AU
    modA = np.sqrt(Ax**2 + Ay**2 + Az**2)
    
    plt.figure(figsize=(10,6), dpi = 200, tight_layout=True)
    plt.plot(x_AU, Ax*unitFactor, '.-r', markersize = 10)
    plt.plot(x_AU, Ay*unitFactor, '.-g', markersize = 10)
    plt.plot(x_AU, Az*unitFactor, '.-b', markersize = 10)
    plt.plot(x_AU, modA*unitFactor, '.-k', markersize = 10)
    plt.axhline(y = 0, linestyle = '--', alpha = 0.4)
    plt.xlim(np.min(x_AU), np.max(x_AU))
    plt.legend(['$' + mag + '_' + labels[0] + '$', '$' + mag + '_' + labels[1] + '$', '$' + mag + '_' + labels[2] + '$', '$|' + mag + '|$'], facecolor='white')
    plt.xlabel('x (reconstruction frame) (AU)')
    plt.ylabel(mag + ' components (' + unit + ')')
    plt.title(title)
    plt.savefig(path + '/' + title + '.png')
    plt.show()