import numpy as np
import matplotlib.pyplot as plt

c = float(input("Enter surface isolevel: "))
d = float(input("Enter target Schwarz-P inlet diameter: "))

N = 1000
theta = 2*np.pi*np.linspace(0,1,N+1)[:-1]
r_TPMS = np.zeros((N,))
for i in range(N):
    r = 0.25
    m = 2*np.pi*np.array([np.cos(theta[i]),np.sin(theta[i])])
    residual = (c+1)-np.sum(np.cos(m*r))
    if abs(residual) > 1e-12:
        r -= residual/np.sum(m*np.sin(r*m))
        residual = (c+1)-np.sum(np.cos(m*r))
    r_TPMS[i] = r

Size = np.sum(r_TPMS*d/2)/np.sum(r_TPMS*r_TPMS)

r_TPMS *= Size
plt.plot(r_TPMS*np.cos(theta),r_TPMS*np.sin(theta),c="b")
plt.plot((d/2)*np.cos(theta),(d/2)*np.sin(theta),c="k")
plt.show()

print("Calculated unit cell edge length: "+str(Size))