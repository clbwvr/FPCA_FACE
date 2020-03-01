from fpca_face import FPCA_FACE
import numpy as np
import pandas as pd
from math import pi
import matplotlib.pyplot as plt

### Sample 1: Simulated Data

# Generate Data
np.random.seed(123)
I = 50 
J = 3000 
t = [(j + 1) / J for j in range(J)] 
N = 4 
sigma = 2 
lambdaTrue = [1,0.5,0.5**2,0.5**3]
phi = np.sqrt(2) * pd.DataFrame({"phi1":[np.sin(2*pi*ti) for ti in t],
              "phi2":[np.cos(2*pi*ti) for ti in t],
              "phi3":[np.sin(4*pi*ti) for ti in t],
              "phi4":[np.cos(4*pi*ti) for ti in t]})
xi = np.reshape(np.random.normal(size=I*N), (I,N))
xi = xi @ np.diag(np.sqrt(lambdaTrue))
X = xi @ (phi.values.T); # of size I by J
Y = X + sigma*np.reshape(np.random.normal(size=I*J), (I,J))

# Fit PCA
mod = FPCA_FACE(Y=Y,argvals=t,npc=4,knots=100,center=True)
fit = mod.fit()

# Plot Estimates against truth
pd.DataFrame(fit.efunctions).plot()
phi.plot()



### Sample 2: Real Data

# Read in data
Y = pd.read_csv("weather.csv", index_col=0)
Y.plot(legend=False)
Y = Y.T

# Fit PCA
mod = FPCA_FACE(Y=Y,npc=3,knots=100, center=True)
fit = mod.fit()

# Plot Estimates against truth
pd.DataFrame(fit.efunctions).plot()

# Plot functional variance explained
plt.bar(["PC1","PC2","PC3"], fit.fve)
