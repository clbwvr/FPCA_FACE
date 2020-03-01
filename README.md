# fpca_face

## Overview
Functional principal component analysis with fast covariance estimation via the sandwich smoother (Xiao et al., 2013) for covariance matrix smoothing.

Reference:
Xiao, L., Li, Y., Ruppert, D.: Fast bivariate P-splines: the sandwich smoother. J. R. Stat. Soc. B 75, 577–599 (2013)


```
git clone https://github.com/clbwvr/fpca_face
cd fpca_face
```

```py
from fpca_face import FPCA_FACE
mod = FPCA_FACE(Y=Y,argvals=argvals)
fit = mod.fit()
```

