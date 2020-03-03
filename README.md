# fpca_face

## Overview
Functional principal component analysis with fast covariance estimation via the sandwich smoother [1,2] for covariance matrix smoothing.

References:

[1] Xiao, L., Li, Y., & Ruppert, D. Fast bivariate P-splines: The sandwich smoother. Journal of the Royal Statistical Society. Series B (Statistical Methodology), 75(3), 577-599. (2013).

[2] Xiao, L., Zipunnikov, V., Ruppert, D. et al. Fast covariance estimation for high-dimensional functional data. Stat Comput 26, 409–421 (2016).

```
git clone https://github.com/clbwvr/fpca_face
cd fpca_face
```

```py
from fpca_face import FPCA_FACE
mod = FPCA_FACE(Y=Y,argvals=argvals)
fit = mod.fit()
```

