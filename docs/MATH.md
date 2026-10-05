# Mathematical model

## Parametric dog fit

BITE estimates D-SMAL shape coefficients $\boldsymbol{\beta}$, limb scales $\boldsymbol{\ell}$, joint rotations $\boldsymbol{\theta}$, per-vertex offsets $\boldsymbol{\Delta}$, alignment $\mathbf{A}$, global log scale $\gamma$, and translation $\mathbf{t}$.

$$
\mathbf{V}=e^\gamma\mathbf{A}\operatorname{SMAL}(\boldsymbol{\beta},\boldsymbol{\ell},\boldsymbol{\theta},\boldsymbol{\Delta})+\mathbf{t}
$$

The implementation keeps all 35 D-SMAL joint weights and the full 3,889-vertex topology.

## Gaussian appearance

Each appearance element is an anisotropic Gaussian with center $\boldsymbol{\mu}_i$, covariance $\boldsymbol{\Sigma}_i$, opacity $\alpha_i$, and RGB color $\mathbf{c}_i$.

$$
G_i(\mathbf{x})=\alpha_i\exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu}_i)^T\boldsymbol{\Sigma}_i^{-1}(\mathbf{x}-\boldsymbol{\mu}_i)\right)
$$

Projected Gaussians are composited from front to back.

$$
\mathbf{C}(p)=\sum_i T_i(p)\alpha_i(p)\mathbf{c}_i
$$

$$
T_i(p)=\prod_{j<i}\left(1-\alpha_j(p)\right)
$$

## Reconstruction objective

Rendered targets use an unlit white background and spherical harmonics degree zero. The photometric term combines pixel error and structural similarity.

$$
\mathcal{L}_{\text{photo}}=0.8\lVert I-\hat I\rVert_1+0.2\left(1-\operatorname{SSIM}(I,\hat I)\right)
$$

Both optimization stages add edge-length, uniform Laplacian, selected pose, and vertex-offset penalties.

$$
\mathcal{R}=\mathcal{L}_{\text{edge}}+\mathcal{L}_{\text{lap}}+\mathcal{L}_{\text{pose}}+\mathcal{L}_{\text{offset}}
$$

The bound stage encourages visible surface Gaussians.

$$
\mathcal{L}_{\text{bound}}=\mathcal{L}_{\text{photo}}+\mathcal{R}-0.001\operatorname{mean}(\alpha)
$$

The free stage adds point-to-triangle distance with weight 10, a scale regularizer, and opacity entropy.

$$
\mathcal{L}_{\text{free}}=\mathcal{L}_{\text{photo}}+\mathcal{R}+10\mathcal{L}_{\text{dist}}+\mathcal{L}_{\text{scale}}-0.001\operatorname{mean}(\alpha\log\alpha)
$$

## Surface binding

Each exported Gaussian binds to the ten nearest rest-face centers. The weights are normalized inverse distances with $\varepsilon=10^{-8}$.

$$
w_k=\frac{1/\max(d_k,\varepsilon)}{\sum_l 1/\max(d_l,\varepsilon)}
$$

For rest center $\mathbf{c}_k$, posed center $\mathbf{c}'_k$, and face rotation $\mathbf{R}_k$, the Gaussian center becomes

$$
\boldsymbol{\mu}'=\sum_k w_k\left[\mathbf{c}'_k+\mathbf{R}_k(\boldsymbol{\mu}-\mathbf{c}_k)\right]
$$

Gaussian rotations use hemisphere-aligned normalized quaternion blending. Gaussian scale uses the weighted square root of each posed-to-rest face perimeter ratio. Degenerate faces retain centroid translation with identity rotation and unit scale.

## Motion curves

Authored motion uses quintic smootherstep interpolation.

$$
h(t)=6t^5-15t^4+10t^3
$$

The first and second derivatives are zero at both endpoints, reducing visible snapping at action boundaries.
