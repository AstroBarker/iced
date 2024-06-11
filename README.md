# iced
Imex ConvEction Diffusion (in Python 3.x)

Implements a simple convection diffusion model -- the viscous Burger's equation.
The purpose here is to prototype strong stability preserving (SSP) implicit-explicit (IMEX) additive Runge-Kutta (ARK) methods.
The spatial discretizations are not given that much attention. 
For the hyperbolic term, I use a finite volume approach with WENO reconstructions modified from [here](https://github.com/python-hydro/hydro_examples).
The diffusion term is simply finite differenced.
We implement fully coupled IMEX evolution using the linear-nonlinear (LNL) SSP IMEX methods presented in [Conde et al.](https://arxiv.org/abs/1702.04621). They present optimal SSP tableau pairs that have higher linear order than nonlinear order.

The viscious Burger's equation in one spatial dimension is 
$$\frac{\partial u}{\partial t} + \frac{1}{2} \frac{\partial u^2}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}$$
or, compactly, 
$$u' = F(u) + G(u)$$

where $F(u)$ is the "slow" evolving hyperbolic flux component that will be evolved explicitly and $G(u)$ is the diffusive component that will be evolved implicitly. An ARK iteration takes the form

$$u^{(i)} = u^{n} + \Delta t \sum_{j=1}^{i-1} a_{ij} F(u^{j}) + \Delta t \sum_{j=1}^{i} \tilde{a}_{ij} G(u^{j})$$
...
$$u^{n+1} = u^{n} + \Delta t \sum_{i=1}^{s} b_{i} F(u^{i}) + \Delta t \sum_{i=1}^{s} \tilde{b}_{i} G(u^{i})$$

## Literature References
- [Ascher, Ruuth, and Spiteri](https://www.sciencedirect.com/science/article/abs/pii/S0168927497000561) formulate IMEX methods.
- ARK methods are investigated for convection-diffusion-reactions problems by [Kennedy and Carpenter](https://www.sciencedirect.com/science/article/abs/pii/S0168927402001381). 
- Some SSP IMEX methods are developed by [Higueras et al](https://www.sciencedirect.com/science/article/pii/S0377042714002477?ref=cra_js_challenge&fr=RR-1), focusing on astrophysical interest.
- [Conde et al. 2017](https://arxiv.org/abs/1702.04621) present SSP IMEX methods with higher linear than nonlinear order (LNL methods).
- [Gottlieb et al. 2001](https://epubs.siam.org/doi/10.1137/S003614450036757X) is a nice overview of high order SSP methods.

### GARK methods
Generalizing the ARK methods here leads to GARK (G for generalized) methods.See [Sandu and Gunther](https://arxiv.org/abs/1310.5573)
These are further generalized to multirate GARK methods:
- [Gunther and Sandu](https://arxiv.org/abs/1310.6055)
- [Chinomona and Reynolds](https://arxiv.org/abs/2007.09776)
