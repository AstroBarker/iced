#!/usr/bin/env python3
# finite-volume implementation of the inviscid Burger's
#
# We are solving u_t + u u_x = nu u_xx
#
# Author : Brandon L. Barker
# FV/WENO implementation based on M. Zingale (2013-03-26)
# See: https://github.com/python-hydro/hydro_examples/burers

import argparse
from enum import Enum
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

from solvers import Solvers
import weno_coefficients


def norm(q, p=2):
  """
  return the p-norm of quantity q
  """

  return np.sqrt(np.sum(np.power(q, p)) / len(q))


def tanh_solution(x, t, nu):
  return 1.0 - np.tanh((x - t) / (2.0 * nu))


def tanh_laplacian(x, t, nu):
  return np.tanh((x - t) / (2.0 * nu)) / (
    2.0 * nu * nu * np.power(np.cosh((t - x) / (2.0 * nu)), 2.0)
  )


def weno(order, q):
  """
  Do WENO reconstruction

  Parameters:

    order : int
      The stencil width
    q : numpy array
      Scalar data to reconstruct

  Returns:

    qL : numpy array
      Reconstructed data - boundary points are zero
  """
  C = weno_coefficients.C_all[order]
  a = weno_coefficients.a_all[order]
  sigma = weno_coefficients.sigma_all[order]

  qL = np.zeros_like(q)
  beta = np.zeros((order, len(q)))
  w = np.zeros_like(beta)
  num_p = len(q) - 2 * order
  epsilon = 1e-16
  for i in range(order, num_p + order):
    q_stencils = np.zeros(order)
    alpha = np.zeros(order)
    for k in range(order):
      for l in range(order):
        for m in range(l + 1):
          beta[k, i] += sigma[k, l, m] * q[i + k - l] * q[i + k - m]
      alpha[k] = C[k] / (epsilon + beta[k, i] ** 2)
      for l in range(order):
        q_stencils[k] += a[k, l] * q[i + k - l]
    w[:, i] = alpha / np.sum(alpha)
    qL[i] = np.dot(w[:, i], q_stencils)

  return qL


class Coupling(Enum):
  Strang = 0
  MOL = 1


# End Coupling


class TableauType(Enum):
  Implicit = 0
  Explicit = 1


# End TableauType


class Reconstruction(Enum):
  PLM = 0
  WENO = 1


# End Reconstruction


class Tableau:
  """
  Class for Runge Kutta tableau.

  Args:
    nStages (int) : number of RK stages
    tOrder (int) : convergence orger
    tableau_type (TableauType) enum for specifying implicit/explicit

  """

  def __init__(self, nStages, tOrder, tableau_type):
    self.nStages = nStages
    self.tOrder = tOrder
    self.tableau_type = tableau_type

    # setup tableau

    self.a_ij = np.zeros((nStages, nStages))
    self.b_i = np.zeros(nStages)

    if self.tableau_type == TableauType.Explicit:
      # forward euler
      if nStages == 1 and tOrder == 1:
        self.a_ij[0, 0] = 0.0
        self.b_i[0] = 1.0

      elif nStages == 2 and tOrder == 2:
        self.a_ij[1, 0] = 1.0
        self.b_i[0] = 0.5
        self.b_i[1] = 0.5

      # SSPRK(3,3) of Shu Osher
      elif nStages == 3 and tOrder == 3:
        self.a_ij[1, 0] = 1.0
        self.a_ij[2, 0] = 0.25
        self.a_ij[2, 1] = 0.25
        self.b_i[0] = 1.0 / 6.0
        self.b_i[1] = 1.0 / 6.0
        self.b_i[2] = 2.0 / 3.0

      elif nStages == 5 and tOrder == 4:
        self.a_ij[0, 0] = 0.0
        self.a_ij[1, 0] = 0.51047914
        self.a_ij[2, 0] = 0.0851508
        self.a_ij[3, 0] = 0.299021
        self.a_ij[4, 0] = 0.01438455
        self.a_ij[2, 1] = 0.21940489
        self.a_ij[3, 1] = 0.07704762
        self.a_ij[4, 1] = 0.03706414
        self.a_ij[3, 2] = 0.46190055
        self.a_ij[4, 2] = 0.22219957
        self.a_ij[3, 4] = 0.63274729
        self.b_i[0] = 0.12051432
        self.b_i[1] = 0.22614012
        self.b_i[2] = 0.27630606
        self.b_i[3] = 0.12246455
        self.b_i[4] = 0.25457495

      else:
        raise ValueError(
          "Oops! Pick a valid IMEX RK scheme (1,1), (2,2), (3,3), (5,4)"
        )

    if self.tableau_type == TableauType.Implicit:
      # Backward Euler tableau
      if nStages == 1 and tOrder == 1:
        self.a_ij[0, 0] = 1.0
        self.b_i[0] = 1.0

      elif nStages == 2 and tOrder == 2:
        self.a_ij[0, 0] = 0.71921758
        self.a_ij[1, 0] = 0.11776435
        self.a_ij[1, 1] = 0.16301806
        self.b_i[0] = 0.5
        self.b_i[1] = 0.5

      # L-stable
      elif nStages == 3 and tOrder == 3:
        self.a_ij[2, 0] = 1.0 / 6.0
        self.a_ij[1, 1] = 1.0
        self.a_ij[2, 1] = -1.0 / 3.0
        self.a_ij[2, 2] = 2.0 / 3.0
        self.b_i[0] = 1.0 / 6.0
        self.b_i[1] = 1.0 / 6.0
        self.b_i[2] = 2.0 / 3.0

      elif nStages == 5 and tOrder == 4:
        self.a_ij[0, 0] = 1.03217796e-16  # just 0?
        self.a_ij[1, 0] = 0.510479144
        self.a_ij[2, 0] = 5.06048136e-3
        self.a_ij[3, 0] = 8.321807e-2
        self.a_ij[4, 0] = 7.56636565e-2
        self.a_ij[1, 1] = 1.00124199e-14
        self.a_ij[2, 1] = 1.00953283e-1
        self.a_ij[3, 1] = 1.60838280e-1
        self.a_ij[4, 1] = 1.25319139e-1
        self.a_ij[2, 2] = 1.98541931e-1
        self.a_ij[3, 2] = 3.28641063e-1
        self.a_ij[4, 2] = 7.08147871e-2
        self.a_ij[3, 3] = -3.84714236e-3
        self.a_ij[4, 3] = 6.34597980e-1
        self.a_ij[4, 4] = -7.22101223e-17
        self.b_i[0] = 0.12051432
        self.b_i[1] = 0.22614012
        self.b_i[2] = 0.27630606
        self.b_i[3] = 0.12246455
        self.b_i[4] = 0.25457495

      else:
        raise ValueError(
          "Oops! Pick a valid IMEX RK scheme (1,1), (2,2), (3,3), (5,4)"
        )

  # End __init__

  def __str__(self):
    print("RK method: ")
    print(f"nStages : {self.nStages}")
    print(f"tOrder  : {self.tOrder}")
    print(f"Type    : {self.tableau_type}")
    return ""

  # End __str__


class Grid(object):
  def __init__(self, nx, ng, xmin=0.0, xmax=1.0, bc="outflow"):
    self.nx = nx
    self.ng = ng

    self.xmin = xmin
    self.xmax = xmax

    self.bc = bc

    # grid limits
    self.ilo = ng
    self.ihi = ng + nx - 1

    # physical coords -- cell-centered, left and right edges
    self.dx = (xmax - xmin) / (nx)
    self.x = xmin + (np.arange(nx + 2 * ng) - ng + 0.5) * self.dx

    # storage for the solution
    self.u = np.zeros((nx + 2 * ng), dtype=np.float64)
    # interface states.
    self.ul = self.scratch_array()
    self.ur = self.scratch_array()
    # laplacian
    self.laplacian = np.zeros((nx + 2 * ng), dtype=np.float64)

  def scratch_array(self):
    """
    return a scratch array dimensioned for our grid
    """
    return np.zeros((self.nx + 2 * self.ng), dtype=np.float64)

  def fill_bcs(self, u):
    """
    fill all ghostcells as periodic
    """

    if self.bc == "periodic":
      # left boundary
      u[0 : self.ilo] = self.u[self.ihi - self.ng + 1 : self.ihi + 1]

      # right boundary
      u[self.ihi + 1 :] = self.u[self.ilo : self.ilo + self.ng]

    elif self.bc == "outflow":
      # left boundary
      u[0 : self.ilo] = self.u[self.ilo]

      # right boundary
      u[self.ihi + 1 :] = self.u[self.ihi]

    else:
      print("Invalid boundary condition!")
      sys.exit(os.EX_SOFTWARE)


class ConvectionDiffusionModel(object):
  def __init__(
    self,
    grid,
    nu=0.0,
    nStages=1,
    tOrder=1,
    recon=Reconstruction.WENO,
    weno_order=2,
    coupling=Coupling.MOL,
  ):
    self.grid = grid
    self.t = 0.0
    self.nu = nu  # viscocity

    self.nStages = nStages
    self.tOrder = tOrder

    self.recon = recon
    self.weno_order = weno_order

    self.coupling = coupling

    # solver
    max_iters = 100
    fptol = 1.0e-16
    self.solver = Solvers(max_iters, fptol)

    # stage data
    self.u_s = np.zeros(
      (nStages, self.grid.nx + 2 * self.grid.ng), dtype=np.float64
    )

    self.explicit_tableau = Tableau(nStages, tOrder, TableauType.Explicit)
    self.implicit_tableau = Tableau(nStages, tOrder, TableauType.Implicit)

  # End __init__

  def __str__(self):
    # print("Simulation parameters: ")
    print(f"nStages  : {self.nStages}")
    print(f"tOrder   : {self.tOrder}")
    print(f"weno     : {self.weno_order}")
    print(f"Recon    : {self.recon}")
    print(f"Coupling : {self.coupling}")
    print(f"nu       : {self.nu}")
    return ""

  # End __str__

  def init_cond(self, sim_type="tophat"):
    if sim_type == "tophat":
      self.grid.u[
        np.logical_and(self.grid.x >= 0.333, self.grid.x <= 0.666)
      ] = 1.0

    elif sim_type == "sine":
      self.grid.u[:] = 1.0

      index = np.logical_and(self.grid.x >= 0.333, self.grid.x <= 0.666)
      self.grid.u[index] += 0.5 * np.sin(
        2.0 * np.pi * (self.grid.x[index] - 0.333) / 0.333
      )

    elif sim_type == "rarefaction":
      self.grid.u[:] = 1.0
      self.grid.u[self.grid.x > 0.5] = 2.0
    elif sim_type == "tanh":
      self.grid.u[:] = tanh_solution(self.grid.x[:], 0.0, self.nu)
    else:
      raise ValueError("Invalid sim type!")

  def hyperbolic_timestep(self, C):
    return (
      C
      * self.grid.dx
      / np.max(np.abs(self.grid.u[self.grid.ilo : self.grid.ihi + 1]))
    )

  def states(self, u, dt):
    if self.recon == Reconstruction.PLM:
      self.states_plm(u, dt)
    else:
      raise NotImplementedError("Other reconstructions not yet implemented!")

  def states_plm(self, u, dt):
    """
    compute the left and right interface states using piecewise linear reconstruction
    """

    g = self.grid
    # compute the piecewise linear slopes via 2nd order MC limiter
    # includes 1 ghost cell on either side
    ib = g.ilo - 1
    ie = g.ihi + 1

    # u = g.u

    # this is the MC limiter from van Leer (1977), as given in
    # LeVeque (2002).  Note that this is slightly different than
    # the expression from Colella (1990)

    # TODO: move scratch data into grid.
    dc = g.scratch_array()
    dl = g.scratch_array()
    dr = g.scratch_array()

    dc[ib : ie + 1] = 0.5 * (u[ib + 1 : ie + 2] - u[ib - 1 : ie])
    dl[ib : ie + 1] = u[ib + 1 : ie + 2] - u[ib : ie + 1]
    dr[ib : ie + 1] = u[ib : ie + 1] - u[ib - 1 : ie]

    # these where's do a minmod()
    d1 = 2.0 * np.where(np.fabs(dl) < np.fabs(dr), dl, dr)
    d2 = np.where(np.fabs(dc) < np.fabs(d1), dc, d1)
    ldeltau = np.where(dl * dr > 0.0, d2, 0.0)

    # reset interface states
    self.grid.ul[:] = 0.0
    self.grid.ur[:] = 0.0

    self.grid.ur[ib : ie + 2] = (
      u[ib : ie + 2]
      - 0.5 * (1.0 + u[ib : ie + 2] * dt / self.grid.dx) * ldeltau[ib : ie + 2]
    )

    self.grid.ul[ib + 1 : ie + 2] = (
      u[ib : ie + 1]
      + 0.5 * (1.0 - u[ib : ie + 1] * dt / self.grid.dx) * ldeltau[ib : ie + 1]
    )

  # End states

  def parabolic_term_fd(self, u):
    """
    finite difference approximation for the parabolic diffusion term
    (u_{i+1} - 2 u_{i} + u_{i-1}) / dx**2
    """

    ilo = self.grid.ilo
    ihi = self.grid.ihi

    # TODO: better than this
    self.grid.laplacian[ilo - 1 : ihi + 2] = np.diff(u[ilo - 2 : ihi + 3], 2)
    return self.nu * self.grid.laplacian / (self.grid.dx * self.grid.dx)

  # End parabolic_term_fd

  def riemann(self):  # , ul, ur):
    """
    Riemann problem for Burgers' equation.
    """

    S = 0.5 * (self.grid.ul + self.grid.ur)
    ushock = np.where(S > 0.0, self.grid.ul, self.grid.ur)
    ushock = np.where(S == 0.0, 0.0, ushock)

    # rarefaction solution
    urare = np.where(self.grid.ur <= 0.0, self.grid.ur, 0.0)
    urare = np.where(self.grid.ul >= 0.0, self.grid.ul, urare)

    us = np.where(self.grid.ul > self.grid.ur, ushock, urare)

    return 0.5 * us * us

  def burgers_flux(self, q):
    return 0.5 * q**2

  def rk_substep(self, u):
    g = self.grid
    g.fill_bcs(u)
    f = self.burgers_flux(u)
    alpha = np.max(np.abs(u))
    fp = (f + alpha * u) / 2.0
    fm = (f - alpha * u) / 2.0
    fpr = g.scratch_array()
    fml = g.scratch_array()
    flux = g.scratch_array()
    fpr[1:] = weno(self.weno_order, fp[:-1])
    fml[-1::-1] = weno(self.weno_order, fm[-1::-1])
    flux[1:-1] = fpr[1:-1] + fml[1:-1]
    rhs = g.scratch_array()
    # rhs[1:-1] = 1 / g.dx * (flux[1:-1] - flux[2:])
    rhs[g.ilo : g.ihi + 1] = (
      1 / g.dx * (flux[g.ilo : g.ihi + 1] - flux[g.ilo + 1 : g.ihi + 2 :])
    )
    return rhs

  def update_imex(self, dt):
    """
    Fully coupled (non splitting) IMEX SSPRK update
    """

    g = self.grid
    f = self.parabolic_term_fd

    # cleanup
    self.u_s[:, :] = 0.0
    self.u_s[0, :] = self.grid.u[:]

    sum_im1 = g.scratch_array()
    for i in range(self.nStages):
      # Solve u^(i) = dt a_ii f(u^(i))
      sum_im1[:] = self.grid.u[:]
      for j in range(i):
        g.fill_bcs(self.u_s[j])
        sum_im1 += (
          dt
          * self.explicit_tableau.a_ij[i, j]
          * self.rk_substep(self.u_s[j, :])
        )
        sum_im1 += dt * self.implicit_tableau.a_ij[i, j] * f(self.u_s[j])

      # self.u_s[i, :] = sum_im1
      def target(u):
        return sum_im1 + dt * self.implicit_tableau.a_ij[i, i] * f(u)

      # Real?
      self.u_s[i, :] = self.solver.fixed_point(target, self.u_s[i, :])

    # u^(n+1) from the stages
    for i in range(self.nStages):
      g.u += dt * self.explicit_tableau.b_i[i] * self.rk_substep(self.u_s[i, :])
      g.u += dt * self.implicit_tableau.b_i[i] * f(self.u_s[i, :])

    return self.grid.u

  # End update_imex

  def update_explicit(self, dt):
    """
    explicit conservative finite volume update
    TODO: remove unnecessary alloc
    """

    # cleanup
    self.u_s[:, :] = 0.0
    self.u_s[0, :] = self.grid.u[:]

    def fv(u):
      # get the interface states
      # ul, ur = self.states(dt)
      self.states(u, dt)

      # solve the Riemann problem at all interfaces
      # flux = self.riemann(ul, ur)
      flux = self.riemann()

      return (1.0 / g.dx) * (
        flux[g.ilo : g.ihi + 1] - flux[g.ilo + 1 : g.ihi + 2]
      )

    g = self.grid

    # unew = self.grid.u.copy()

    sum_im1 = g.scratch_array()
    for i in range(self.nStages):
      # Solve u^(i) = dt a_ii f(u^(i))
      sum_im1[:] = self.grid.u[:]
      for j in range(i):
        g.fill_bcs(self.u_s[j])
        # sum_im1[g.ilo : g.ihi + 1] += (
        sum_im1 += (
          dt
          * self.explicit_tableau.a_ij[i, j]
          * self.rk_substep(self.u_s[j, :])
        )
      # sum_im1 += self.grid.u[g.ilo : g.ihi + 1]

      # self.u_s[i][g.ilo : g.ihi + 1] = sum_im1
      self.u_s[i] = sum_im1

    # u^(n+1) from the stages
    for i in range(self.nStages):
      # self.grid.u[g.ilo : g.ihi + 1] += (
      self.grid.u += (
        dt * self.explicit_tableau.b_i[i] * self.rk_substep(self.u_s[i, :])
      )

    # unew = self.grid.u

    return self.grid.u

  def update_implicit(self, dt):
    """
    Given DIRK tableau and rhs function f of du/dt = f(u), compute u^(n+1)
    """

    # cleanup
    self.u_s[:, :] = 0.0
    self.u_s[0, :] = self.grid.u[:]
    g = self.grid

    f = self.parabolic_term_fd

    sum_im1 = g.scratch_array()
    for i in range(self.nStages):
      # Solve u^(i) = dt a_ii f(u^(i))
      sum_im1[:] = g.u[:]
      for j in range(i):
        sum_im1 += dt * self.implicit_tableau.a_ij[i, j] * f(self.u_s[j])

      def target(u):
        return sum_im1 + dt * self.implicit_tableau.a_ij[i, i] * f(u)

      self.u_s[i, :] = self.solver.fixed_point(target, self.u_s[i, :])

    # u^(n+1) from the stages
    for i in range(self.nStages):
      g.u += dt * self.implicit_tableau.b_i[i] * f(self.u_s[i, :])

    return self.grid.u

  # End update_implicit

  def evolve(self, C, tmax):
    self.t = 0.0

    g = self.grid

    # main evolution loop
    while self.t < tmax:
      # fill the boundary conditions
      g.fill_bcs(self.grid.u)

      # get the hyperbolic_timestep
      dt = self.hyperbolic_timestep(C)

      if self.t + dt > tmax:
        dt = tmax - self.t

      # The actual update
      # 1. Strang Split
      if self.coupling == Coupling.Strang:
        self.grid.u = self.update_implicit(0.5 * dt)
        self.grid.u = self.update_explicit(dt)
        self.grid.u = self.update_implicit(0.5 * dt)
      if self.coupling == Coupling.MOL:
        self.grid.u = self.update_imex(dt)

      self.t += dt

  # End evolve


def main():
  # === Argparse ===
  parser = argparse.ArgumentParser(
    prog="iced", description="IMEX Convection Diffusion model"
  )
  parser.add_argument(
    "-c",
    "--convergence",
    help="Run a convergence test",
    action="store_true",
    default=False,
  )
  parser.add_argument(
    "-n", "--n_stages", help="IMEX RK stages", default=3, nargs="?", type=int
  )
  parser.add_argument(
    "-t", "--t_order", help="IMEX RK order", default=3, nargs="?", type=int
  )
  parser.add_argument(
    "-w", "--weno_order", help="WENO order", default=3, nargs="?", type=int
  )
  parser.add_argument(
    "-nx", "--nx", help="Number of cells", default=128, nargs="?", type=int
  )
  parser.add_argument(
    "-nu",
    "--nu",
    help="Constant coefficient viscocity",
    default=0.01,
    nargs="?",
    type=float,
  )
  parser.add_argument(
    "-m",
    "--method",
    help="Physics coupling method (MOL or Strang)",
    default="MOL",
    type=str,
    nargs="?",
  )
  parser.add_argument(
    "problems", nargs="*", default="sine", help="Problems to run"
  )
  args = parser.parse_args()

  # some checking
  valid_n_stages = [1, 2, 3, 4, 5]
  valid_t_order = [1, 2, 3, 4]
  valid_weno = [2, 3, 4, 5, 6, 7]
  valid_problems = ["sine", "tanh", "tophat", "rarefaction"]
  valid_couplings = ["strang", "mol"]

  if args.nu < 0.0:
    raise ValueError("Please enter a non-negative viscocity.")
  if args.nx <= 0:
    raise ValueError("Please enter a vaid number of cells")
  if args.n_stages not in valid_n_stages:
    raise ValueError("Please enter a valid number of RK stages ([1,2,3,4,5])")
  if args.t_order not in valid_t_order:
    raise ValueError("Please enter a valid RK order ([1,2,3,4]]")
  if args.weno_order not in valid_weno:
    raise ValueError("Please enter a valid WENO order ([2,3,4,5,6,7])")
  if args.method.lower() not in valid_couplings:
    raise ValueError("Please select a valid coupling (strang or mol)")
  if not args.convergence:
    for p in args.problems:
      if p.lower() not in valid_problems:
        raise ValueError(
          "Please enter a valid problem (sine, rarefaction, tanh, tophat)"
        )

  weno_order = args.weno_order
  nStages = args.n_stages
  tOrder = args.t_order
  nx = args.nx
  nu = args.nu
  coupling = Coupling.MOL if args.method.lower() == "mol" else Coupling.Strang

  # sine problem
  if "sine" in args.problems and not args.convergence:
    print("Running sine problem with following parameters..")
    xmin = 0.0
    xmax = 1.0
    ng = weno_order + 1
    g = Grid(nx, ng, bc="periodic")

    # maximum evolution time based on period for unit velocity
    tmax = (xmax - xmin) / 1.0

    C = 0.7

    plt.clf()

    s = ConvectionDiffusionModel(
      g,
      nu=nu,
      nStages=nStages,
      tOrder=tOrder,
      weno_order=weno_order,
      coupling=coupling,
    )
    print(s)

    t0 = time.time()
    for i in range(0, 10):
      tend = (i + 1) * 0.02 * tmax
      s.init_cond("sine")

      uinit = s.grid.u.copy()

      s.evolve(C, tend)

      c = 1.0 - (0.1 + i * 0.1)
      g = s.grid
      plt.plot(g.x[g.ilo : g.ihi + 1], g.u[g.ilo : g.ihi + 1], color=str(c))
    t1 = time.time()
    print(f"Sine time : {t1-t0}")

    g = s.grid
    plt.plot(
      g.x[g.ilo : g.ihi + 1],
      uinit[g.ilo : g.ihi + 1],
      ls=":",
      color="0.9",
      zorder=-1,
    )

    plt.xlabel("$x$")
    plt.ylabel("$u$")
    plt.savefig("fv-burger-sine.pdf")

  # rarefaction
  if "rarefaction" in args.problems and not args.convergence:
    print("Running rarefaction problem with following parameters..")
    xmin = 0.0
    xmax = 1.0
    ng = weno_order + 1
    g = Grid(nx, ng, bc="outflow")

    # maximum evolution time based on period for unit velocity
    tmax = (xmax - xmin) / 1.0

    C = 0.5

    plt.clf()

    s = ConvectionDiffusionModel(
      g,
      nu=nu,
      nStages=nStages,
      tOrder=tOrder,
      weno_order=weno_order,
      coupling=coupling,
    )
    print(s)

    t0 = time.time()
    for i in range(0, 10):
      tend = (i + 1) * 0.02 * tmax

      s.init_cond("rarefaction")

      uinit = s.grid.u.copy()

      s.evolve(C, tend)

      c = 1.0 - (0.1 + i * 0.1)
      plt.plot(g.x[g.ilo : g.ihi + 1], g.u[g.ilo : g.ihi + 1], color=str(c))
    t1 = time.time()
    print(f"Rarefaction time : {t1-t0}")

    plt.plot(
      g.x[g.ilo : g.ihi + 1],
      uinit[g.ilo : g.ihi + 1],
      ls=":",
      color="0.9",
      zorder=-1,
    )

    plt.xlabel("$x$")
    plt.ylabel("$u$")

    plt.savefig("fv-burger-rarefaction.pdf")

  # tophat
  if "tophat" in args.problems and not args.convergence:
    print("Running tophat problem with following parameters..")
    xmin = 0.0
    xmax = 1.0
    ng = weno_order + 1
    g = Grid(nx, ng, bc="outflow")

    # maximum evolution time based on period for unit velocity
    tmax = (xmax - xmin) / 1.0

    C = 0.5

    plt.clf()

    s = ConvectionDiffusionModel(
      g,
      nu=nu,
      nStages=nStages,
      tOrder=tOrder,
      weno_order=weno_order,
      coupling=coupling,
    )
    print(s)

    t0 = time.time()
    for i in range(0, 10):
      tend = (i + 1) * 0.02 * tmax

      s.init_cond("tophat")

      uinit = s.grid.u.copy()

      s.evolve(C, tend)

      c = 1.0 - (0.1 + i * 0.1)
      plt.plot(g.x[g.ilo : g.ihi + 1], g.u[g.ilo : g.ihi + 1], color=str(c))
    t1 = time.time()
    print(f"Tophat time : {t1-t0}")

    plt.plot(
      g.x[g.ilo : g.ihi + 1],
      uinit[g.ilo : g.ihi + 1],
      ls=":",
      color="0.9",
      zorder=-1,
    )

    plt.xlabel("$x$")
    plt.ylabel("$u$")

    plt.savefig("fv-burger-tophat.pdf")

  # tanh
  if "tanh" in args.problems and not args.convergence:
    print("Running tanh problem with following parameters..")
    xmin = -10.0
    xmax = 10.0
    ng = weno_order + 1
    g = Grid(nx, ng, xmin=xmin, xmax=xmax, bc="outflow")

    # maximum evolution time based on period for unit velocity
    tmax = (xmax - xmin) / 1.0
    tmax = 5.0 / 2.0

    C = 0.3

    plt.clf()

    s = ConvectionDiffusionModel(
      g,
      nu=nu,
      nStages=nStages,
      tOrder=tOrder,
      weno_order=weno_order,
      coupling=coupling,
    )
    print(s)

    t0 = time.time()
    for i in range(0, 10):
      tend = (i + 1) * 0.1 * tmax

      s.init_cond("tanh")

      uinit = s.grid.u.copy()

      s.evolve(C, tend)

      c = 1.0 - (0.1 + i * 0.1)
      plt.plot(g.x[g.ilo : g.ihi + 1], g.u[g.ilo : g.ihi + 1], color=str(c))
    t1 = time.time()
    print(f"Tanh time : {t1-t0}")

    plt.plot(
      g.x[g.ilo : g.ihi + 1],
      uinit[g.ilo : g.ihi + 1],
      ls=":",
      color="0.9",
      zorder=-1,
    )

    sol = tanh_solution(g.x[g.ilo : g.ihi + 1], tmax, nu)
    plt.plot(
      g.x[g.ilo : g.ihi + 1],
      sol,
      ls="--",
      color="teal",
    )

    print(np.linalg.norm(np.abs(sol - g.u[g.ilo : g.ihi + 1])))

    plt.xlabel("$x$")
    plt.ylabel("$u$")

    plt.savefig("fv-burger-tanh.pdf")

  # WENO Convergence
  if args.convergence:
    xmin = -10.0
    xmax = 10.0
    nxs = np.array([128, 192, 256, 384, 512, 768])
    weno_order = np.array([2, 3, 4])
    n1 = len(nxs)
    n2 = len(weno_order)
    errors = np.zeros((n2, n1))
    ng = weno_order + 1
    nStages = 5
    tOrder = 4

    tend = 5.0
    C = 0.3
    nu = 0.15
    for i in range(len(weno_order)):
      w = weno_order[i]
      for j in range(len(nxs)):
        nx = nxs[j]
        print(f"weno{w}, nx = {nx}...")

        g = Grid(nx, ng[i], xmin=xmin, xmax=xmax, bc="outflow")
        s = ConvectionDiffusionModel(
          g,
          nu=nu,
          nStages=nStages,
          tOrder=tOrder,
          weno_order=w,
          coupling=coupling,
        )

        s.init_cond("tanh")
        uinit = s.grid.u.copy()

        s.evolve(C, tend)

        solution = tanh_solution(s.grid.x[g.ilo : g.ihi + 1], tend, nu)
        errors[i, j] = norm(s.grid.u[g.ilo : g.ihi + 1] - solution)

    plt.clf()
    fig, ax = plt.subplots()
    for i in range(len(weno_order)):
      ax.loglog(nxs, errors[i, :], label="weno"+str(weno_order[i]), marker="s", ls=" ")
    print(errors)
    ax.legend(frameon=True)
    ax.set(xlabel=r"N$_{x}$", ylabel=r"L$_{2}$")
    plt.savefig("conv.png")


# End main

if __name__ == "__main__":
  main()
