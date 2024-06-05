# 2nd-order accurate finite-volume implementation of the inviscid Burger's
# equation with piecewise linear slope reconstruction
#
# We are solving u_t + u u_x = nu u_xx with outflow boundary conditions
#
# FV implementation based on M. Zingale (2013-03-26)

from enum import Enum
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt


class TableauType(Enum):
  Implicit = 0
  Explicit = 1


# End TableauType


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
        self.a_ij[0,0] = 0.0
        self.b_i[0] = 1.0

      # Ralston (minimum error Huen)
      if nStages == 2 and tOrder == 2:
        self.a_ij[1,0] = 2.0 / 3.0
        self.b_i[0] = 0.25
        self.b_i[1] = 0.75

      # SSPRK(3,3)
      if nStages == 3 and tOrder == 3:
        self.a_ij[1,0] = 1.0
        self.a_ij[2,0] = 0.25
        self.a_ij[2,1] = 0.25
        self.b_i[0] = 1.0 / 6.0
        self.b_i[1] = 1.0 / 6.0
        self.b_i[2] = 2.0 / 3.0

    if self.tableau_type == TableauType.Implicit:
      # Backward Euler tableau
      if nStages == 1 and tOrder == 1:
        self.a_ij[0, 0] = 1.0
        self.b_i[0] = 1.0

      # L-stable
      if nStages == 2 and tOrder == 2:
        x = 1.0 - np.sqrt(2.0) / 2.0
        self.a_ij[0, 0] = x
        self.a_ij[1, 0] = 1.0 - x
        self.a_ij[1, 1] = x
        self.b_i[0] = 1.0 - x
        self.b_i[1] = x

      # L-stable
      if nStages == 3 and tOrder == 3:
        x = 0.4358665215
        self.a_ij[0, 0] = x
        self.a_ij[1, 0] = (1.0 - x) / 2.0
        self.a_ij[2, 0] = (-3.0 * x * x / 2.0) + 4.0 * x - 0.25
        self.a_ij[1, 1] = x
        self.a_ij[2, 1] = (3.0 * x * x / 2.0) - 5.0 * x + 5.0 / 4.0
        self.a_ij[2, 2] = x
        self.b_i[0] = (-3.0 * x * x / 2.0) + 4.0 * x - 0.25
        self.b_i[1] = (3.0 * x * x / 2.0) - 5.0 * x + 5.0 / 4.0
        self.b_i[2] = x

      # L-stable
      if nStages == 4 and tOrder == 3:
        self.a_ij[0, 0] = 0.5
        self.a_ij[1, 0] = 1.0 / 6.0
        self.a_ij[2, 0] = -0.5
        self.a_ij[3, 0] = 1.5
        self.a_ij[1, 1] = 0.5
        self.a_ij[2, 1] = 0.5
        self.a_ij[3, 1] = -1.5
        self.a_ij[2, 2] = 0.5
        self.a_ij[3, 2] = 0.5
        self.a_ij[3, 3] = 0.5
        self.b_i[0] = 1.5
        self.b_i[1] = -1.5
        self.b_i[2] = 0.5
        self.b_i[3] = 0.5

      if nStages == 3 and tOrder == 4:
        x = 1.06858
        self.a_ij[0, 0] = x
        self.a_ij[1, 0] = 0.5 - x
        self.a_ij[2, 0] = 2.0 * x
        self.a_ij[1, 1] = x
        self.a_ij[2, 1] = 1.0 - 4.0 * x
        self.a_ij[2, 2] = x
        self.b_i[0] = 1.0 / (6.0 * (1.0 - 2.0 * x) ** 2.0)
        self.b_i[1] = (3.0 * (1.0 - 2.0 * x) ** 2 - 1.0) / (
          3.0 * (1.0 - 2.0 * x) ** 2.0
        )
        self.b_i[2] = 1.0 / (6.0 * (1.0 - 2.0 * x) ** 2.0)

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

  def norm(self, e, p=2):
    """
    return the p-norm of quantity e which lives on the grid
    """
    assert len(e) == 2 * self.ng + self.nx, "Error in norm"

    return np.sqrt(self.dx * np.sum(np.power(e[self.ilo : self.ihi + 1], p)))


class ConvectionDiffusionModel(object):
  def __init__(self, grid, nu=0.0, nStages=1, tOrder=1):
    self.grid = grid
    self.t = 0.0
    self.nu = nu  # viscocity

    self.nStages = nStages
    self.tOrder = tOrder

    # stage data
    self.u_s = np.zeros((nStages, self.grid.nx + 2 * self.grid.ng), dtype=np.float64)

    self.explicit_tableau = Tableau(nStages, tOrder, TableauType.Explicit)

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

  def hyperbolic_timestep(self, C):
    return (
      C
      * self.grid.dx
      / np.max(np.abs(self.grid.u[self.grid.ilo : self.grid.ihi + 1]))
    )

  def states(self, u, dt):
    """
    compute the left and right interface states
    """

    g = self.grid
    # compute the piecewise linear slopes via 2nd order MC limiter
    # includes 1 ghost cell on either side
    ib = g.ilo - 1
    ie = g.ihi + 1

    #u = g.u

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

  def parabolic_term_fd(self):
    """
    finite difference approximation for the parabolic diffusion term
    (u_{i+1} - 2 u_{i} + u_{i-1}) / dx**2
    """

    ilo = self.grid.ilo
    ihi = self.grid.ihi

    # TODO: better than this
    self.grid.laplacian[ilo - 1 : ihi + 2] = np.diff(self.grid.u, 2)

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

  def update_explicit(self, dt):
    """
    explicit conservative finite volume update
    TODO: remove unnecessary alloc
    """

    # cleanup
    self.u_s[:,:] = 0.0
    self.u_s[0,:] = self.grid.u[:]

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

    #unew = self.grid.u.copy()

    sum_im1 = g.scratch_array()
    for i in range(self.nStages):
      # Solve u^(i) = dt a_ii f(u^(i))
      sum_im1[:] = self.grid.u[:]
      for j in range(i):
        g.fill_bcs(self.u_s[j])
        sum_im1[g.ilo : g.ihi + 1] += dt * self.explicit_tableau.a_ij[i, j] * fv(self.u_s[j,:])
      #sum_im1 += self.grid.u[g.ilo : g.ihi + 1]

      #self.u_s[i][g.ilo : g.ihi + 1] = sum_im1
      self.u_s[i] = sum_im1

    # u^(n+1) from the stages
    for i in range(self.nStages):
      self.grid.u[g.ilo : g.ihi + 1] += dt * self.explicit_tableau.b_i[i] * fv(self.u_s[i,:])

    #unew = self.grid.u

    return self.grid.u

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


      # do the explicit conservative finite volume update
      self.parabolic_term_fd()
      self.grid.u = self.update_explicit(dt)

      # explicit finite difference the diffusive term
      self.grid.u += 1.0 * self.nu * self.grid.laplacian

      self.t += dt


if __name__ == "__main__":

  # Testing
  implicit_tableau = Tableau(1,1,TableauType.Explicit)

  # =======

  # -----------------------------------------------------------------------------
  # sine

  xmin = 0.0
  xmax = 1.0
  nx = 100
  ng = 2
  g = Grid(nx, ng, bc="periodic")

  # maximum evolution time based on period for unit velocity
  tmax = (xmax - xmin) / 1.0

  C = 0.5

  plt.clf()

  s = ConvectionDiffusionModel(g, nu=0.0, nStages = 3, tOrder = 3)

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

  # -----------------------------------------------------------------------------
  # rarefaction

  xmin = 0.0
  xmax = 1.0
  nx = 256
  ng = 2
  g = Grid(nx, ng, bc="outflow")

  # maximum evolution time based on period for unit velocity
  tmax = (xmax - xmin) / 1.0

  C = 0.1

  plt.clf()

  s = ConvectionDiffusionModel(g, nu=0.0)

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

  # -----------------------------------------------------------------------------
  # tophat

  xmin = 0.0
  xmax = 1.0
  nx = 256
  ng = 2
  g = Grid(nx, ng, bc="outflow")

  # maximum evolution time based on period for unit velocity
  tmax = (xmax - xmin) / 1.0

  C = 0.1

  plt.clf()

  s = ConvectionDiffusionModel(g, nu=0.00)

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
# End main
