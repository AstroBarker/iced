#!/usr/bin/env python3

# Real a Conde et al. 2017 LNL tableau
# see https://github.com/SSPmethods/imexLNL

import argparse
from os.path import join as pjoin

import scipy.io as sio


def main():
  # === Argparse ===
  parser = argparse.ArgumentParser(
    prog="read_matlab_table", description="Reads IMEX SSPRK tableaus from .mat"
  )

  parser.add_argument("filename", nargs="?", help=".mat filename")

  args = parser.parse_args()
  fn = args.filename

  # load fn
  datadir = "./"  # assumes they are local
  mat_fname = pjoin(datadir, fn)

  mat_contents = sio.loadmat(mat_fname)

  for key in sorted(mat_contents.keys()):
    print(key)
    print(mat_contents[key])
    print("===")


# End main

if __name__ == "__main__":
  main()
