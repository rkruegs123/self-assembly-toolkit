# self-assembly-toolkit: optimization

Current TODO:
- reorganize directory
- test that old dimers code still works
  - then move irrelevant stuff to bak
- test that current arvind code still works
  - move irrelevant stuff to bak
- make a `optimization/data` directory
  - refactor code to store relevant data in this directory, run things as submodules
  - test
- make notes on how to run things as submodules -- e.g. `python3 -m optimization.dimers.optimize`
- move onto what we actually care about:
  - Livia determined the correct values for `sigma`. This seems to resolve the optimization issues for solving concentration
  - So, we can compute a (seemingly) correct set of yield values. However, gradients of the loss w.r.t. parameters of the Hamiltonian are 0. Why?
  - note: will need to define an implicit derivative for the yield calculation. Also, can maybe use `jaxopt` if sigmas were the issue after all