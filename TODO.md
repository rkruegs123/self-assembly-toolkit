# Team Livia and Ryan

## October 3, 2023

Livia:
- get on cluster
- try to get `dimers_refactor/compute.py` running
  - will probably want to use your updated `jax_transformations_3d` code
  - when you change that, reproduce the old plot, make sure it's the same
- futz around with stuff if you want -- e.g. make things branchless, try to take a forward gradient
  - it would be helpful for you to just see what goes wrong

When you feel ready, let me know, and then we can do the following...

Ryan and Livia:
- pair program a simple forwards gradient
  - fix issues as we run into them
- note: this will use forwards-mode automatic differentiation, and will *not* JIT compile
  - see the JAX docs for what exactly this means and why this is important
- once we prove that we can take grads w.r.t. this procedure via the crude forwards-mode, then we will go off and try to make it so that we can use reverse-mode and JIT
