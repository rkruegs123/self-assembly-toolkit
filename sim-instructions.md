# Instructions for running simulation stuff

Setting up environment:
1. `conda create -n "hoomd2" python=3.6`
2. `conda activate hoomd2`
3. `conda install -c conda-forge hoomd==2.9.7`
4. `conda install -c conda-forge gsd==2.4`
Then your environment is ready.

To run, navigate into `dimers/simulations`. Do the following once: `mkdir temp_results`. Then, you can run `RunRigidDimers_triang.py` and it will make new files for every unique set of parameters you run it with. Check on OVITO for visualization of the `.gsd` files.
