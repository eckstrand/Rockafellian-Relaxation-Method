## Rockafellian Relaxation Method (RRM)

See `scripts/mnist_rrm.sh` for an example of how to execute `train.py` for on-prem HPC. The program can be executed 
stand-alone as well. 

A decription of program arguments can be found in the argument parsing section under `"__main__"` in `train.py`. 

Neural network and pyomo model instantiation can be found in `model.py`. 

Email `eric.eckstrand@nps.edu` for any questions or comments.

## Python Package Dependencies
`tensorflow numpy matplotlib pandas pyomo scikit-learn`

## Cite
```
@misc{royset2023rockafellian,
      title={Rockafellian Relaxation and Stochastic Optimization under Perturbations}, 
      author={Johannes O. Royset and Louis L. Chen and Eric Eckstrand},
      year={2023},
      eprint={2204.04762},
      archivePrefix={arXiv},
      primaryClass={math.OC}
}
```
