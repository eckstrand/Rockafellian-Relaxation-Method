## Rockafellian Relaxation Method (RRM)

See `scripts/mnist_rrm.sh` for an example of how to execute `train.py` for slurm HPC. The program can be executed stand-alone as well.

A decription of program arguments can be found in the argument parsing section under `"__main__"` in `train.py`.

Neural network and pyomo models can be found in `model.py`.

Email `eric.eckstrand@nps.edu` for any questions or comments.

## Python Package Dependencies
`tensorflow numpy matplotlib pandas pyomo scikit-learn`

## Example
In order to reproduce the MNIST experiments of section 6.2 of [Rockafellian Relaxation and Stochastic Optimization under Perturbations](https://arxiv.org/pdf/2204.04762.pdf),
run as follows, adjusting the level of label contamination (`--swap_pct`) accordingly. `--theta`, `--mu`, and `--lr` can be tuned for optimal results.

```
python ./train.py \
--results_dir /data/eric.eckstrand/out/rrm/mnist_rrm \
--solver cplex \
--solver_exe /share/apps/cplex/12.8.0/cplex/bin/x86-64_linux/cplex \
--theta 0.4 \
--u_reg 'l1' \
--n_iterations 50 \
--epochs 10 \
--batch 100 \
--nn_opt 'sgd' \
--use_model 'fc_royset_norton' \
--swap_pct 0.65 \
--lr 0.1 \
--mu 0.5 \
--small_mnist \
--u_opt \
```

Replace the `--solver` and `--solver_exe` arguments with the appropriate pyomo solver. 

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
