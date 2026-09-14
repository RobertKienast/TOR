# Topology-Aware Optimization with Graph Neural Networks

This project investigates whether a graph neural network (GNN) can act as a
learned optimizer that uses the topology of the model it is optimizing.
Conventional optimizers such as Adam update parameters coordinate by
coordinate. Here, the optimizee is represented as a graph so that information
can pass between related parameters before an update is applied.

The work evaluates graph-based learned optimizers on the Learning to Optimize
(L2O) benchmark, including LASSO, Rastrigin, and neural-network training tasks.
It also studies transfer across datasets and architectures.


## Motivation

A coordinate-wise optimizer sees each parameter's gradient but does not
explicitly model how an update propagates through the surrounding network. A
GNN provides a natural way to include those relationships:

- model parameters and structural elements become graph features;
- message passing shares information between connected components;
- edge features can include gradients, momentum, and related optimizer state;
- shared GNN weights allow the optimizer to operate on different graph sizes;
- recurrent variants can retain information across optimization steps.



## Model

The proposed optimizer uses a graph attention network (GAT):

1. A four-head attention layer processes the optimizee graph.
2. GELU activation and normalization are applied.
3. A second graph-convolution stage uses one attention head.
4. The two stages are combined through a residual connection.
5. A learned update head produces parameter updates.

The graph representation is expressive enough to reproduce coordinate-wise
gradient methods when only local edge features are used, while message passing
allows updates to depend on a wider neighborhood.

## Variants

The study compares several extensions to the base GNN optimizer.

### Sparse graphs

Graph sparsification reduces message-passing cost and may limit oversmoothing.
Three strategies are considered:

- **Random pruning:** retain edges according to a fixed sampling probability.
- **Mutual-information pruning:** retain interactions between parameters with
  strongly related trajectories.
- **Top-k cosine similarity:** give each node a bounded neighborhood containing
  its most similar peers.

### Recurrent update heads

The base update head is also replaced with recurrent models that track temporal
optimization state:

- **LSTM**, for longer-range dependencies;
- **GRU/RNN**, for lower recurrent overhead.

## Evaluation

The evaluation follows the methodology of *Learning to Optimize: A Primer and
a Benchmark*. Learned optimizers are compared with classical baselines such as
Adam, RMSProp, and FISTA, as well as the DeepMind LSTM optimizer where
applicable.

Generate plots from saved benchmark JSON results with:

```bash
python plot_all_task_results.py --input-roots <results-directory> --output-dir plots
```

The plotter supports loss curves and available classification accuracy,
macro-recall, and macro-F1 metrics.

| Problem family | Setting | Primary metric |
| --- | --- | --- |
| LASSO | $(m,n)=(5,10)$ and $(25,50)$; $\lambda=0.005$ | Relative objective loss |
| Rastrigin | Non-convex problems with $n=2$ and $n=10$ | Objective loss |
| Neural networks | MNIST, Fashion-MNIST, CIFAR, SVHN, and Covertype | Accuracy, recall, and macro-F1 |
| Out-of-distribution transfer | Changed datasets, activations, widths, and model families | Final classification metrics |

For LASSO, the reported relative-loss metric is

$$
R_{f,Q}(x)=
\frac{\mathbb{E}_{q\sim Q}[f_q(x)-f_q^*]}
     {\mathbb{E}_{q\sim Q}[f_q^*]}.
$$

The reference optimum is estimated using 2,000 FISTA iterations. Evaluated
optimizers use a horizon of 1,000 steps.

## Results

### LASSO

The GNN optimizers closely follow FISTA during the initial descent and finish
near the same raw objective on the small LASSO problem. FISTA remains strongest
around the non-differentiable $L_1$ optimum, where its soft-thresholding update
has a task-specific advantage.

![Relative loss on the small LASSO problem](lasso_testm5n10_rel.png)

### Rastrigin

The learned graph optimizers descend rapidly on both tested dimensions. Adam
catches up within 1,000 steps on the smaller problem, but not on the larger
problem reported in the paper.

![Rastrigin results for n=10](rastrigin_test_large_20260515_023030.png)

### Classification and transfer

Selected cases in which a learned optimizer matches or exceeds the strongest
reported classical baseline are shown below.

| Training / evaluation task | Learned optimizer | Macro-F1 | Best baseline | Baseline macro-F1 | Gain |
| --- | --- | ---: | --- | ---: | ---: |
| Covertype $\rightarrow$ Covertype | GNN500 $\rightarrow$ Adam | **0.802** | Cold start $\rightarrow$ Adam | 0.767 | +0.036 |
| Noisy MNIST $\rightarrow$ Noisy MNIST | GNN-LSTM500 $\rightarrow$ Adam | **0.649** | Cold start $\rightarrow$ Adam | 0.630 | +0.019 |
| Color-MNIST $\rightarrow$ SVHN Tiny CNN | GNN500 $\rightarrow$ Adam | **0.867** | Cold start $\rightarrow$ Adam | 0.850 | +0.017 |
| Permuted MNIST $\rightarrow$ MNIST Medium CNN | GNN500 $\rightarrow$ Adam | **0.926** | Cold start $\rightarrow$ Adam | 0.913 | +0.013 |
| Fashion-MNIST $\rightarrow$ MNIST | GNN500 $\rightarrow$ Adam | **0.950** | Cold start $\rightarrow$ Adam | 0.947 | +0.003 |

These results suggest that topology-aware initialization can help in selected
in-distribution and transfer settings. The full paper also contains cases where
classical baselines remain better, especially on some CIFAR and
out-of-distribution tasks.


## Limitations

- Message passing adds substantial compute and memory overhead relative to
  coordinate-wise optimizers.
- Dense parameter graphs can suffer from oversmoothing and poor scaling.
- The benchmark focuses mainly on small optimizees and relatively short
  horizons, so the results do not establish performance at production scale.
- Generalization is mixed: learned initialization helps on several tasks but is
  not consistently better than Adam across all architecture and dataset shifts.
- The current repository snapshot does not include executable experiment code
  or trained checkpoints.