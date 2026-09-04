# Sparsification of subject–verb agreement circuits

Code accompanying the paper. Install dependencies with:

```bash
pip install -r requirements.txt
```

## Experiments

| Paper experiment | File |
|---|---|
| Experiment 1: neuron attribution and causal frontiers | `experiments/experiment1_neurons.py` |
| Experiment 1: attention-head necessity and sufficiency | `experiments/experiment1_heads.py` |
| Experiment 2: joint neuron–head reorganization | `experiments/experiment2_routing.py` |
| Experiment 2: reorganization figure | `plots/experiment2_routing.py` |
| Experiment 2: final circuit and off-circuit interventions | `experiments/experiment2_circuit.py` |
| Experiment 2: template, cross-template, perturbation, and lexical generalization | `experiments/experiment2_generalization.py` |
| Experiment 2: one-dimensional attention-head channel | `experiments/experiment2_head_channel.py` |

Fixed datasets are in `data/`. Results are written to `results/`.

The final circuit is hard-coded from the paper; circuit discovery and pruning
are not reproduced.
