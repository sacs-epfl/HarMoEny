# Cloning
`git clone --recurse-submodules git@github.com:Shamauk/moe.git`

# Running environment
```bash
./start_image.sh
```

# Running experiment
```bash
python3 start.py num_gpus port_num policy dataset_name experiment_name
```

# Plotting
```bash
cd plotting
python3 plot_experiment.py num_gpus paths_to_collected_traces
```

# Adding new scheduling policies
`vim transformers/src/transformers/models/switch_transformers/modeling_switch_transformers.py `
Under class SwitchTransformersSparseMLP create a new function and update match statement to include new policy
