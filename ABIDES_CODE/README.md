# POW-dTS Experimentation Guide

This README provides instructions on how to experiment with the POW-dTS (Power of Weighted discounted Thompson Sampling) algorithm within the ABIDES market simulation framework. This code is intended to illustrate the methodology presented in the associated research paper.

## Setup Instructions

1. Clone the ABIDES repository: [https://github.com/jpmorganchase/abides-jpmc-public/tree/main/abides-markets/abides_markets](https://github.com/jpmorganchase/abides-jpmc-public/tree/main/abides-markets/abides_markets)
2. Incorporate this code into the ABIDES markets folder
3. Merge the following folders with existing ABIDES directories:
   - `config/`
   - `agent/`
   - `models/`
4. Place the contents of the `obj/` folder (scalers and pre-trained policy agents) in an accessible location and update references in the code accordingly

## Current Code Status

The code is currently in an unrefined phase and contains functionality beyond POW-dTS, including:
- Policy cloning capabilities
- Adversarial rewards
- Other experimental features

While these additional features are not within the scope of the paper, they have been retained for now. Future updates will include a more streamlined version focused specifically on reproducing the results presented in the paper.

## Parameters and Alternative Setups

The codebase contains numerous parameters beyond those specific to POW-dTS. These parameters correspond to various experimental setups and alternative algorithmic approaches that can be configured. Many of these parameters are not documented in detail as they are not directly related to the POW-dTS methodology presented in the paper.

Users should be aware that modifying these undocumented parameters may lead to experimental setups different from those described in the POW-dTS paper.

## Running Experiments

Use the following command to launch an experiment that reproduces the paper's methodology:

```bash
python3.11 -u abides.py -c rmsc03_experiments_POWDTS.py -t ABM -d 20200603 --qlagents=1 --numinvestors=50 --nsims=80 --exp exp_thompson --hedges=1 --multi_weight=0.9 --dql_file_1 '/obj/pretrained_policies/policy0.pkl' --dql_comp_file_1 '/obj/pretrained_policies/policy0.pkl' --avoid_retrain=True --th_alpha=1 --th_beta=1 --th_gamma=0.85 --round_exp=1
```

## POW-dTS Specific Parameters

| Parameter | Description |
|-----------|-------------|
| `--qlagents` | Number of main MM RL agents (not competitors): Always set to 1 |
| `--numinvestors` | Number of independent investors that the MM operates against |
| `--nsims` | Simulations per scenario |
| `--exp` | Output folder name |
| `--multi_weight` | Percentage (0-1) assigned to the first sub-objective, in this case MtM |
| `--dql_file_1` | Pre-trained network used by the main MM. For the paper, a policy trained only against random agents is used |
| `--dql_comp_file` | Pre-trained network of competitor agents |
| `--avoid_retrain` | When set to True, indicates that the agent only exploits the policy without retraining on new experiences |
| `--th_alpha` | The increment in alpha for the algorithm when it correctly identifies the optimal policy |
| `--th_beta` | The increment in beta for the algorithm when it fails to identify the optimal policy |
| `--th_gamma` | The discount factor for the discounted Thompson sampling |

## Important Notes

- This code is provided to illustrate the methodology described in the POW-dTS paper
- Ensure all objects in the `obj/` folder (scalers and pre-trained policies) are correctly referenced within the code
- The codebase contains many other parameters and configuration options beyond those described here, which correspond to different experimental setups not covered in the POW-dTS paper
- Future code releases will include a more focused implementation specifically targeting the POW-dTS methodology