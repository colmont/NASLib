# Neural Architecture Search with Projected Heat Kernel

## Description
This repository contains the code for the paper "Projected heat kernel: a novel approach for using GPs in NAS". In this paper, we introduce a unique graph kernel, namely the projected heat kernel, which has shown promising results in molecular property prediction tasks, and we explore its application within the neural architecture search (NAS) context.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Contact](#contact)

## Installation
To install the required dependencies, please refer to the [original README.md file](./README_original.md). Our fork only needs one additional dependency that can be installed as follows: `pip install simplejson`.

## Usage
To run the GP regression, execute the following command:
```bash
python naslib/runners/predictors/runner.py --config-file naslib/runners/predictors/predictor_config.yaml
```
To modify the hyperparameters of the GP, please refer to line 71 of [runner.py](naslib/runners/predictors/runner.py). The following hyperparameters can be modified:
```python
"gp_heat": GPHeatPredictor(
    ss_type=config.search_space,
    optimize_gp_hyper=True,
    projected=True,
    num_steps=500,
    sigma=3.0,
    kappa=0.05,
    n_approx=10,
)
```
Where `num_steps` is the number of iterations of the GP optimization, `sigma` and `kappa` are hyperparameters of heat kernel, and `n_approx` is the number of permutations used to approximate the projected kernel. To run the GP without the projected kernel, set `projected=False`. Similarly, to run the GP without optimizing the hyperparameters, set `optimize_gp_hyper=False`.

To customize the specifics of the regression experiment, such as number of training and test points, please refer to the [config file](naslib/runners/predictors/predictor_config.yaml).

## License
The project is listed under the Apache license. Please see [LICENSE](./LICENSE) for more details.

## Contact
If you have any questions about our code, or want to report a bug, please raise a GitHub issue.
