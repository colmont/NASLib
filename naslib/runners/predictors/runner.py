import logging
import torch

from naslib.defaults.predictor_evaluator import PredictorEvaluator
from naslib.utils.encodings import EncodingType

from naslib.predictors import (
    BayesianLinearRegression,
    BOHAMIANN,
    BonasPredictor,
    DNGOPredictor,
    EarlyStopping,
    Ensemble,
    GCNPredictor,
    GPPredictor,
    LCEPredictor,
    LCEMPredictor,
    LGBoost,
    MLPPredictor,
    NGBoost,
    OmniNGBPredictor,
    OmniSemiNASPredictor,
    RandomForestPredictor,
    SVR_Estimator,
    SemiNASPredictor,
    SoLosspredictor,
    SparseGPPredictor,
    VarSparseGPPredictor,
    XGBoost,
    ZeroCost,
    GPWLPredictor,
    GPHeatPredictor,
)

from naslib.search_spaces.core.query_metrics import Metric
from naslib.search_spaces import (
    NasBench101SearchSpace,
    NasBench201SearchSpace,
    NasBench301SearchSpace,
    NasBenchNLPSearchSpace,
    TransBench101SearchSpaceMicro,
    TransBench101SearchSpaceMacro,
    NasBenchASRSearchSpace,
)
from naslib import utils
from naslib.utils import setup_logger, get_dataset_api

import yaml

with open("naslib/runners/predictors/gp_config.yaml", 'r') as stream:
    try:
        gp_config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

config = utils.get_config_from_args(config_type="predictor")
utils.set_seed(config.seed)
logger = setup_logger(config.save + "/log.log")
logger.setLevel(logging.INFO)
utils.log_args(config)

supported_predictors = {
    "bananas": Ensemble(predictor_type="bananas", num_ensemble=3, hpo_wrapper=True),
    "bayes_lin_reg": BayesianLinearRegression(
        encoding_type=EncodingType.ADJACENCY_ONE_HOT
    ),
    "bohamiann": BOHAMIANN(
        encoding_type=EncodingType.ADJACENCY_ONE_HOT,
        hparams_from_file=config.hparams_from_file,
    ),
    "bonas": BonasPredictor(encoding_type=EncodingType.BONAS, hpo_wrapper=True),
    "dngo": DNGOPredictor(encoding_type=EncodingType.ADJACENCY_ONE_HOT),
    "fisher": ZeroCost(method_type="fisher"),
    "flops": ZeroCost(method_type="flops"),
    "gcn": GCNPredictor(encoding_type=EncodingType.GCN, hpo_wrapper=True),
    "gp": GPPredictor(
        encoding_type=EncodingType.ADJACENCY_ONE_HOT,
        hparams_from_file=config.hparams_from_file,
    ),
    "gp_heat": GPHeatPredictor(
        ss_type=config.search_space,
        optimize_gp_hyper=gp_config['gp_heat']['optimize_gp_hyper'],
        projected=gp_config['gp_heat']['projected'],
        num_steps=gp_config['gp_heat']['num_steps'],
        sigma=gp_config['gp_heat']['sigma'],
        kappa=gp_config['gp_heat']['kappa'],
        n_approx=gp_config['gp_heat']['n_approx'],
    ),
    "gpwl": GPWLPredictor(
        ss_type=config.search_space,
        kernel_type=gp_config['gpwl']['kernel_type'],
        optimize_gp_hyper=gp_config['gpwl']['optimize_gp_hyper'],
        h=gp_config['gpwl']['h'],
        num_steps=gp_config['gpwl']['num_steps'],
    ),
    "grad_norm": ZeroCost(method_type="grad_norm"),
    "grasp": ZeroCost(method_type="grasp"),
    "jacov": ZeroCost(method_type="jacov"),
    "lce": LCEPredictor(metric=Metric.VAL_ACCURACY),
    "lce_m": LCEMPredictor(metric=Metric.VAL_ACCURACY),
    "lcsvr": SVR_Estimator(
        metric=Metric.VAL_ACCURACY, all_curve=False, require_hyper=False
    ),
    "lgb": LGBoost(encoding_type=EncodingType.ADJACENCY_ONE_HOT, hpo_wrapper=False),
    "mlp": MLPPredictor(
        encoding_type=EncodingType.ADJACENCY_ONE_HOT,
        hpo_wrapper=False,
        hparams_from_file=config.hparams_from_file,
    ),
    "nao": SemiNASPredictor(
        encoding_type=EncodingType.SEMINAS,
        semi=False,
        hpo_wrapper=False,
        hparams_from_file=config.hparams_from_file,
    ),
    "ngb": NGBoost(encoding_type=EncodingType.ADJACENCY_ONE_HOT, hpo_wrapper=False),
    "params": ZeroCost(method_type="params"),
    "rf": RandomForestPredictor(
        encoding_type=EncodingType.ADJACENCY_ONE_HOT,
        hpo_wrapper=False,
        hparams_from_file=config.hparams_from_file,
    ),
    "seminas": SemiNASPredictor(
        encoding_type="seminas",
        semi=True,
        hpo_wrapper=False,
        hparams_from_file=config.hparams_from_file,
    ),
    "snip": ZeroCost(method_type="snip"),
    "sotl": SoLosspredictor(metric=Metric.TRAIN_LOSS, sum_option="SoTL"),
    "sotle": SoLosspredictor(metric=Metric.TRAIN_LOSS, sum_option="SoTLE"),
    "sotlema": SoLosspredictor(metric=Metric.TRAIN_LOSS, sum_option="SoTLEMA"),
    "sparse_gp": SparseGPPredictor(
        encoding_type=EncodingType.ADJACENCY_ONE_HOT,
        optimize_gp_hyper=True,
    ),
    "synflow": ZeroCost(method_type="synflow"),
    "valacc": EarlyStopping(metric=Metric.VAL_ACCURACY),
    "valloss": EarlyStopping(metric=Metric.VAL_LOSS),
    "var_sparse_gp": VarSparseGPPredictor(
        encoding_type=EncodingType.ADJACENCY_ONE_HOT,
        optimize_gp_hyper=True,
    ),
    "xgb": XGBoost(
        encoding_type=EncodingType.ADJACENCY_ONE_HOT,
        hpo_wrapper=False,
        hparams_from_file=config.hparams_from_file,
    ),
    # path encoding experiments:
    "bayes_lin_reg_path": BayesianLinearRegression(encoding_type=EncodingType.PATH),
    "bohamiann_path": BOHAMIANN(encoding_type=EncodingType.PATH),
    "dngo_path": DNGOPredictor(encoding_type=EncodingType.PATH),
    "gp_path": GPPredictor(encoding_type=EncodingType.PATH),
    "lgb_path": LGBoost(encoding_type=EncodingType.PATH, hpo_wrapper=False),
    "ngb_path": NGBoost(encoding_type=EncodingType.PATH, hpo_wrapper=False),
    # omni:
    "omni_ngb": OmniNGBPredictor(
        encoding_type=EncodingType.ADJACENCY_ONE_HOT,
        config=config,
        zero_cost=["jacov"],
        lce=["sotle"],
    ),
    "omni_seminas": OmniSemiNASPredictor(
        encoding_type=EncodingType.SEMINAS,
        config=config,
        semi=True,
        hpo_wrapper=False,
        zero_cost=["jacov"],
        lce=["sotle"],
        jacov_onehot=True,
    ),
    # omni ablation studies:
    "omni_ngb_no_lce": OmniNGBPredictor(
        encoding_type=EncodingType.ADJACENCY_ONE_HOT,
        config=config,
        zero_cost=["jacov"],
        lce=[],
    ),
    "omni_seminas_no_lce": OmniSemiNASPredictor(
        encoding_type=EncodingType.SEMINAS,
        config=config,
        semi=True,
        hpo_wrapper=False,
        zero_cost=["jacov"],
        lce=[],
        jacov_onehot=True,
    ),
    "omni_ngb_no_zerocost": OmniNGBPredictor(
        encoding_type=EncodingType.ADJACENCY_ONE_HOT,
        config=config,
        zero_cost=[],
        lce=["sotle"],
    ),
    "omni_ngb_no_encoding": OmniNGBPredictor(
        encoding_type=None, config=config, zero_cost=["jacov"], lce=["sotle"]
    ),
    "nwot": ZeroCost(method_type="nwot"),
    "epe_nas": ZeroCost(method_type="epe_nas"),
    "zen": ZeroCost(method_type="zen"),
}

supported_search_spaces = {
    "nasbench101": NasBench101SearchSpace(),
    "nasbench201": NasBench201SearchSpace(),
    "nasbench301": NasBench301SearchSpace(),
    "nlp": NasBenchNLPSearchSpace(),
    "transbench101_micro": TransBench101SearchSpaceMicro(config.dataset),
    "transbench101_macro": TransBench101SearchSpaceMacro(),
    "asr": NasBenchASRSearchSpace(),
}

# Check whether code is runnin on GPU
if torch.cuda.is_available():
    print(f"Computations are running on GPU ({torch.cuda.get_device_name(torch.cuda.current_device())})")
else:
    print("Computations are running on CPU")

"""
If the API did not evaluate *all* architectures in the search space, 
set load_labeled=True
"""
load_labeled = True if config.search_space in ["nasbench301", "nlp"] else False
dataset_api = get_dataset_api(config.search_space, config.dataset)

# initialize the search space and predictor
utils.set_seed(config.seed)
predictor = supported_predictors[config.predictor]
search_space = supported_search_spaces[config.search_space]

# initialize the PredictorEvaluator class
predictor_evaluator = PredictorEvaluator(predictor, config=config)
predictor_evaluator.adapt_search_space(
    search_space, load_labeled=load_labeled, dataset_api=dataset_api
)

# evaluate the predictor
predictor_evaluator.evaluate()
