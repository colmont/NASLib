import numpy as np
from scipy.stats import norm
import sys
import logging


logger = logging.getLogger(__name__)


def acquisition_function(
    ensemble, ytrain, gp, acq_fn_type="its", explore_factor=0.5, ei_calibration_factor=5.0
):
    """
    input:  trained ensemble
            ytrain (because some acquisition functions
            need to know the best arch found so far)
            acq_fn_type

    output: a method which takes in an encoded architecture and
            outputs the acquisition function value
    """

    if acq_fn_type == "its":
        # Independent Thompson sampling (ITS) acquisition function

        def its(arch_encoding, info=None):
            if gp == True:
                predictions = ensemble.query(xtest=arch_encoding, gp=gp, info=info)
                mean, cov = predictions[0]
                std = np.sqrt(np.diag(cov))
                sample = np.random.normal(mean, std)
                return sample
            else:
                predictions = ensemble.query(xtest=[arch_encoding], gp=gp, info=info)
                predictions = np.squeeze(predictions)
                mean = np.mean(predictions)
                std = np.std(predictions)
                sample = np.random.normal(mean, std)
                return sample

        return its

    elif acq_fn_type == "ucb":
        # Upper confidence bound (UCB) acquisition function

        def ucb(arch_encoding, info=None):
            if gp == True:
                predictions = ensemble.query(xtest=arch_encoding, gp=gp, info=info)
                mean, cov = predictions[0]
                std = np.sqrt(np.diag(cov))
                return mean + explore_factor * std
            else:
                predictions = ensemble.query(xtest=[arch_encoding], gp=gp, info=info)
                predictions = np.squeeze(predictions)
                mean = np.mean(predictions)
                std = np.std(predictions)
                return mean + explore_factor * std

        return ucb

    elif acq_fn_type == "ei":
        # Expected improvement (EI) acquisition function

        def ei(arch_encoding, info=None):
            if gp == True:
                predictions = ensemble.query(xtest=arch_encoding, gp=gp, info=info)
                mean, cov = predictions[0]
                std = np.sqrt(np.diag(cov))
                factored_std = std / ei_calibration_factor
                max_y = np.array(ytrain).max()
                gam = (mean - max_y) / factored_std
                ei_value = factored_std * (gam * norm.cdf(gam) + norm.pdf(gam))
                return ei_value
            else:
                predictions = ensemble.query(xtest=[arch_encoding], gp=gp, info=info)
                predictions = np.squeeze(predictions)
                mean = np.mean(predictions)
                std = np.std(predictions)
                factored_std = std / ei_calibration_factor
                max_y = np.array(ytrain).max()
                gam = (mean - max_y) / factored_std
                ei_value = factored_std * (gam * norm.cdf(gam) + norm.pdf(gam))
                return ei_value

        return ei

    elif acq_fn_type == "exploit_only":
        # Expected improvement (EI) acquisition function

        def exploit(arch_encoding, info=None):
            if gp == True:
                predictions = ensemble.query(xtest=arch_encoding, gp=gp, info=info)
                mean, cov = predictions[0]
                return mean
            else:
                predictions = ensemble.query(xtest=[arch_encoding], gp=gp, info=info)
                predictions = np.squeeze(predictions)
                mean = np.mean(predictions)
                return mean

        return exploit

    else:
        logger.info("{} is not a valid exploration type".format(acq_fn_type))
        raise NotImplementedError()
