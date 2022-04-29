import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial.distance import cdist

from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import DENSE, UNSIGNED_DATA, INPUT


class AddPointsOperator(AutoSklearnPreprocessingAlgorithm):
    """
    This component performs points adding to DoE
    """

    def __init__(self, **kwargs):
        super().__init__()
        for key, val in kwargs.get('config', {}).items():
            setattr(self, key, val)
        self._criterions_creators = {
            'max_min': self._max_min_creator,
            # 'gp_var': self._gp_creator,
        }

        self._criterion_creator = None

    def _max_min_creator(self, doe):
        def criterion(x):
            return -np.min(cdist(x.reshape(1, -1), doe))

        return criterion

    # def _gp_creator(self, doe):
    #     # TODO: this should be set by user in future too
    #     kernel_func = getattr(gp_sigma, self._config['kernel'])(**self._config['kernel_params'])
    #     noize_var = self._config['noize_var']
    #     cov_xx = kernel_func(doe, doe)  # self covariance for each set of sigmas
    #     cov_xx += np.diag(np.repeat(noize_var, doe.shape[0]))
    
    #     def criterion(x):
    #         return -gp_sigma.evaluate_sigma(doe, x.reshape(1, -1), kernel_func=kernel_func, noize_var=noize_var,
    #                                         cov_xx=cov_xx)

    #     return criterion

    def transform(self, X):
        for i in range(self._n_iterations):
            criterion = self._criterion_creator(X)
            bounds = np.array([X.min(axis=0), X.max(axis=0)]).T.tolist()
            res = differential_evolution(criterion, bounds=bounds, tol=1e-4, seed=0)
            X = np.append(X, res.x.reshape(1, -1), axis=0)
        return X

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'AddPointOperator',
                'name': 'This adds points to initial DoE',
                # 'handles_missing_values': False,
                # 'handles_nominal_values': True,
                # 'handles_numerical_features': True,
                # 'prefers_data_scaled': False,
                # 'prefers_data_normalized': False,
                'handles_regression': True,
                'handles_classification': False,
                'handles_multiclass': False,
                'handles_multilabel': False,
                'handles_multioutput': False,
                'is_deterministic': False,  # is it?
                # 'handles_sparse': False,
                # 'handles_dense': True,
                'input': (DENSE, UNSIGNED_DATA),
                'output': (INPUT,),
                # 'preferred_dtype': None
            }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        space = ConfigurationSpace()
        params = [
            CategoricalHyperparameter(
                'method',
                [
                    'opt',
                    # 'low_discrepancy'
                ],
                default_value='opt',
            ),
            CategoricalHyperparameter(
                'opt_criterion',
                [
                    'max_min',
                    # 'gp_var',
                ],
                default_value='max_min',
            ),
            UniformFloatHyperparameter(
                'n_points_frac',
                lower=0,
                upper=1,
            ),

        ]
        for param in params:
            space.add_hyperparameter(param)
        return space

    def fit(self, X, y):
        self._n_iterations = int(X.shape[0] * self.n_points_frac)
        self._criterion_creator = self._criterions_creators[self.opt_criterion]
        return self