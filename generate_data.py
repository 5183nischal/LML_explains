import numpy as np
import synthetic_datasets


# valid_datasets = {
#     "regression": {
#         "gaussianLinear": synthetic_datasets.GaussianLinearRegression,
#         "gaussianNonLinearAdditive": synthetic_datasets.GaussianNonlinearAdditiveRegression,
#         "gaussianPiecewiseConstant": synthetic_datasets.GaussianPiecewiseConstantRegression,
#         "mixtureLinear": synthetic_datasets.GMLinearRegression,
#         "mixtureNonLinearAdditive": synthetic_datasets.GMNonlinearAdditiveRegression,
#         "mixturePiecewiseConstant": synthetic_datasets.GMPiecewiseConstantRegression
#     },
#     "classification": {
#         "gaussianLinear": synthetic_datasets.GaussianLinearBinary,
#         "gaussianNonLinearAdditive": synthetic_datasets.GaussianNonlinearAdditiveBinary,
#         "gaussianPiecewiseConstant": synthetic_datasets.GaussianPiecewiseConstantBinary,
#     },
# }


# This scripts generate synthetic gaussian data of dimension 'd' and covariance matrix with correlation between features of 'r'. You can specify the number of samples you'd want using 'n_sample
d = 5
r = 0.5

dataset = synthetic_datasets.GaussianNonlinearAdditiveBinary( 
    mu = np.zeros(d),
    sigma = np.identity(d),
    dim = d,
    rho = r,
    weight = np.array([4, 3, 2, 1, 0]),
    noise = 0.01)

X, y = dataset.generate(n_sample=1000)
print(X.shape)
print(y[1:10])
