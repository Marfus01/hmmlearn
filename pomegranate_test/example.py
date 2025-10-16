from pomegranate.hmm import DenseHMM
from pomegranate.distributions import Normal
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
n, l, d = 100, 25, 15  # number of sequences, length of each sequence, dimensionality of data
X = torch.randn(n, l, d)

k = 25 # number of hidden states

dists1, dists2 = [], []
for i in range(k):
    mu = torch.randn(d)
    covs = torch.exp(torch.randn(d))

    dist1 = Normal(mu, covs, covariance_type='diag')
    dist2 = Normal(mu, covs, covariance_type='diag').cuda()

    dists1.append(dist1)
    dists2.append(dist2)


model1 = DenseHMM(dists1, max_iter=3)
model2 = DenseHMM(dists2, max_iter=3).cuda()

X_cuda = X.cuda()

model1.fit(X)
model2.fit(X_cuda)
