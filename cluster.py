# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import faiss

# Assume tensor_data is your input PyTorch tensor (each row is a vector)
data = torch.rand(4096, 1)

d = 1  # Dimensionality of the vectors
k = 16  # Number of clusters

# Perform k-means clustering using FAISS
kmeans = faiss.Kmeans(d=d, k=k, niter=20)
kmeans.train(x=data)
D, I = kmeans.assign(x=data)
centroids = kmeans.centroids

# D is the distances from each vector to its assigned cluster center
# I is the index of the assigned cluster for each vector

print(D)
print(I)
print(centroids)
