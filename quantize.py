import numpy as np
import faiss

d = 32  # data dimension

# train set 
nt = 10000
xt = np.random.rand(nt, d).astype('float32')

# dataset to encode (could be same as train)
n = 20000
x = np.random.rand(n, d).astype('float32')

# QT_8bit allocates 8 bits per dimension (QT_4bit also works)
sq = faiss.ScalarQuantizer(d, faiss.ScalarQuantizer.QT_8bit)
sq.train(xt)

# encode 
codes = sq.compute_codes(x)

# decode
x2 = sq.decode(codes)

# compute reconstruction error
avg_relative_error = ((x - x2)**2).sum() / (x ** 2).sum()

breakpoint()