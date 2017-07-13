# cython: profile=True
# cython: linetrace=True
cimport cython
cimport numpy as np
import numpy as np
import scipy.sparse as sps

@cython.boundscheck(False)
def FunkSVD_sgd(R, num_factors=50, lrate=0.01, reg=0.015, iters=10, init_mean=0.0, init_std=0.1, lrate_decay=1.0, rnd_seed=42):
    if not isinstance(R, sps.csr_matrix):
        raise ValueError('R must be an instance of scipy.sparse.csr_matrix')

    # use Cython MemoryViews for fast access to the sparse structure of R
    cdef int [:] col_indices = R.indices, indptr = R.indptr
    cdef float [:] data = R.data
    cdef int M = R.shape[0], N = R.shape[1]
    cdef int nnz = len(R.data)

    # in csr format, indices correspond to column indices
    # let's build the vector of row_indices
    cdef np.ndarray[np.int64_t, ndim=1] row_nnz = np.diff(indptr).astype(np.int64)
    cdef np.ndarray[np.int64_t, ndim=1] row_indices = np.repeat(np.arange(M), row_nnz).astype(np.int64)

    # set the seed of the random number generator
    np.random.seed(rnd_seed)

    # randomly initialize the user and item latent factors
    cdef np.ndarray[np.float32_t, ndim=2] U = np.random.normal(init_mean, init_std, (M, num_factors)).astype(np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] V = np.random.normal(init_mean, init_std, (N, num_factors)).astype(np.float32)
    
    # build random index to iterate over the non-zero elements in R
    cdef np.ndarray[np.int64_t, ndim=1] shuffled_idx = np.random.permutation(nnz).astype(np.int64)
    
    # here we define some auxiliary variables
    cdef int i, j, idx, it, n
    cdef float rij, rij_pred, err, loss
    cdef np.ndarray[np.float32_t, ndim=1] U_i = np.zeros(num_factors, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] V_j = np.zeros(num_factors, dtype=np.float32)

    #
    # Stochastic Gradient Descent starts here
    #
    for it in range(iters):     # for each iteration
        loss = 0.0
        for n in range(nnz):    # iterate over non-zero values in R only
            idx = shuffled_idx[n]
            rij = data[idx]
            # get the row and col indices of x_ij 
            i = row_indices[idx]
            j = col_indices[idx]
            U_i = U[i].copy()
            V_j = V[j].copy()

            # compute the predicted value of R
            rij_pred = np.dot(U_i, V_j)

            # compute the prediction error
            err = rij - rij_pred

            # update the loss
            loss += err**2

            # adjust the latent factors
            U[i] += lrate * (err * V_j - reg * U_i)
            V[j] += lrate * (err * U_i - reg * V_j)

        loss /= nnz
        print('Iter {} - loss: {:.4f}'.format(it+1, loss))
        # update the learning rate
        lrate *= lrate_decay

    return U, V

@cython.boundscheck(False)
def AsySVD_sgd(R, num_factors=50, lrate=0.01, reg=0.015, iters=10, init_mean=0.0, init_std=0.1, lrate_decay=1.0, rnd_seed=42):
    if not isinstance(R, sps.csr_matrix):
        raise ValueError('R must be an instance of scipy.sparse.csr_matrix')

    # use Cython MemoryViews for fast access to the sparse structure of R
    cdef int [:] col_indices = R.indices, indptr = R.indptr
    cdef float [:] data = R.data
    cdef int M = R.shape[0], N = R.shape[1]
    cdef int nnz = len(R.data)

    # in csr format, indices correspond to column indices
    # let's build the vector of row_indices
    cdef np.ndarray[np.int64_t, ndim=1] row_nnz = np.diff(indptr).astype(np.int64)
    cdef np.ndarray[np.int64_t, ndim=1] row_indices = np.repeat(np.arange(M), row_nnz).astype(np.int64)

    # set the seed of the random number generator
    np.random.seed(rnd_seed)

    # randomly initialize the item latent factors
    cdef np.ndarray[np.float32_t, ndim=2] X = np.random.normal(init_mean, init_std, (N, num_factors)).astype(np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] Y = np.random.normal(init_mean, init_std, (N, num_factors)).astype(np.float32)
    
    # build random index to iterate over the non-zero elements in R
    cdef np.ndarray[np.int64_t, ndim=1] shuffled_idx = np.random.permutation(nnz).astype(np.int64)
    
    # here we define some auxiliary variables
    cdef int i, j, it, n, idx, n_rated, start, end
    cdef float rij, rij_pred, err, loss
    cdef np.ndarray[np.float32_t, ndim=1] X_j = np.zeros(num_factors, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] Y_acc = np.zeros(num_factors, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] Y_copy = np.zeros_like(Y, dtype=np.float32)

    #
    # Stochastic Gradient Descent starts here
    #
    for it in range(iters):     # for each iteration
        loss = 0.0
        for n in range(nnz):    # iterate over non-zero values in R only
            idx = shuffled_idx[n]
            rij = data[idx]
            # get the row and col indices of x_ij 
            i = row_indices[idx]
            j = col_indices[idx]
            # get the latent factor of item j
            X_j = X[j].copy()
            # accumulate the item latent factors over the other items rated by i
            Y_acc = np.zeros(num_factors, dtype=np.float32)
            n_rated = 0
            start, end = indptr[i], indptr[i+1]
            for l in col_indices[start:end]:
                x_il = data[start + n_rated]
                Y_acc += x_il * Y[l]
                n_rated += 1
            if n_rated > 0:
                Y_acc /= np.sqrt(n_rated)
            # compute the predicted rating
            rij_pred = np.dot(X_j, Y_acc)
            # compute the prediction error
            err = rij - rij_pred
            # update the loss
            loss += err**2
            # adjust the latent factors
            X[j] += lrate * (err * Y_acc - reg * X_j)
            # copy the current item preference factors
            Y_copy = Y.copy()
            for l in col_indices[indptr[i]:indptr[i+1]]:
                Y_l = Y_copy[l]
                Y[l] += lrate * (err * X_j - reg * Y_l)

        loss /= nnz
        print('Iter {} - loss: {:.4f}'.format(it+1, loss))
        # update the learning rate
        lrate *= lrate_decay

    return X, Y

@cython.boundscheck(False)
def AsySVD_compute_user_factors(user_profile, Y):
    if not isinstance(user_profile, sps.csr_matrix):
        raise ValueError('user_profile must be an instance of scipy.sparse.csr_matrix')
    assert user_profile.shape[0] == 1, 'user_profile must be a 1-dimensional vector'

    # use Cython MemoryViews for fast access to the sparse structure of user_profile
    cdef int [:] col_indices = user_profile.indices
    cdef float [:] data = user_profile.data

    # intialize the accumulated user profile
    cdef int num_factors = Y.shape[1]
    cdef np.ndarray[np.float32_t, ndim=1] Y_acc = np.zeros(num_factors, dtype=np.float32)
    cdef int n_rated = len(col_indices)
    # aux variables
    cdef int n
    # accumulate the item vectors for the items rated by the user
    for n in range(n_rated):
        ril = data[n]
        Y_acc += ril * Y[col_indices[n]]
    if n_rated > 0:
        Y_acc /= np.sqrt(n_rated)
    return Y_acc


from libc.math cimport exp, log

@cython.boundscheck(False)
def BPRMF_sgd(R, num_factors=50, lrate=0.01, user_reg=0.015, pos_reg=0.015, neg_reg=0.0015, iters=10, 
    sample_with_replacement=True, use_resampling=False,  init_mean=0.0, init_std=0.1, lrate_decay=1.0, rnd_seed=42):
    if not isinstance(R, sps.csr_matrix):
        raise ValueError('R must be an instance of scipy.sparse.csr_matrix')

    # use Cython MemoryViews for fast access to the sparse structure of R
    cdef int [:] col_indices = R.indices, indptr = R.indptr
    cdef float [:] data = R.data
    cdef int M = R.shape[0], N = R.shape[1]
    cdef int nnz = len(R.data)

    # set the seed of the random number generator
    np.random.seed(rnd_seed)
    # randomly initialize the user and item latent factors
    cdef np.ndarray[np.float32_t, ndim=2] X = np.random.normal(init_mean, init_std, (M, num_factors)).astype(np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] Y = np.random.normal(init_mean, init_std, (N, num_factors)).astype(np.float32)
    
    # sample the training triples
    cdef np.ndarray[np.int64_t, ndim=2] sample = user_uniform_item_uniform_sampling(R, nnz, replace=sample_with_replacement, seed=rnd_seed)
    
    # here we define some auxiliary variables
    cdef int i, j, k, idx, it, n
    cdef float rij, rik, loss, deriv
    cdef np.ndarray[np.float32_t, ndim=1] X_i = np.zeros(num_factors, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] Y_j = np.zeros(num_factors, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] Y_k = np.zeros(num_factors, dtype=np.float32)

    #
    # Stochastic Gradient Descent starts here
    #
    for it in range(iters):     # for each iteration
        loss = 0.0
        print(it),
        print(iters)
        for n in range(nnz):
            i, j, k = sample[n]
            # get the user and item factors
            X_i = X[i].copy()
            Y_j = Y[j].copy()
            Y_k = Y[k].copy()
            # compute the difference of the predicted scores
            diff_yjk = Y_j - Y_k
            zijk = np.dot(X_i, diff_yjk)
            # compute the sigmoid
            sig = 1. / (1. + exp(-zijk))
            # update the loss
            loss += log(sig)

            # adjust the latent factors
            deriv = 1. - sig
            X[i] += lrate * (deriv * diff_yjk - user_reg * X_i)
            Y[j] += lrate * (deriv * X_i - pos_reg * Y_j)
            Y[k] += lrate * (-deriv * X_i - neg_reg * Y_k)

        loss /= nnz
        print('Iter {} - loss: {:.4f}'.format(it+1, loss))
        # update the learning rate
        lrate *= lrate_decay
        if use_resampling:
            sample = user_uniform_item_uniform_sampling(R, nnz, replace=sample_with_replacement, seed=rnd_seed)

    return X, Y

def user_uniform_item_uniform_sampling(R, size, replace=True, seed=1234):
    # use Cython MemoryViews for fast access to the sparse structure of R
    cdef int [:] col_indices = R.indices, indptr = R.indptr
    cdef int M = R.shape[0], N = R.shape[1]
    cdef int nnz = len(R.data)

    cdef np.ndarray[np.int64_t, ndim=2] sample = np.zeros((size, 3), dtype=np.int64)
    cdef np.ndarray[np.int8_t, ndim=1] is_sampled # boolean arrays are not yet supported by Cython
    if not replace:
        is_sampled = np.zeros(nnz, dtype=np.int8)

    # set the seed of the random number generator
    np.random.seed(seed)

    cdef int i=0, start, end, iid, jid, kid, idx
    cdef np.ndarray[np.int64_t, ndim=1] aux, neg_candidates
    cdef int [:] pos_candidates
    while i < size:
        # 1) sample a user from a uniform distribution
        iid  = np.random.choice(M)

        # 2) sample a positive item uniformly at random
        start = indptr[iid]
        end = indptr[iid+1]
        pos_candidates = col_indices[start:end]
        if start == end:
            # empty candidate set
            continue
        if replace:
            # sample positive items with replacement
            jid = np.random.choice(pos_candidates)
        else:            
            # sample positive items without replacement
            # use a index vector between start and end
            aux = np.arange(start, end)
            if np.all(is_sampled[aux]):
                # all positive items have been already sampled
                continue
            idx = np.random.choice(aux)
            while is_sampled[idx]:
                # TODO: remove idx from aux to speed up the sampling
                idx = np.random.choice(aux)
            is_sampled[idx] = 1
            jid = col_indices[idx]

        # 3) sample a negative item uniformly at random
        # build the candidate set of negative items
        # TODO: precompute the negative candidate set for speed-up
        neg_candidates = np.delete(np.arange(N), pos_candidates)
        kid = np.random.choice(neg_candidates)
        sample[i, :] = [iid, jid, kid]
        i += 1
        if i % 10000 == 0:
            print('Sampling... {:.2f}% complete'.format(i/size*100))
    return sample


def ciaoni():
    print("hello worldone")