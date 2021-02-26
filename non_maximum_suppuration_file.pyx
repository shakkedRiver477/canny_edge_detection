import numpy as np
cimport numpy as np
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t
cpdef np.ndarray non_maximum_suppuration(np.ndarray[DTYPE_t,ndim = 2] grad, np.ndarray[DTYPE_t,ndim = 2]  angels):
    """

    :param grad:
    :param angels:
    :return:
    """
    cdef int height = grad.shape[0]
    cdef int width = grad.shape[1]


    result = grad.copy()
    angels = angels * 8 / np.pi

    cdef int i
    cdef int j

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if 3 < abs(angels[i, j]) <= 4:
                if not (grad[i, j] > grad[i + 1, j] and (grad[i, j] > grad[i - 1, j])):
                    result[i, j] = 0
            elif 1 < angels[i, j] <= 3:
                if not (grad[i, j] > grad[i - 1, j + 1] and (grad[i, j] > grad[i + 1, j - 1])):
                    result[i, j] = 0
            elif -1 < angels[i, j] <= 1:
                if not (grad[i, j] > grad[i, j + 1] and (grad[i, j] > grad[i, j - 1])):
                    result[i, j] = 0
            else:
                if not (grad[i, j] > grad[i + 1, j + 1] and (grad[i, j] > grad[i - 1, j - 1])):
                    result[i, j] = 0

    return result