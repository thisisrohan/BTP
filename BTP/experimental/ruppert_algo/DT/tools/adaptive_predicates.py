import numpy as np

def njit(f):
    return f
from numba import njit


@njit(cache=True)
def Fast_Two_Sum_Tail(a, b, x):
    bvirt = x - a
    y = b - bvirt
    return y


@njit(cache=True)
def Fast_Two_Sum(a, b):
    x = a + b
    y = Fast_Two_Sum_Tail(a, b, x)
    return x, y


@njit(cache=True)
def Fast_Two_Diff_Tail(a, b, x):
    bvirt = a - x
    y = bvirt - b
    return y


@njit(cache=True)
def Fast_Two_Diff(a, b):
    x = a - b
    y = Fast_Two_Diff_Tail(a, b, x)
    return x, y


@njit(cache=True)
def Two_Sum_Tail(a, b, x):
    bvirt = x - a
    avirt = x - bvirt
    bround = b - bvirt
    around = a - avirt
    y = around + bround
    return y


@njit(cache=True)
def Two_Sum(a, b):
    x = a + b
    y = Two_Sum_Tail(a, b, x)
    return x, y


@njit(cache=True)
def Two_Diff_Tail(a, b, x):
    bvirt = a - x
    avirt = x + bvirt
    bround = bvirt - b
    around = a - avirt
    y = around + bround
    return y


@njit(cache=True)
def Two_Diff(a, b):
    x = a - b
    y = Two_Diff_Tail(a, b, x)
    return x, y


@njit(cache=True)
def Split(a, splitter):
    c = splitter * a
    abig = c - a
    ahi = c - abig
    alo = a - ahi
    return ahi, alo


@njit(cache=True)
def Two_Product_Tail(a, b, x, splitter):
    ahi, alo = Split(a, splitter)
    bhi, blo = Split(b, splitter)
    err1 = x - (ahi * bhi)
    err2 = err1 - (alo * bhi)
    err3 = err2 - (ahi * blo)
    y = (alo * blo) - err3
    return y


@njit(cache=True)
def Two_Product(a, b, splitter):
    x = a * b
    y = Two_Product_Tail(a, b, x, splitter)
    return x, y


# Two_Product_Presplit() is Two_Product() where one of the inputs has already 
# been split. Avoids redundant splitting.

@njit(cache=True)
def Two_Product_Presplit(a, b, bhi, blo, splitter):
    x = a * b
    ahi, alo = Split(a, splitter)
    err1 = x - (ahi * bhi)
    err2 = err1 - (alo * bhi)
    err3 = err2 - (ahi * blo)
    y = (alo * blo) - err3
    return x, y


# Two_Product_2Presplit() is Two_Product() where both of the inputs have
# already been split. Avoids redundant splitting.

@njit(cache=True)
def Two_Product_2Presplit(a, ahi, alo, b, bhi, blo):
    x = a * b
    err1 = x - (ahi * bhi)
    err2 = err1 - (alo * bhi)
    err3 = err2 - (ahi * blo)
    y = (alo * blo) - err3
    return x, y


# Square() can be done more quickly than Two_Product().

@njit(cache=True)
def Square_Tail(a, x, splitter):
    ahi, alo = Split(a, splitter)
    err1 = x - (ahi * ahi)
    err3 = err1 - ((ahi + ahi) * alo)
    y = (alo * alo) - err3
    return y


@njit(cache=True)
def Square(a, splitter):
    x = a * a
    y = Square_Tail(a, x, splitter)
    return x, y


# Functions for summing expansions of various fixed lengths. These are all
# unrolled versions of Expansion_Sum().

@njit(cache=True)
def Two_One_Sum(a1, a0, b):
    _i, x0 = Two_Sum(a0, b)
    x2, x1 = Two_Sum(a1, _i)
    return x2, x1, x0


@njit(cache=True)
def Two_One_Diff(a1, a0, b):
    _i, x0 = Two_Diff(a0, b)
    x2, x1 = Two_Sum(a1, _i)
    return x2, x1, x0


@njit(cache=True)
def Two_Two_Sum(a1, a0, b1, b0):
    _j, _0, x0 = Two_One_Sum(a1, a0, b0)
    x3, x2, x1 = Two_One_Sum(_j, _0, b1)
    return x3, x2, x1, x0


@njit(cache=True)
def Two_Two_Diff(a1, a0, b1, b0):
    _j, _0, x0 = Two_One_Diff(a1, a0, b0)
    x3, x2, x1 = Two_One_Diff(_j, _0, b1)
    return x3, x2, x1, x0


@njit(cache=True)
def Four_One_Sum(a3, a2, a1, a0, b):
    _j, x1, x0 = Two_One_Sum(a1, a0, b)
    x4, x3, x2 = Two_One_Sum(a3, a2, _j)
    return x4, x3, x2, x1, x0


@njit(cache=True)
def Four_Two_Sum(a3, a2, a1, a0, b1, b0):
    _k, _2, _1, _0, x0 = Four_One_Sum(a3, a2, a1, a0, b0)
    x5, x4, x3, x2, x1 = Four_One_Sum(_k, _2, _1, _0, b1)
    return x5, x4, x3, x2, x1, x0


@njit(cache=True)
def Four_Four_Sum(a3, a2, a1, a0, b4, b3, b1, b0):
    _l, _2, _1, _0, x1, x0 = Four_Two_Sum(a3, a2, a1, a0, b1, b0)
    x7, x6, x5, x4, x3, x2 = Four_Two_Sum(_l, _2, _1, _0, b4, b3)
    return x7, x6, x5, x4, x3, x2, x1, x0


@njit(cache=True)
def Eight_One_Sum(a7, a6, a5, a4, a3, a2, a1, a0, b):
    _j, x3, x2, x1, x0 = Four_One_Sum(a3, a2, a1, a0, b)
    x8, x7, x6, x5, x4 = Four_One_Sum(a7, a6, a5, a4, _j)
    return x8, x7, x6, x5, x4, x3, x2, x1, x0


@njit(cache=True)
def Eight_Two_Sum(a7, a6, a5, a4, a3, a2, a1, a0, b1, b0):
    _k, _6, _5, _4, _3, _2, _1, _0, x0 = Eight_One_Sum(a7, a6, a5, a4, a3,
                                                       a2, a1, a0, b0)
    x9, x8, x7, x6, x5, x4, x3, x2, x1 = Eight_One_Sum(_k, _6, _5, _4, _3,
                                                       _2, _1, _0, b1)
    return x9, x8, x7, x6, x5, x4, x3, x2, x1, x0


@njit(cache=True)
def Eight_Four_Sum(a7, a6, a5, a4, a3, a2, a1, a0, b4, b3, b1, b0):
    _l, _6, _5, _4, _3, _2, _1, _0, x1, x0 = Eight_Two_Sum(a7, a6, a5, a4, a3,
                                                           a2, a1, a0, b1, b0)
    x11, x10, x9, x8, x7, x6, x5, x4, x3, x2 = Eight_Two_Sum(_l, _6, _5, _4,
                                                             _3, _2, _1, _0,
                                                             b4, b3)
    return x11, x10, x9, x8, x7, x6, x5, x4, x3, x2, x1, x0


# Functions for multiplying expansions of various fixed lengths.

@njit(cache=True)
def Two_One_Product(a1, a0, b, splitter):
    bhi, blo = Split(b, splitter)
    _i, x0 = Two_Product_Presplit(a0, b, bhi, blo, splitter)
    _j, _0 = Two_Product_Presplit(a1, b, bhi, blo, splitter)
    _k, x1 = Two_Sum(_i, _0)
    x3, x2 = Fast_Two_Sum(_j, _k)
    return x3, x2, x1, x0


@njit(cache=True)
def Four_One_Product(a3, a2, a1, a0, b, splitter):
    bhi, blo = Split(b, splitter)
    _i, x0 = Two_Product_Presplit(a0, b, bhi, blo, splitter)
    _j, _0 = Two_Product_Presplit(a1, b, bhi, blo, splitter)
    _k, x1 = Two_Sum(_i, _0)
    _i, x2 = Fast_Two_Sum(_j, _k)
    _j, _0 = Two_Product_Presplit(a2, b, bhi, blo, splitter)
    _k, x3 = Two_Sum(_i, _0)
    _i, x4 = Fast_Two_Sum(_j, _k)
    _j, _0 = Two_Product_Presplit(a3, b, bhi, blo, splitter)
    _k, x5 = Two_Sum(_i, _0)
    x7, x6 = Fast_Two_Sum(_j, _k)
    return x7, x6, x5, x4, x3, x2, x1, x0


@njit(cache=True)
def Two_Two_Product(a1, a0, b1, b0, splitter):
    a0hi, a0lo = Split(a0, splitter)
    bhi, blo = Split(b0, splitter)
    _i, x0 = Two_Product_2Presplit(a0, a0hi, a0lo, b0, bhi, blo)
    a1hi, a1lo = Split(a1, splitter)
    _j, _0 = Two_Product_2Presplit(a1, a1hi, a1lo, b0, bhi, blo)
    _k, _1 = Two_Sum(_i, _0)
    _l, _2 = Fast_Two_Sum(_j, _k)
    bhi, blo = Split(b1, splitter)
    _i, _0 = Two_Product_2Presplit(a0, a0hi, a0lo, b1, bhi, blo)
    _k, x1 = Two_Sum(_1, _0)
    _j, _1 = Two_Sum(_2, _k)
    _m, _2 = Two_Sum(_l, _j)
    _j, _0 = Two_Product_2Presplit(a1, a1hi, a1lo, b1, bhi, blo)
    _n, _0 = Two_Sum(_i, _0)
    _i, x2 = Two_Sum(_1, _0)
    _k, _1 = Two_Sum(_2, _i)
    _l, _2 = Two_Sum(_m, _k)
    _k, _0 = Two_Sum(_j, _n)
    _j, x3 = Two_Sum(_1, _0)
    _i, _1 = Two_Sum(_2, _j)
    _m, _2 = Two_Sum(_l, _i)
    _i, x4 = Two_Sum(_1, _k)
    _k, x5 = Two_Sum(_2, _i)
    x7, x6 = Two_Sum(_m, _k)
    return x7, x6, x5, x4, x3, x2, x1, x0


# An expansion of length two can be squared more quickly than finding the
# product of two different expansions of length two, and the result is
# guaranteed to have no more than six (rather than eight) components.

@njit(cache=True)
def Two_Square(a1, a0, splitter):
    _j, x0 = Square(a0, splitter)
    _0 = a0 + a0
    _k, _1 = Two_Product(a1, _0, splitter)
    _l, _2, x1 = Two_One_Sum(_k, _1, _j)
    _j, _1 = Square(a1, splitter)
    x5, x4, x3, x2 = Two_Two_Sum(_j, _1, _l, _2)
    return x5, x4, x3, x2, x1, x0


#*****************************************************************************#
#                                                                             #
#   exactinit()   Initialize the variables used for exact arithmetic.         #
#                                                                             #
#   `epsilon' is the largest power of two such that 1.0 + epsilon = 1.0 in    #
#   floating-point arithmetic.  `epsilon' bounds the relative roundoff        #
#   error.  It is used for floating-point error analysis.                     #
#                                                                             #
#   `splitter' is used to split floating-point numbers into two half-         #
#   length significands for exact multiplication.                             #
#                                                                             #
#   I imagine that a highly optimizing compiler might be too smart for its    #
#   own good, and somehow cause this routine to fail, if it pretends that     #
#   floating-point arithmetic is too much like real arithmetic.               #
#                                                                             #
#   Don't change this routine unless you fully understand it.                 #
#                                                                             #
#*****************************************************************************#

@njit(cache=True)
def exactinit2d(points, res_arr):

    every_other = True
    half = np.float64(0.5)
    epsilon = np.float64(1.0)
    splitter = np.float64(1.0)
    one = np.float64(1.0)
    two = np.float64(2.0)
    while True:
        epsilon *= half
        if every_other:
          splitter *= two
        every_other = not every_other
        if one + epsilon != one:
            pass
        else:
            break
    splitter += one
    res_arr[0] = splitter

    xmin = np.min(points[:, 0])
    ymin = np.min(points[:, 1])
    xmax = np.max(points[:, 0])
    ymax = np.max(points[:, 1])
    b = max(xmax-xmin, ymax-ymin)

    # Error bounds for orientation and incircle tests.
    res_arr[1] = (3.0 + 8.0 * epsilon) * epsilon  # resulterrbound
    res_arr[2] = (3.0 + 16.0 * epsilon) * epsilon  # ccwerrboundA
    res_arr[3] = (2.0 + 12.0 * epsilon) * epsilon  # ccwerrboundB
    res_arr[4] = (9.0 + 64.0 * epsilon) * epsilon * epsilon  # ccwerrboundC
    res_arr[5] = (10.0 + 96.0 * epsilon) * epsilon  # iccerrboundA
    res_arr[6] = (4.0 + 48.0 * epsilon) * epsilon  # iccerrboundB
    res_arr[7] = (44.0 + 576.0 * epsilon) * epsilon * epsilon  # iccerrboundC
    res_arr[8] = 32 * epsilon * b * b  # static_filter_o2d
    res_arr[9] = 1984 * epsilon * b * b * b  # static_filter_i2d

    return


@njit(cache=True)
def exactinit3d(points):

    every_other = True
    half = np.float64(0.5)
    epsilon = np.float64(1.0)
    splitter = np.float64(1.0)
    one = np.float64(1.0)
    two = np.float64(2.0)
    while True:
        epsilon *= half
        if every_other:
          splitter *= two
        every_other = not every_other
        if one + epsilon != one:
            pass
        else:
            break
    splitter += one

    xmin = np.min(points[:, 0])
    ymin = np.min(points[:, 1])
    zmin = np.min(points[:, 2])
    xmax = np.max(points[:, 0])
    ymax = np.max(points[:, 1])
    zmax = np.max(points[:, 2])
    b = max(xmax-xmin, ymax-ymin, zmax-zmin)

    # Error bounds for orientation and incircle tests.
    resulterrbound = (3.0 + 8.0 * epsilon) * epsilon
    o3derrboundA = (7.0 + 56.0 * epsilon) * epsilon
    o3derrboundB = (3.0 + 28.0 * epsilon) * epsilon
    o3derrboundC = (26.0 + 288.0 * epsilon) * epsilon * epsilon
    isperrboundA = (16.0 + 224.0 * epsilon) * epsilon
    isperrboundB = (5.0 + 72.0 * epsilon) * epsilon
    isperrboundC = (71.0 + 1408.0 * epsilon) * epsilon * epsilon
    static_filter_o3d = 352 * epsilon * b * b * b
    static_filter_i3d = 33024 * epsilon * b * b * b * b

    return resulterrbound, o3derrboundA, o3derrboundB, o3derrboundC, \
           isperrboundA, isperrboundB, isperrboundC, splitter, \
           static_filter_o3d, static_filter_i3d


#*****************************************************************************#
#                                                                             #
#   grow_expansion()   Add a scalar to an expansion.                          #
#                                                                             #
#   Sets h = e + b.  See the long version of Shewchuk's paper for details.    #
#                                                                             #
#   Maintains the nonoverlapping property.  If round-to-even is used (as      #
#   with IEEE 754), maintains the strongly nonoverlapping and nonadjacent     #
#   properties as well.  (That is, if e has one of these properties, so       #
#   will h.)                                                                  #
#                                                                             #
#*****************************************************************************#

@njit(cache=True)
def grow_expansion(elen, e, b, h):
    # e and h can be the same.

    Q = b
    for eindex in range(0, elen):
        enow = e[eindex]
        Q, h[eindex] = Two_Sum(Q, enow)

    h[elen] = Q

    return elen + 1


#*****************************************************************************#
#                                                                             #
#   grow_expansion_zeroelim()   Add a scalar to an expansion, eliminating     #
#                               zero components from the output expansion.    #
#                                                                             #
#   Sets h = e + b.  See the long version of Shewchuk's paper for details.    #
#                                                                             #
#   Maintains the nonoverlapping property.  If round-to-even is used (as      #
#   with IEEE 754), maintains the strongly nonoverlapping and nonadjacent     #
#   properties as well.  (That is, if e has one of these properties, so       #
#   will h.)                                                                  #
#                                                                             #
#*****************************************************************************#

@njit(cache=True)
def grow_expansion_zeroelim(elen, e, b, h):
    # e and h can be the same.

    hindex = int(0)
    Q = b
    for eindex in range(0, elen):
        enow = e[eindex]
        Q, hh = Two_Sum(Q, enow)
        if hh != 0.0:
            h[hindex] = hh
            hindex += 1

    if Q != 0.0 or hindex == 0:
        h[hindex] = Q
        hindex += 1
    
    return hindex


#*****************************************************************************#
#                                                                             #
#   expansion_sum()   Sum two expansions.                                     #
#                                                                             #
#   Sets h = e + f.  See the long version of Shewchuk's paper for details.    #
#                                                                             #
#   Maintains the nonoverlapping property.  If round-to-even is used (as      #
#   with IEEE 754), maintains the nonadjacent property as well.  (That is,    #
#   if e has one of these properties, so will h.)  Does NOT maintain the      #
#   strongly nonoverlapping property.                                         #
#                                                                             #
#*****************************************************************************#

@njit(cache=True)
def expansion_sum(elen, e, flen, f, h):
    # e and h can be the same, but f and h cannot.

    Q = f[0]
    for hindex in range(0, elen):
        hnow = e[hindex]
        Q, h[hindex] = Two_Sum(Q, hnow)

    hindex = elen
    h[hindex] = Q
    hlast = hindex
    for findex in range(1, flen):
        Q = f[findex]
        for hindex in range(findex, hlast+1):
            hnow = h[hindex]
            Q, h[hindex] = Two_Sum(Q, hnow)
        hlast += 1
        h[hlast] = Q

    return hlast + 1


#*****************************************************************************#
#                                                                             #
#   expansion_sum_zeroelim1()   Sum two expansions, eliminating zero          #
#                               components from the output expansion.         #
#                                                                             #
#   Sets h = e + f.  See the long version of Shewchuk's paper for details.    #
#                                                                             #
#   Maintains the nonoverlapping property.  If round-to-even is used (as      #
#   with IEEE 754), maintains the nonadjacent property as well.  (That is,    #
#   if e has one of these properties, so will h.)  Does NOT maintain the      #
#   strongly nonoverlapping property.                                         #
#                                                                             #
#*****************************************************************************#

@njit(cache=True)
def expansion_sum_zeroelim1(elen, e, flen, f, h):
    # e and h can be the same, but f and h cannot.

    Q = f[0]
    for hindex in range(0, elen):
        hnow = e[hindex]
        Q, h[hindex] = Two_Sum(Q, hnow)

    hindex = elen
    h[hindex] = Q
    hlast = hindex
    for findex in range(1, flen):
        Q = f[findex]
        for hindex in range(findex, hlast+1):
            hnow = h[hindex]
            Q, h[hindex] = Two_Sum(Q, hnow)
        hlast += 1
        h[hlast] = Q

    hindex = -1
    for index in range(0, hlast+1):
        hnow = h[index]
        if hnow != 0.0:
            hindex += 1
            h[hindex] = hnow

    if hindex == -1:
        return 1
    else:
        return hindex + 1


#*****************************************************************************#
#                                                                             #
#   expansion_sum_zeroelim2()   Sum two expansions, eliminating zero          #
#                               components from the output expansion.         #
#                                                                             #
#   Sets h = e + f.  See the long version of Shewchuk's paper for details.    #
#                                                                             #
#   Maintains the nonoverlapping property.  If round-to-even is used (as      #
#   with IEEE 754), maintains the nonadjacent property as well.  (That is,    #
#   if e has one of these properties, so will h.)  Does NOT maintain the      #
#   strongly nonoverlapping property.                                         #
#                                                                             #
#*****************************************************************************#

@njit(cache=True)
def expansion_sum_zeroelim2(elen, e, flen, f, h):
    # e and h can be the same, but f and h cannot.

    hindex = 0
    Q = f[0]
    for eindex in range(0, elen):
        enow = e[eindex]
        Q, hh = Two_Sum(Q, enow)
        if hh != 0.0:
            h[hindex] = hh
            hindex += 1

    h[hindex] = Q
    hlast = hindex
    for findex in range(1, flen):
        hindex = 0
        Q = f[findex]
        for eindex in range(0, hlast+1):
            enow = h[eindex]
            Q, hh = Two_Sum(Q, enow)
            if hh != 0:
                h[hindex] = hh
                hindex += 1

        h[hindex] = Q
        hlast = hindex

    return hlast + 1


#*****************************************************************************#
#                                                                             #
#   fast_expansion_sum()   Sum two expansions.                                #
#                                                                             #
#   Sets h = e + f.  See the long version of Shewchuk's paper for details.    #
#                                                                             #
#   If round-to-even is used (as with IEEE 754), maintains the strongly       #
#   nonoverlapping property.  (That is, if e is strongly nonoverlapping, h    #
#   will be also.)  Does NOT maintain the nonoverlapping or nonadjacent       #
#   properties.                                                               #
#                                                                             #
#*****************************************************************************#

@njit(cache=True)
def fast_expansion_sum(elen, e, flen, f, h):
    # h cannot be e or f.

    enow = e[0]
    fnow = f[0]
    eindex = 0
    findex = 0
    if (fnow > enow) == (fnow > -enow):
        Q = enow
        eindex += 1
        enow = e[eindex]
    else:
        Q = fnow
        findex += 1
        fnow = f[findex]

    hindex = 0
    if (eindex < elen) and (findex < flen):
        if (fnow > enow) == (fnow > -enow):
            Q, h[0] = Fast_Two_Sum(enow, Q)
            eindex += 1
            enow = e[eindex]
        else:
            Q, h[0] = Fast_Two_Sum(fnow, Q)
            findex += 1
            fnow = f[findex]

        hindex = 1
        while eindex < elen and findex < flen:
            if (fnow > enow) == (fnow > -enow):
                Q, h[hindex] = Two_Sum(Q, enow)
                eindex += 1
                enow = e[eindex]
            else:
                Q, h[hindex] = Two_Sum(Q, fnow)
                findex += 1
                fnow = f[findex]
            hindex += 1

    while eindex < elen:
        Q, h[hindex] = Two_Sum(Q, enow)
        eindex += 1
        enow = e[eindex]
        hindex += 1

    while findex < flen:
        Q, h[hindex] = Two_Sum(Q, fnow)
        findex += 1
        fnow = f[findex]
        hindex += 1

    h[hindex] = Q

    return hindex + 1


#*****************************************************************************#
#                                                                             #
#   fast_expansion_sum_zeroelim()   Sum two expansions, eliminating zero      #
#                                   components from the output expansion.     #
#                                                                             #
#   Sets h = e + f.  See the long version of Shewchuk's paper for details.    #
#                                                                             #
#   If round-to-even is used (as with IEEE 754), maintains the strongly       #
#   nonoverlapping property.  (That is, if e is strongly nonoverlapping, h    #
#   will be also.)  Does NOT maintain the nonoverlapping or nonadjacent       #
#   properties.                                                               #
#                                                                             #
#*****************************************************************************#

@njit(cache=True)
def fast_expansion_sum_zeroelim(elen, e, flen, f, h):
    # h cannot be e or f.

    enow = e[0]
    fnow = f[0]
    eindex = 0
    findex = 0
    if (fnow > enow) == (fnow > -enow):
        Q = enow
        eindex += 1
        enow = e[eindex]
    else:
        Q = fnow
        findex += 1
        fnow = f[findex]

    hindex = 0
    if (eindex < elen) and (findex < flen):
        if (fnow > enow) == (fnow > -enow):
            Q, hh = Fast_Two_Sum(enow, Q)
            eindex += 1
            enow = e[eindex]
        else:
            Q, hh = Fast_Two_Sum(fnow, Q)
            findex += 1
            fnow = f[findex]

        if hh != 0.0:
            h[hindex] = hh
            hindex += 1

        while (eindex < elen) and (findex < flen):
            if (fnow > enow) == (fnow > -enow):
                Q, hh = Two_Sum(Q, enow)
                eindex += 1
                enow = e[eindex]
            else:
                Q, hh = Two_Sum(Q, fnow)
                findex += 1
                fnow = f[findex]

            if (hh != 0.0):
                h[hindex] = hh
                hindex += 1

    while eindex < elen:
        Q, hh = Two_Sum(Q, enow)
        eindex += 1
        enow = e[eindex]
        if hh != 0.0:
            h[hindex] = hh
            hindex += 1

    while findex < flen:
        Q, hh = Two_Sum(Q, fnow)
        findex += 1
        fnow = f[findex]
        if hh != 0.0:
            h[hindex] = hh
            hindex += 1

    if (Q != 0.0) or (hindex == 0):
        h[hindex] = Q
        hindex += 1

    return hindex


#*****************************************************************************#
#                                                                             #
#   linear_expansion_sum()   Sum two expansions.                              #
#                                                                             #
#   Sets h = e + f.  See either version of Shewchuk's paper for details.      #
#                                                                             #
#   Maintains the nonoverlapping property.  (That is, if e is                 #
#   nonoverlapping, h will be also.)                                          #
#                                                                             #
#*****************************************************************************#

@njit(cache=True)
def linear_expansion_sum(elen, e, flen, f, h):
    # h cannot be e or f.

    enow = e[0]
    fnow = f[0]
    eindex = 0
    findex = 0
    if (fnow > enow) == (fnow > -enow):
        g0 = enow
        eindex += 1
        enow = e[eindex]
    else:
        g0 = fnow
        findex += 1
        fnow = f[findex]

    if (eindex < elen) and \
        ((findex >= flen) or ((fnow > enow) == (fnow > -enow))):
        Q, q = Fast_Two_Sum(enow, g0)
        eindex += 1
        enow = e[eindex]
    else:
        Q, q = Fast_Two_Sum(fnow, g0)
        findex += 1
        fnow = f[findex]

    for i in range(0, elen+flen-2):
        hindex = i
        if (eindex < elen) and \
            ((findex >= flen) or ((fnow > enow) == (fnow > -enow))):
            R, h[hindex] = Fast_Two_Sum(enow, q)
            eindex += 1
            enow = e[eindex]
        else:
            R, h[hindex] = Fast_Two_Sum(fnow, q)
            findex += 1
            fnow = f[findex]
        Q, q = Two_Sum(Q, R)

    h[hindex] = q
    h[hindex + 1] = Q

    return hindex + 2


#*****************************************************************************#
#                                                                             #
#   linear_expansion_sum_zeroelim()   Sum two expansions, eliminating zero    #
#                                     components from the output expansion.   #
#                                                                             #
#   Sets h = e + f.  See either version of Shewchuk's paper for details.      #
#                                                                             #
#   Maintains the nonoverlapping property.  (That is, if e is                 #
#   nonoverlapping, h will be also.)                                          #
#                                                                             #
#*****************************************************************************#

@njit(cache=True)
def linear_expansion_sum_zeroelim(elen, e, flen, f, h):
    # h cannot be e or f.

    enow = e[0]
    fnow = f[0]
    eindex = 0
    findex = 0
    hindex = 0
    if (fnow > enow) == (fnow > -enow):
        g0 = enow
        eindex += 1
        enow = e[eindex]
    else:
        g0 = fnow
        findex += 1
        fnow = f[findex]

    if (eindex < elen) and \
        ((findex >= flen) or ((fnow > enow) == (fnow > -enow))):
        Q, q = Fast_Two_Sum(enow, g0)
        eindex += 1
        enow = e[eindex]
    else:
        Q, q = Fast_Two_Sum(fnow, g0)
        findex += 1
        fnow = f[findex]

    for count in range(2, elen+flen+1):
        if (eindex < elen) and \
            ((findex >= flen) or ((fnow > enow) == (fnow > -enow))):
            R, hh = Fast_Two_Sum(enow, q)
            eindex += 1
            enow = e[eindex]
        else:
            R, hh = Fast_Two_Sum(fnow, q)
            findex += 1
            fnow = f[findex]
        Q, q = Two_Sum(Q, R)
        if hh != 0:
            h[hindex] = hh
            hindex += 1

    if q != 0:
        h[hindex] = q
        hindex += 1

    if (Q != 0.0) or (hindex == 0):
        h[hindex] = Q
        hindex += 1

    return hindex


#*****************************************************************************#
#                                                                             #
#   scale_expansion()   Multiply an expansion by a scalar.                    #
#                                                                             #
#   Sets h = e + f.  See either version of Shewchuk's paper for details.      #
#                                                                             #
#   Maintains the nonoverlapping property.  If round-to-even is used (as      #
#   with IEEE 754), maintains the strongly nonoverlapping and nonadjacent     #
#   properties as well.  (That is, if e has one of these properties, so       #
#   will h.)                                                                  #
#                                                                             #
#*****************************************************************************#

@njit(cache=True)
def scale_expansion(elen, e, b, h, splitter):
    # e and h cannot be the same.

    bhi, blo = Split(b, splitter)
    Q, h[0] = Two_Product_Presplit(e[0], b, bhi, blo, splitter)
    hindex = 1
    for eindex in range(1, elen):
        enow = e[eindex]
        product1, product0 = Two_Product_Presplit(enow, b, bhi, blo, splitter)
        sum_, h[hindex] = Two_Sum(Q, product0)
        hindex += 1
        Q, h[hindex] = Two_Sum(product1, sum_)
        hindex += 1

    h[hindex] = Q
    return elen + elen


#*****************************************************************************#
#                                                                             #
#   scale_expansion_zeroelim()   Multiply an expansion by a scalar,           #
#                                eliminating zero components from the         #
#                                output expansion.                            #
#                                                                             #
#   Sets h = e + f.  See either version of Shewchuk's paper for details.      #
#                                                                             #
#   Maintains the nonoverlapping property.  If round-to-even is used (as      #
#   with IEEE 754), maintains the strongly nonoverlapping and nonadjacent     #
#   properties as well.  (That is, if e has one of these properties, so       #
#   will h.)                                                                  #
#                                                                             #
#*****************************************************************************#

@njit(cache=True)
def scale_expansion_zeroelim(elen, e, b, h, splitter):
    # e and h cannot be the same.

    bhi, blo = Split(b, splitter)
    Q, hh = Two_Product_Presplit(e[0], b, bhi, blo, splitter)
    hindex = 0
    if hh != 0:
        h[hindex] = hh
        hindex += 1

    for eindex in range(1, elen):
        enow = e[eindex]
        product1, product0 = Two_Product_Presplit(enow, b, bhi, blo, splitter)
        sum_, hh = Two_Sum(Q, product0)
        if hh != 0:
            h[hindex] = hh
            hindex +=1
        Q, hh = Fast_Two_Sum(product1, sum_)
        if hh != 0:
            h[hindex] = hh
            hindex += 1

    if (Q != 0.0) or (hindex == 0):
        h[hindex] = Q
        hindex += 1

    return hindex


#*****************************************************************************#
#                                                                             #
#   compress()   Compress an expansion.                                       #
#                                                                             #
#   See the long version of Shewchuk's paper for details.                     #
#                                                                             #
#   Maintains the nonoverlapping property.  If round-to-even is used (as      #
#   with IEEE 754), then any nonoverlapping expansion is converted to a       #
#   nonadjacent expansion.                                                    #
#                                                                             #
#*****************************************************************************#

@njit(cache=True)
def compress(elen, e, h):
    # e and h may be the same.

    bottom = elen - 1
    Q = e[bottom]
    for eindex in range(elen-2, -1, -1):
        enow = e[eindex]
        Qnew, q = Fast_Two_Sum(Q, enow)
        if q != 0:
            h[bottom] = Qnew
            bottom -= 1
            Q = q
        else :
            Q = Qnew

    top = 0
    for hindex in range(bottom+1, elen):
        hnow = h[hindex]
        Qnew, q = Fast_Two_Sum(hnow, Q)
        if q != 0:
            h[top] = q
            top += 1
        Q = Qnew

    h[top] = Q
    return top + 1


#*****************************************************************************#
#                                                                             #
#   estimate()   Produce a one-word estimate of an expansion's value.         #
#                                                                             #
#   See either version of Shewchuk's paper for details.                       #
#                                                                             #
#*****************************************************************************#

@njit(cache=True)
def estimate(elen, e):

    Q = e[0]
    for eindex in range(1, elen):
        Q += e[eindex]
    return Q


#*****************************************************************************#
#                                                                             #
#   orient2d()   Adaptive exact 2D orientation test.  Robust.                 #
#                                                                             #
#                Return a positive value if the points pa, pb, and pc occur   #
#                in counterclockwise order; a negative value if they occur    #
#                in clockwise order; and zero if they are collinear.  The     #
#                result is also a rough approximation of twice the signed     #
#                area of the triangle defined by the three points.            #
#                                                                             #
#   This uses exact arithmetic to ensure a correct answer.  The               #
#   result returned is the determinant of a matrix.  In orient2d() only,      #
#   this determinant is computed adaptively, in the sense that exact          #
#   arithmetic is used only to the degree it is needed to ensure that the     #
#   returned value has the correct sign.  Hence, orient2d() is usually quite  #
#   fast, but will run more slowly when the input points are collinear or     #
#   nearly so.                                                                #
#                                                                             #
#*****************************************************************************#

@njit(cache=True)
def orient2dadapt(
        pa_x, pa_y, pb_x, pb_y, pc_x, pc_y, detsum, splitter, global_arr,
        ccwerrboundB, ccwerrboundC, resulterrbound):
    '''
    len(B) = 4
    len(C1) = 8
    len(C2) = 12
    len(D) = 16
    len(u) = 4
    '''

    B = global_arr[0:4]
    C1 = global_arr[4:12]
    C2 = global_arr[12:24]
    D = global_arr[24:40]
    u = global_arr[40:44]

    bcx = pb_x - pc_x
    acx = pa_x - pc_x
    acy = pa_y - pc_y
    bcy = pb_y - pc_y

    detleft, detlefttail = Two_Product(acx, bcy, splitter)
    detright, detrighttail = Two_Product(acy, bcx, splitter)

    B[3], B[2], B[1], B[0] = Two_Two_Diff(detleft, detlefttail,
                                          detright, detrighttail)

    det = estimate(4, B)
    errbound = ccwerrboundB * detsum
    if (det >= errbound) or (-det >= errbound):
        return det

    acxtail = Two_Diff_Tail(pa_x, pc_x, acx)
    bcxtail = Two_Diff_Tail(pb_x, pc_x, bcx)
    acytail = Two_Diff_Tail(pa_y, pc_y, acy)
    bcytail = Two_Diff_Tail(pb_y, pc_y, bcy)

    if (acxtail == 0.0) and (acytail == 0.0) and \
            (bcxtail == 0.0) and (bcytail == 0.0):
        return det

    errbound = ccwerrboundC * detsum + resulterrbound * np.abs(det)
    det += (acx * bcytail + bcy * acxtail) - \
           (acy * bcxtail + bcx * acytail)
    # if (det >= errbound) or (-det >= errbound):
    if np.abs(det) >= errbound:
        return det

    s1, s0 = Two_Product(acxtail, bcy, splitter)
    t1, t0 = Two_Product(acytail, bcx, splitter)
    u[3], u[2], u[1], u[0] = Two_Two_Diff(s1, s0, t1, t0)
    C1length = fast_expansion_sum_zeroelim(4, B, 4, u, C1)

    s1, s0 = Two_Product(acx, bcytail, splitter)
    t1, t0 = Two_Product(acy, bcxtail, splitter)
    u[3], u[2], u[1], u[0] = Two_Two_Diff(s1, s0, t1, t0)
    C2length = fast_expansion_sum_zeroelim(C1length, C1, 4, u, C2)

    s1, s0 = Two_Product(acxtail, bcytail, splitter)
    t1, t0 = Two_Product(acytail, bcxtail, splitter)
    u[3], u[2], u[1], u[0] = Two_Two_Diff(s1, s0, t1, t0)
    Dlength = fast_expansion_sum_zeroelim(C2length, C2, 4, u, D)

    return D[Dlength - 1]


@njit(cache=True)
def orient2d(pa_x, pa_y, pb_x, pb_y, pc_x, pc_y, res_arr, global_arr):
    '''
    len(B) = 4
    len(C1) = 8
    len(C2) = 12
    len(D) = 16
    len(u) = 4
    '''
    splitter = res_arr[0]
    resulterrbound = res_arr[1]
    ccwerrboundA = res_arr[2]
    ccwerrboundB = res_arr[3]
    ccwerrboundC = res_arr[4]
    static_filter_o2d = res_arr[8]

    det_left = (pa_x - pc_x)*(pb_y - pc_y)
    det_right = (pa_y - pc_y)*(pb_x - pc_x)
    det = det_left - det_right
    if np.abs(det) > static_filter_o2d:
        return det

    detsum = np.abs(det_left) + np.abs(det_right)
    errbound = ccwerrboundA * detsum
    if np.abs(det) > errbound:
        return det

    return orient2dadapt(
        pa_x, pa_y, pb_x, pb_y, pc_x, pc_y, detsum, splitter, global_arr,
        ccwerrboundB, ccwerrboundC, resulterrbound)


#*****************************************************************************#
#                                                                             #
#   orient3d()   Adaptive exact 3D orientation test.  Robust.                 #
#                                                                             #
#                Return a positive value if the point pd lies below the       #
#                plane passing through pa, pb, and pc; "below" is defined so  #
#                that pa, pb, and pc appear in counterclockwise order when    #
#                viewed from above the plane.  Returns a negative value if    #
#                pd lies above the plane.  Returns zero if the points are     #
#                coplanar.  The result is also a rough approximation of six   #
#                times the signed volume of the tetrahedron defined by the    #
#                four points.                                                 #
#                                                                             #
#   This uses exact arithmetic to ensure a correct answer.  The               #
#   result returned is the determinant of a matrix.  In orient3d() only,      #
#   this determinant is computed adaptively, in the sense that exact          #
#   arithmetic is used only to the degree it is needed to ensure that the     #
#   returned value has the correct sign.  Hence, orient3d() is usually quite  #
#   fast, but will run more slowly when the input points are coplanar or      #
#   nearly so.                                                                #
#                                                                             #
#*****************************************************************************#

@njit(cache=True)
def orient3dadapt(pa, pb, pc, pd, permanent, bc, ca, ab, adet, bdet, cdet,
                  abdet, fin1, fin2, at_b, at_c, bt_c, bt_a, ct_a, ct_b,
                  bct, cat, abt, u, v, w, splitter, o3derrboundB, o3derrboundC,
                  resulterrbound):
    '''
    len(bc) = 4
    len(ca) = 4
    len(ab) = 4
    len(adet) = 8
    len(bdet) = 8
    len(cdet) = 8
    len(abdet) = 16
    len(fin1) = 192
    len(fin2) = 192
    len(at_b) = 4
    len(at_c) = 4
    len(bt_c) = 4
    len(bt_a) = 4
    len(ct_a) = 4
    len(ct_b) = 4
    len(bct) = 8
    len(cat) = 8
    len(abt) = 8
    len(u) = 4
    len(v) = 12
    len(w) = 16
    '''

    adx = pa[0] - pd[0]
    bdx = pb[0] - pd[0]
    cdx = pc[0] - pd[0]
    ady = pa[1] - pd[1]
    bdy = pb[1] - pd[1]
    cdy = pc[1] - pd[1]
    adz = pa[2] - pd[2]
    bdz = pb[2] - pd[2]
    cdz = pc[2] - pd[2]

    bdxcdy1, bdxcdy0 = Two_Product(bdx, cdy, splitter)
    cdxbdy1, cdxbdy0 = Two_Product(cdx, bdy, splitter)
    bc[3], bc[2], bc[1], bc[0] = Two_Two_Diff(bdxcdy1, bdxcdy0,
                                              cdxbdy1, cdxbdy0)
    alen = scale_expansion_zeroelim(4, bc, adz, adet, splitter)

    cdxady1, cdxady0 = Two_Product(cdx, ady, splitter)
    adxcdy1, adxcdy0 = Two_Product(adx, cdy, splitter)
    ca[3], ca[2], ca[1], ca[0] = Two_Two_Diff(cdxady1, cdxady0,
                                              adxcdy1, adxcdy0)
    blen = scale_expansion_zeroelim(4, ca, bdz, bdet, splitter)

    adxbdy1, adxbdy0 = Two_Product(adx, bdy, splitter)
    bdxady1, bdxady0 = Two_Product(bdx, ady, splitter)
    ab[3], ab[2], ab[1], ab[0] = Two_Two_Diff(adxbdy1, adxbdy0,
                                              bdxady1, bdxady0)
    clen = scale_expansion_zeroelim(4, ab, cdz, cdet, splitter)

    ablen = fast_expansion_sum_zeroelim(alen, adet, blen, bdet, abdet)
    finlength = fast_expansion_sum_zeroelim(ablen, abdet, clen, cdet, fin1)

    det = estimate(finlength, fin1)
    errbound = o3derrboundB * permanent
    if (det >= errbound) or (-det >= errbound):
        return det

    adxtail = Two_Diff_Tail(pa[0], pd[0], adx)
    bdxtail = Two_Diff_Tail(pb[0], pd[0], bdx)
    cdxtail = Two_Diff_Tail(pc[0], pd[0], cdx)
    adytail = Two_Diff_Tail(pa[1], pd[1], ady)
    bdytail = Two_Diff_Tail(pb[1], pd[1], bdy)
    cdytail = Two_Diff_Tail(pc[1], pd[1], cdy)
    adztail = Two_Diff_Tail(pa[2], pd[2], adz)
    bdztail = Two_Diff_Tail(pb[2], pd[2], bdz)
    cdztail = Two_Diff_Tail(pc[2], pd[2], cdz)

    if (adxtail == 0.0) and (bdxtail == 0.0) and (cdxtail == 0.0) and \
        (adytail == 0.0) and (bdytail == 0.0) and (cdytail == 0.0) and \
            (adztail == 0.0) and (bdztail == 0.0) and (cdztail == 0.0):
        return det

    errbound = o3derrboundC * permanent + resulterrbound * np.abs(det)
    det += (adz * ((bdx * cdytail + cdy * bdxtail)
                   - (bdy * cdxtail + cdx * bdytail))
            + adztail * (bdx * cdy - bdy * cdx)) + \
           (bdz * ((cdx * adytail + ady * cdxtail)
                   - (cdy * adxtail + adx * cdytail))
            + bdztail * (cdx * ady - cdy * adx)) + \
           (cdz * ((adx * bdytail + bdy * adxtail)
                   - (ady * bdxtail + bdx * adytail))
            + cdztail * (adx * bdy - ady * bdx))
    if (det >= errbound) or (-det >= errbound):
        return det

    finnow = fin1
    finother = fin2

    if adxtail == 0.0:
        if adytail == 0.0:
            at_b[0] = 0.0
            at_blen = 1
            at_c[0] = 0.0
            at_clen = 1
        else:
            negate = -adytail
            at_blarge, at_b[0] = Two_Product(negate, bdx, splitter)
            at_b[1] = at_blarge
            at_blen = 2
            at_clarge, at_c[0] = Two_Product(adytail, cdx, splitter)
            at_c[1] = at_clarge
            at_clen = 2
    else:
        if adytail == 0.0:
            at_blarge, at_b[0] = Two_Product(adxtail, bdy, splitter)
            at_b[1] = at_blarge
            at_blen = 2
            negate = -adxtail
            at_clarge, at_c[0] = Two_Product(negate, cdy, splitter)
            at_c[1] = at_clarge
            at_clen = 2
        else:
            adxt_bdy1, adxt_bdy0 = Two_Product(adxtail, bdy, splitter)
            adyt_bdx1, adyt_bdx0 = Two_Product(adytail, bdx, splitter)
            at_blarge, at_b[2], at_b[1], at_b[0] = Two_Two_Diff(adxt_bdy1,
                                                                adxt_bdy0,
                                                                adyt_bdx1,
                                                                adyt_bdx0)
            at_b[3] = at_blarge
            at_blen = 4
            adyt_cdx1, adyt_cdx0 = Two_Product(adytail, cdx, splitter)
            adxt_cdy1, adxt_cdy0 = Two_Product(adxtail, cdy, splitter)
            at_clarge, at_c[2], at_c[1], at_c[0] = Two_Two_Diff(adyt_cdx1,
                                                                adyt_cdx0,
                                                                adxt_cdy1,
                                                                adxt_cdy0)
            at_c[3] = at_clarge
            at_clen = 4

    if bdxtail == 0.0:
        if bdytail == 0.0:
            bt_c[0] = 0.0
            bt_clen = 1
            bt_a[0] = 0.0
            bt_alen = 1
        else:
            negate = -bdytail
            bt_clarge, bt_c[0] = Two_Product(negate, cdx, splitter)
            bt_c[1] = bt_clarge
            bt_clen = 2
            bt_alarge, bt_a[0] = Two_Product(bdytail, adx, splitter)
            bt_a[1] = bt_alarge
            bt_alen = 2
    else:
        if bdytail == 0.0:
            bt_clarge, bt_c[0] = Two_Product(bdxtail, cdy, splitter)
            bt_c[1] = bt_clarge
            bt_clen = 2
            negate = -bdxtail
            bt_alarge, bt_a[0] = Two_Product(negate, ady, splitter)
            bt_a[1] = bt_alarge
            bt_alen = 2
        else:
            bdxt_cdy1, bdxt_cdy0 = Two_Product(bdxtail, cdy, splitter)
            bdyt_cdx1, bdyt_cdx0 = Two_Product(bdytail, cdx, splitter)
            bt_clarge, bt_c[2], bt_c[1], bt_c[0] = Two_Two_Diff(bdxt_cdy1,
                                                                bdxt_cdy0,
                                                                bdyt_cdx1,
                                                                bdyt_cdx0)
            bt_c[3] = bt_clarge
            bt_clen = 4
            bdyt_adx1, bdyt_adx0 = Two_Product(bdytail, adx, splitter)
            bdxt_ady1, bdxt_ady0 = Two_Product(bdxtail, ady, splitter)
            bt_alarge, bt_a[2], bt_a[1], bt_a[0] = Two_Two_Diff(bdyt_adx1,
                                                                bdyt_adx0,
                                                                bdxt_ady1,
                                                                bdxt_ady0)
            bt_a[3] = bt_alarge
            bt_alen = 4

    if cdxtail == 0.0:
        if cdytail == 0.0:
            ct_a[0] = 0.0
            ct_alen = 1
            ct_b[0] = 0.0
            ct_blen = 1
        else:
            negate = -cdytail
            ct_alarge, ct_a[0] = Two_Product(negate, adx, splitter)
            ct_a[1] = ct_alarge
            ct_alen = 2
            ct_blarge, ct_b[0] = Two_Product(cdytail, bdx, splitter)
            ct_b[1] = ct_blarge
            ct_blen = 2
    else:
        if cdytail == 0.0:
            ct_alarge, ct_a[0] = Two_Product(cdxtail, ady, splitter)
            ct_a[1] = ct_alarge
            ct_alen = 2
            negate = -cdxtail
            ct_blarge, ct_b[0] = Two_Product(negate, bdy, splitter)
            ct_b[1] = ct_blarge
            ct_blen = 2
        else:
            cdxt_ady1, cdxt_ady0 = Two_Product(cdxtail, ady, splitter)
            cdyt_adx1, cdyt_adx0 = Two_Product(cdytail, adx, splitter)
            ct_alarge, ct_a[2], ct_a[1], ct_a[0] = Two_Two_Diff(cdxt_ady1,
                                                                cdxt_ady0,
                                                                cdyt_adx1,
                                                                cdyt_adx0)
            ct_a[3] = ct_alarge
            ct_alen = 4
            cdyt_bdx1, cdyt_bdx0 = Two_Product(cdytail, bdx, splitter)
            cdxt_bdy1, cdxt_bdy0 = Two_Product(cdxtail, bdy, splitter)
            ct_blarge, ct_b[2], ct_b[1], ct_b[0] = Two_Two_Diff(cdyt_bdx1,
                                                                cdyt_bdx0,
                                                                cdxt_bdy1,
                                                                cdxt_bdy0)
            ct_b[3] = ct_blarge
            ct_blen = 4

    bctlen = fast_expansion_sum_zeroelim(bt_clen, bt_c, ct_blen, ct_b, bct)
    wlength = scale_expansion_zeroelim(bctlen, bct, adz, w, splitter)
    finlength = fast_expansion_sum_zeroelim(finlength, finnow, wlength, w,
                                            finother)
    finswap = finnow
    finnow = finother
    finother = finswap

    catlen = fast_expansion_sum_zeroelim(ct_alen, ct_a, at_clen, at_c, cat)
    wlength = scale_expansion_zeroelim(catlen, cat, bdz, w, splitter)
    finlength = fast_expansion_sum_zeroelim(finlength, finnow, wlength, w,
                                            finother)
    finswap = finnow
    finnow = finother
    finother = finswap

    abtlen = fast_expansion_sum_zeroelim(at_blen, at_b, bt_alen, bt_a, abt)
    wlength = scale_expansion_zeroelim(abtlen, abt, cdz, w, splitter)
    finlength = fast_expansion_sum_zeroelim(finlength, finnow, wlength, w,
                                            finother)
    finswap = finnow
    finnow = finother
    finother = finswap

    if adztail != 0.0:
        vlength = scale_expansion_zeroelim(4, bc, adztail, v, splitter)
        finlength = fast_expansion_sum_zeroelim(finlength, finnow, vlength, v,
                                                finother)
        finswap = finnow
        finnow = finother
        finother = finswap

    if bdztail != 0.0:
        vlength = scale_expansion_zeroelim(4, ca, bdztail, v, splitter)
        finlength = fast_expansion_sum_zeroelim(finlength, finnow, vlength, v,
                                                finother)
        finswap = finnow
        finnow = finother
        finother = finswap

    if cdztail != 0.0:
        vlength = scale_expansion_zeroelim(4, ab, cdztail, v, splitter)
        finlength = fast_expansion_sum_zeroelim(finlength, finnow, vlength, v,
                                                finother)
        finswap = finnow
        finnow = finother
        finother = finswap

    if adxtail != 0.0:
        if bdytail != 0.0:
            adxt_bdyt1, adxt_bdyt0 = Two_Product(adxtail, bdytail, splitter)
            u[3], u[2], u[1], u[0] = Two_One_Product(adxt_bdyt1, adxt_bdyt0,
                                                     cdz, splitter)
            finlength = fast_expansion_sum_zeroelim(finlength, finnow, 4, u,
                                                    finother)
            finswap = finnow
            finnow = finother
            finother = finswap
            if cdztail != 0.0:
                u[3], u[2], u[1], u[0] = Two_One_Product(adxt_bdyt1,
                                                         adxt_bdyt0,
                                                         cdztail,
                                                         splitter)
                finlength = fast_expansion_sum_zeroelim(finlength, finnow,
                                                        4, u, finother)
                finswap = finnow
                finnow = finother
                finother = finswap
        if cdytail != 0.0:
            negate = -adxtail
            adxt_cdyt1, adxt_cdyt0 = Two_Product(negate, cdytail, splitter)
            u[3], u[2], u[1], u[0] = Two_One_Product(adxt_cdyt1, adxt_cdyt0,
                                                     bdz, splitter)
            finlength = fast_expansion_sum_zeroelim(finlength, finnow, 4, u,
                                                    finother)
            finswap = finnow
            finnow = finother
            finother = finswap
            if bdztail != 0.0:
                u[3], u[2], u[1], u[0] = Two_One_Product(adxt_cdyt1,
                                                         adxt_cdyt0,
                                                         bdztail,
                                                         splitter)
                finlength = fast_expansion_sum_zeroelim(finlength, finnow,
                                                        4, u, finother)
                finswap = finnow
                finnow = finother
                finother = finswap

    if bdxtail != 0.0:
        if cdytail != 0.0:
            bdxt_cdyt1, bdxt_cdyt0 = Two_Product(bdxtail, cdytail, splitter)
            u[3], u[2], u[1], u[0] = Two_One_Product(bdxt_cdyt1, bdxt_cdyt0,
                                                     adz, splitter)
            finlength = fast_expansion_sum_zeroelim(finlength, finnow, 4, u,
                                                  finother)
            finswap = finnow
            finnow = finother
            finother = finswap
            if adztail != 0.0:
                u[3], u[2], u[1], u[0] = Two_One_Product(bdxt_cdyt1,
                                                         bdxt_cdyt0,
                                                         adztail,
                                                         splitter)
                finlength = fast_expansion_sum_zeroelim(finlength, finnow,
                                                        4, u, finother)
                finswap = finnow
                finnow = finother
                finother = finswap
        if adytail != 0.0:
            negate = -bdxtail
            bdxt_adyt1, bdxt_adyt0 = Two_Product(negate, adytail, splitter)
            u3, u[2], u[1], u[0] = Two_One_Product(bdxt_adyt1, bdxt_adyt0,
                                                   cdz, splitter)
            finlength = fast_expansion_sum_zeroelim(finlength, finnow, 4, u,
                                                    finother)
            finswap = finnow
            finnow = finother
            finother = finswap
            if cdztail != 0.0:
                u[3], u[2], u[1], u[0] = Two_One_Product(bdxt_adyt1,
                                                         bdxt_adyt0,
                                                         cdztail,
                                                         splitter)
                finlength = fast_expansion_sum_zeroelim(finlength, finnow,
                                                        4, u, finother)
                finswap = finnow
                finnow = finother
                finother = finswap

    if cdxtail != 0.0:
        if adytail != 0.0:
            cdxt_adyt1, cdxt_adyt0 = Two_Product(cdxtail, adytail, splitter)
            u[3], u[2], u[1], u[0] = Two_One_Product(cdxt_adyt1, cdxt_adyt0,
                                                     bdz, splitter)
            finlength = fast_expansion_sum_zeroelim(finlength, finnow, 4, u,
                                                    finother)
            finswap = finnow
            finnow = finother
            finother = finswap
            if bdztail != 0.0:
                u[3], u[2], u[1], u[0] = Two_One_Product(cdxt_adyt1,
                                                         cdxt_adyt0,
                                                         bdztail,
                                                         splitter)
                finlength = fast_expansion_sum_zeroelim(finlength, finnow,
                                                        4, u, finother)
                finswap = finnow
                finnow = finother
                finother = finswap
        if bdytail != 0.0:
            negate = -cdxtail
            cdxt_bdyt1, cdxt_bdyt0 = Two_Product(negate, bdytail, splitter)
            u[3], u[2], u[1], u[0] = Two_One_Product(cdxt_bdyt1, cdxt_bdyt0,
                                                     adz, splitter)
            finlength = fast_expansion_sum_zeroelim(finlength, finnow, 4, u,
                                                    finother)
            finswap = finnow
            finnow = finother
            finother = finswap
            if adztail != 0.0:
                u[3], u[2], u[1], u[0] = Two_One_Product(cdxt_bdyt1,
                                                         cdxt_bdyt0,
                                                         adztail,
                                                         splitter)
                finlength = fast_expansion_sum_zeroelim(finlength, finnow,
                                                        4, u, finother)
                finswap = finnow
                finnow = finother
                finother = finswap

    if adztail != 0.0:
        wlength = scale_expansion_zeroelim(bctlen, bct, adztail, w, splitter)
        finlength = fast_expansion_sum_zeroelim(finlength, finnow, wlength, w,
                                                finother)
        finswap = finnow
        finnow = finother
        finother = finswap

    if bdztail != 0.0:
        wlength = scale_expansion_zeroelim(catlen, cat, bdztail, w, splitter)
        finlength = fast_expansion_sum_zeroelim(finlength, finnow, wlength, w,
                                                finother)
        finswap = finnow
        finnow = finother
        finother = finswap

    if cdztail != 0.0:
        wlength = scale_expansion_zeroelim(abtlen, abt, cdztail, w, splitter)
        finlength = fast_expansion_sum_zeroelim(finlength, finnow, wlength, w,
                                                finother)
        finswap = finnow
        finnow = finother
        finother = finswap

    return finnow[finlength - 1]


@njit(cache=True)
def orient3d(pa, pb, pc, pd, permanent, bc, ca, ab, adet, bdet, cdet,
             abdet, fin1, fin2, at_b, at_c, bt_c, bt_a, ct_a, ct_b,
             bct, cat, abt, u, v, w, splitter, o3derrboundA, o3derrboundB,
             o3derrboundC, resulterrbound):
    '''
    len(bc) = 4
    len(ca) = 4
    len(ab) = 4
    len(adet) = 8
    len(bdet) = 8
    len(cdet) = 8
    len(abdet) = 16
    len(fin1) = 192
    len(fin2) = 192
    len(at_b) = 4
    len(at_c) = 4
    len(bt_c) = 4
    len(bt_a) = 4
    len(ct_a) = 4
    len(ct_b) = 4
    len(bct) = 8
    len(cat) = 8
    len(abt) = 8
    len(u) = 4
    len(v) = 12
    len(w) = 16
    '''

    adx = pa[0] - pd[0]
    bdx = pb[0] - pd[0]
    cdx = pc[0] - pd[0]
    ady = pa[1] - pd[1]
    bdy = pb[1] - pd[1]
    cdy = pc[1] - pd[1]
    adz = pa[2] - pd[2]
    bdz = pb[2] - pd[2]
    cdz = pc[2] - pd[2]

    bdxcdy = bdx * cdy
    cdxbdy = cdx * bdy

    cdxady = cdx * ady
    adxcdy = adx * cdy

    adxbdy = adx * bdy
    bdxady = bdx * ady

    det = adz * (bdxcdy - cdxbdy) + \
          bdz * (cdxady - adxcdy) + \
          cdz * (adxbdy - bdxady)

    permanent = (np.abs(bdxcdy) + np.abs(cdxbdy)) * np.abs(adz) + \
                (np.abs(cdxady) + np.abs(adxcdy)) * np.abs(bdz) + \
                (np.abs(adxbdy) + np.abs(bdxady)) * np.abs(cdz)
    errbound = o3derrboundA * permanent
    if (det > errbound) or (-det > errbound):
        return det

    return orient3dadapt(pa, pb, pc, pd, permanent)


#*****************************************************************************#
#                                                                             #
#   incircle()   Adaptive exact 2D incircle test.  Robust.                    #
#                                                                             #
#                Return a positive value if the point pd lies inside the      #
#                circle passing through pa, pb, and pc; a negative value if   #
#                it lies outside; and zero if the four points are cocircular. #
#                The points pa, pb, and pc must be in counterclockwise        #
#                order, or the sign of the result will be reversed.           #
#                                                                             #
#   This uses exact arithmetic to ensure a correct answer.  The               #
#   result returned is the determinant of a matrix.  In incircle() only,      #
#   this determinant is computed adaptively, in the sense that exact          #
#   arithmetic is used only to the degree it is needed to ensure that the     #
#   returned value has the correct sign.  Hence, incircle() is usually quite  #
#   fast, but will run more slowly when the input points are cocircular or    #
#   nearly so.                                                                #
#                                                                             #
#*****************************************************************************#

@njit(cache=True)
def incircleadapt(
        pa_x, pa_y, pb_x, pb_y, pc_x, pc_y, pd_x, pd_y, permanent, global_arr,
        splitter, iccerrboundB, iccerrboundC, resulterrbound):
    '''
    len(bc) = 4
    len(ca) = 4
    len(ab) = 4
    len(axbc) = 8
    len(axxbc) = 16
    len(aybc) = 8
    len(ayybc) = 16
    len(adet) = 32
    len(bxca) = 8
    len(bxxca) = 16
    len(byca) = 8
    len(byyca) = 16
    len(bdet) = 32
    len(cxab) = 8
    len(cxxab) = 16
    len(cyab) = 8
    len(cyyab) = 16
    len(cdet) = 32
    len(abdet) = 64
    len(fin1) = 1152
    len(fin2) = 1152
    len(aa) = 4
    len(bb) = 4
    len(cc) = 4
    len(u) = 4
    len(v) = 4
    len(temp8) = 8
    len(temp16a) = 16
    len(temp16b) = 16
    len(temp16c) = 16
    len(temp32a) = 32
    len(temp32b) = 32
    len(temp48) = 48
    len(temp64) = 64
    len(axtbb) = 8
    len(axtcc) = 8
    len(aytbb) = 8
    len(aytcc) = 8
    len(bxtaa) = 8
    len(bxtcc) = 8
    len(bytaa) = 8
    len(bytcc) = 8
    len(cxtaa) = 8
    len(cxtbb) = 8
    len(cytaa) = 8
    len(cytbb) = 8
    len(axtbc) = 8
    len(aytbc) = 8
    len(bxtca) = 8
    len(bytca) = 8
    len(cxtab) = 8
    len(cytab) = 8
    len(axtbct) = 16
    len(aytbct) = 16
    len(bxtcat) = 16
    len(bytcat) = 16
    len(cxtabt) = 16
    len(cytabt) = 16
    len(axtbctt) = 8
    len(aytbctt) = 8
    len(bxtcatt) = 8
    len(bytcatt) = 8
    len(cxtabtt) = 8
    len(cytabtt) = 8
    len(abt) = 8
    len(bct) = 8
    len(cat) = 8
    len(abtt) = 4
    len(bctt) = 4
    len(catt) = 4
    '''

    u = global_arr[40:44]
    v = global_arr[44:48]
    bc = global_arr[48:52]
    ca = global_arr[52:56]
    ab = global_arr[56:60]
    axbc = global_arr[60:68]
    axxbc = global_arr[68:84]
    aybc = global_arr[84:92]
    ayybc = global_arr[92:108]
    adet = global_arr[108:140]
    bxca = global_arr[140:148]
    bxxca = global_arr[148:164]
    byca = global_arr[164:172]
    byyca = global_arr[172:188]
    bdet = global_arr[188:220]
    cxab = global_arr[220:228]
    cxxab = global_arr[228:244]
    cyab = global_arr[244:252]
    cyyab = global_arr[252:268]
    cdet = global_arr[268:300]
    abdet = global_arr[300:364]
    fin1 = global_arr[364:1516]
    fin2 = global_arr[1516:2668]
    aa = global_arr[2668:2672]
    bb = global_arr[2672:2676]
    cc = global_arr[2676:2680]
    temp8 = global_arr[2680:2688]
    temp16a = global_arr[2688:2704]
    temp16b = global_arr[2704:2720]
    temp16c = global_arr[2720:2736]
    temp32a = global_arr[2736:2768]
    temp32b = global_arr[2768:2800]
    temp48 = global_arr[2800:2848]
    temp64 = global_arr[2848:2912]
    axtbb = global_arr[2912:2920]
    axtcc = global_arr[2920:2928]
    aytbb = global_arr[2928:2936]
    aytcc = global_arr[2936:2944]
    bxtaa = global_arr[2944:2952]
    bxtcc = global_arr[2952:2960]
    bytaa = global_arr[2960:2968]
    bytcc = global_arr[2968:2976]
    cxtaa = global_arr[2976:2984]
    cxtbb = global_arr[2984:2992]
    cytaa = global_arr[2992:3000]
    cytbb = global_arr[3000:3008]
    axtbc = global_arr[3008:3016]
    aytbc = global_arr[3016:3024]
    bxtca = global_arr[3024:3032]
    bytca = global_arr[3032:3040]
    cxtab = global_arr[3040:3048]
    cytab = global_arr[3048:3056]
    axtbct = global_arr[3056:3072]
    aytbct = global_arr[3072:3088]
    bxtcat = global_arr[3088:3104]
    bytcat = global_arr[3104:3120]
    cxtabt = global_arr[3120:3136]
    cytabt = global_arr[3136:3152]
    axtbctt = global_arr[3152:3160]
    aytbctt = global_arr[3160:3168]
    bxtcatt = global_arr[3168:3176]
    bytcatt = global_arr[3176:3184]
    cxtabtt = global_arr[3184:3192]
    cytabtt = global_arr[3192:3200]
    abt = global_arr[3200:3208]
    bct = global_arr[3208:3216]
    cat = global_arr[3216:3224]
    abtt = global_arr[3224:3228]
    bctt = global_arr[3228:3232]
    catt = global_arr[3232:3236]

    adx = pa_x - pd_x
    bdx = pb_x - pd_x
    cdx = pc_x - pd_x
    ady = pa_y - pd_y
    bdy = pb_y - pd_y
    cdy = pc_y - pd_y

    bdxcdy1, bdxcdy0 = Two_Product(bdx, cdy, splitter)
    cdxbdy1, cdxbdy0 = Two_Product(cdx, bdy, splitter)
    bc[3], bc[2], bc[1], bc[0] = Two_Two_Diff(bdxcdy1, bdxcdy0,
                                              cdxbdy1, cdxbdy0)
    axbclen = scale_expansion_zeroelim(4, bc, adx, axbc, splitter)
    axxbclen = scale_expansion_zeroelim(axbclen, axbc, adx, axxbc, splitter)
    aybclen = scale_expansion_zeroelim(4, bc, ady, aybc, splitter)
    ayybclen = scale_expansion_zeroelim(aybclen, aybc, ady, ayybc, splitter)
    alen = fast_expansion_sum_zeroelim(axxbclen, axxbc, ayybclen, ayybc, adet)

    cdxady1, cdxady0 = Two_Product(cdx, ady, splitter)
    adxcdy1, adxcdy0 = Two_Product(adx, cdy, splitter)
    ca[3], ca[2], ca[1], ca[0] = Two_Two_Diff(cdxady1, cdxady0,
                                              adxcdy1, adxcdy0)
    bxcalen = scale_expansion_zeroelim(4, ca, bdx, bxca, splitter)
    bxxcalen = scale_expansion_zeroelim(bxcalen, bxca, bdx, bxxca, splitter)
    bycalen = scale_expansion_zeroelim(4, ca, bdy, byca, splitter)
    byycalen = scale_expansion_zeroelim(bycalen, byca, bdy, byyca, splitter)
    blen = fast_expansion_sum_zeroelim(bxxcalen, bxxca, byycalen, byyca, bdet)

    adxbdy1, adxbdy0 = Two_Product(adx, bdy, splitter)
    bdxady1, bdxady0 = Two_Product(bdx, ady, splitter)
    ab[3], ab[2], ab[1], ab[0] = Two_Two_Diff(adxbdy1, adxbdy0,
                                              bdxady1, bdxady0)
    cxablen = scale_expansion_zeroelim(4, ab, cdx, cxab, splitter)
    cxxablen = scale_expansion_zeroelim(cxablen, cxab, cdx, cxxab, splitter)
    cyablen = scale_expansion_zeroelim(4, ab, cdy, cyab, splitter)
    cyyablen = scale_expansion_zeroelim(cyablen, cyab, cdy, cyyab, splitter)
    clen = fast_expansion_sum_zeroelim(cxxablen, cxxab, cyyablen, cyyab, cdet)

    ablen = fast_expansion_sum_zeroelim(alen, adet, blen, bdet, abdet)
    finlength = fast_expansion_sum_zeroelim(ablen, abdet, clen, cdet, fin1)

    det = estimate(finlength, fin1)
    errbound = iccerrboundB * permanent
    if (det >= errbound) or (-det >= errbound):
        return det

    adxtail = Two_Diff_Tail(pa_x, pd_x, adx)
    adytail = Two_Diff_Tail(pa_y, pd_y, ady)
    bdxtail = Two_Diff_Tail(pb_x, pd_x, bdx)
    bdytail = Two_Diff_Tail(pb_y, pd_y, bdy)
    cdxtail = Two_Diff_Tail(pc_x, pd_x, cdx)
    cdytail = Two_Diff_Tail(pc_y, pd_y, cdy)
    if (adxtail == 0.0) and (bdxtail == 0.0) and (cdxtail == 0.0) and \
            (adytail == 0.0) and (bdytail == 0.0) and (cdytail == 0.0):
        return det


    errbound = iccerrboundC * permanent + resulterrbound * np.abs(det)
    det += ((adx * adx + ady * ady) * ((bdx * cdytail + cdy * bdxtail)
                                       - (bdy * cdxtail + cdx * bdytail))
            + 2.0*(adx * adxtail + ady * adytail)*(bdx * cdy - bdy * cdx)) + \
           ((bdx * bdx + bdy * bdy) * ((cdx * adytail + ady * cdxtail)
                                       - (cdy * adxtail + adx * cdytail))
            + 2.0*(bdx * bdxtail + bdy * bdytail)*(cdx * ady - cdy * adx)) + \
           ((cdx * cdx + cdy * cdy) * ((adx * bdytail + bdy * adxtail)
                                       - (ady * bdxtail + bdx * adytail))
            + 2.0*(cdx * cdxtail + cdy * cdytail)*(adx * bdy - ady * bdx))
    if (det >= errbound) or (-det >= errbound):
        return det

    finnow = fin1
    finother = fin2

    if (bdxtail != 0.0) or (bdytail != 0.0) or \
            (cdxtail != 0.0) or (cdytail != 0.0):
        adxadx1, adxadx0 = Square(adx, splitter)
        adyady1, adyady0 = Square(ady, splitter)
        aa[3], aa[2], aa[1], aa[0] = Two_Two_Sum(adxadx1, adxadx0,
                                               adyady1, adyady0)

    if (cdxtail != 0.0) or (cdytail != 0.0) or \
            (adxtail != 0.0) or (adytail != 0.0):
        bdxbdx1, bdxbdx0 = Square(bdx, splitter)
        bdybdy1, bdybdy0 = Square(bdy, splitter)
        bb[3], bb[2], bb[1], bb[0] = Two_Two_Sum(bdxbdx1, bdxbdx0,
                                                 bdybdy1, bdybdy0)

    if (adxtail != 0.0) or (adytail != 0.0) or \
            (bdxtail != 0.0) or (bdytail != 0.0):
        cdxcdx1, cdxcdx0 = Square(cdx, splitter)
        cdycdy1, cdycdy0 = Square(cdy, splitter)
        cc[3], cc[2], cc[1], cc[0] = Two_Two_Sum(cdxcdx1, cdxcdx0,
                                                 cdycdy1, cdycdy0)

    if adxtail != 0.0:
        axtbclen = scale_expansion_zeroelim(4, bc, adxtail, axtbc, splitter)
        temp16alen = scale_expansion_zeroelim(axtbclen, axtbc, 2.0 * adx,
                                              temp16a, splitter)

        axtcclen = scale_expansion_zeroelim(4, cc, adxtail, axtcc, splitter)
        temp16blen = scale_expansion_zeroelim(axtcclen, axtcc, bdy, temp16b,
                                              splitter)

        axtbblen = scale_expansion_zeroelim(4, bb, adxtail, axtbb, splitter)
        temp16clen = scale_expansion_zeroelim(axtbblen, axtbb, -cdy, temp16c,
                                              splitter)

        temp32alen = fast_expansion_sum_zeroelim(temp16alen, temp16a,
                                                temp16blen, temp16b, temp32a)
        temp48len = fast_expansion_sum_zeroelim(temp16clen, temp16c,
                                                temp32alen, temp32a, temp48)
        finlength = fast_expansion_sum_zeroelim(finlength, finnow, temp48len,
                                                temp48, finother)
        finswap = finnow
        finnow = finother
        finother = finswap

    if adytail != 0.0:
        aytbclen = scale_expansion_zeroelim(4, bc, adytail, aytbc, splitter)
        temp16alen = scale_expansion_zeroelim(aytbclen, aytbc, 2.0 * ady,
                                              temp16a, splitter)

        aytbblen = scale_expansion_zeroelim(4, bb, adytail, aytbb, splitter)
        temp16blen = scale_expansion_zeroelim(aytbblen, aytbb, cdx, temp16b,
                                              splitter)

        aytcclen = scale_expansion_zeroelim(4, cc, adytail, aytcc, splitter)
        temp16clen = scale_expansion_zeroelim(aytcclen, aytcc, -bdx, temp16c,
                                              splitter)

        temp32alen = fast_expansion_sum_zeroelim(temp16alen, temp16a,
                                                temp16blen, temp16b, temp32a)
        temp48len = fast_expansion_sum_zeroelim(temp16clen, temp16c,
                                                temp32alen, temp32a, temp48)
        finlength = fast_expansion_sum_zeroelim(finlength, finnow, temp48len,
                                                temp48, finother)
        finswap = finnow
        finnow = finother
        finother = finswap

    if bdxtail != 0.0:
        bxtcalen = scale_expansion_zeroelim(4, ca, bdxtail, bxtca, splitter)
        temp16alen = scale_expansion_zeroelim(bxtcalen, bxtca, 2.0 * bdx,
                                              temp16a, splitter)

        bxtaalen = scale_expansion_zeroelim(4, aa, bdxtail, bxtaa, splitter)
        temp16blen = scale_expansion_zeroelim(bxtaalen, bxtaa, cdy, temp16b,
                                              splitter)

        bxtcclen = scale_expansion_zeroelim(4, cc, bdxtail, bxtcc, splitter)
        temp16clen = scale_expansion_zeroelim(bxtcclen, bxtcc, -ady, temp16c,
                                              splitter)

        temp32alen = fast_expansion_sum_zeroelim(temp16alen, temp16a,
                                                temp16blen, temp16b, temp32a)
        temp48len = fast_expansion_sum_zeroelim(temp16clen, temp16c,
                                                temp32alen, temp32a, temp48)
        finlength = fast_expansion_sum_zeroelim(finlength, finnow, temp48len,
                                                temp48, finother)
        finswap = finnow
        finnow = finother
        finother = finswap

    if bdytail != 0.0:
        bytcalen = scale_expansion_zeroelim(4, ca, bdytail, bytca, splitter)
        temp16alen = scale_expansion_zeroelim(bytcalen, bytca, 2.0 * bdy,
                                              temp16a, splitter)

        bytcclen = scale_expansion_zeroelim(4, cc, bdytail, bytcc, splitter)
        temp16blen = scale_expansion_zeroelim(bytcclen, bytcc, adx, temp16b,
                                              splitter)

        bytaalen = scale_expansion_zeroelim(4, aa, bdytail, bytaa, splitter)
        temp16clen = scale_expansion_zeroelim(bytaalen, bytaa, -cdx, temp16c,
                                              splitter)

        temp32alen = fast_expansion_sum_zeroelim(temp16alen, temp16a,
                                                temp16blen, temp16b, temp32a)
        temp48len = fast_expansion_sum_zeroelim(temp16clen, temp16c,
                                                temp32alen, temp32a, temp48)
        finlength = fast_expansion_sum_zeroelim(finlength, finnow, temp48len,
                                                temp48, finother)
        finswap = finnow
        finnow = finother
        finother = finswap

    if cdxtail != 0.0:
        cxtablen = scale_expansion_zeroelim(4, ab, cdxtail, cxtab, splitter)
        temp16alen = scale_expansion_zeroelim(cxtablen, cxtab, 2.0 * cdx,
                                              temp16a, splitter)

        cxtbblen = scale_expansion_zeroelim(4, bb, cdxtail, cxtbb, splitter)
        temp16blen = scale_expansion_zeroelim(cxtbblen, cxtbb, ady, temp16b,
                                              splitter)

        cxtaalen = scale_expansion_zeroelim(4, aa, cdxtail, cxtaa, splitter)
        temp16clen = scale_expansion_zeroelim(cxtaalen, cxtaa, -bdy, temp16c,
                                              splitter)

        temp32alen = fast_expansion_sum_zeroelim(temp16alen, temp16a,
                                                temp16blen, temp16b, temp32a)
        temp48len = fast_expansion_sum_zeroelim(temp16clen, temp16c,
                                                temp32alen, temp32a, temp48)
        finlength = fast_expansion_sum_zeroelim(finlength, finnow, temp48len,
                                                temp48, finother)
        finswap = finnow
        finnow = finother
        finother = finswap

    if cdytail != 0.0:
        cytablen = scale_expansion_zeroelim(4, ab, cdytail, cytab, splitter)
        temp16alen = scale_expansion_zeroelim(cytablen, cytab, 2.0 * cdy,
                                              temp16a, splitter)

        cytaalen = scale_expansion_zeroelim(4, aa, cdytail, cytaa, splitter)
        temp16blen = scale_expansion_zeroelim(cytaalen, cytaa, bdx, temp16b,
                                              splitter)

        cytbblen = scale_expansion_zeroelim(4, bb, cdytail, cytbb, splitter)
        temp16clen = scale_expansion_zeroelim(cytbblen, cytbb, -adx, temp16c,
                                              splitter)

        temp32alen = fast_expansion_sum_zeroelim(temp16alen, temp16a,
                                                temp16blen, temp16b, temp32a)
        temp48len = fast_expansion_sum_zeroelim(temp16clen, temp16c,
                                                temp32alen, temp32a, temp48)
        finlength = fast_expansion_sum_zeroelim(finlength, finnow, temp48len,
                                                temp48, finother)
        finswap = finnow
        finnow = finother
        finother = finswap


    if (adxtail != 0.0) or (adytail != 0.0):
        if (bdxtail != 0.0) or (bdytail != 0.0) or \
                (cdxtail != 0.0) or (cdytail != 0.0):
            ti1, ti0 = Two_Product(bdxtail, cdy, splitter)
            tj1, tj0 = Two_Product(bdx, cdytail, splitter)
            u[3], u[2], u[1], u[0] = Two_Two_Sum(ti1, ti0, tj1, tj0)
            negate = -bdy
            ti1, ti0 = Two_Product(cdxtail, negate, splitter)
            negate = -bdytail
            tj1, tj0 = Two_Product(cdx, negate, splitter)
            v[3], v[2], v[1], v[0] = Two_Two_Sum(ti1, ti0, tj1, tj0)
            bctlen = fast_expansion_sum_zeroelim(4, u, 4, v, bct)

            ti1, ti0 = Two_Product(bdxtail, cdytail, splitter)
            tj1, tj0 = Two_Product(cdxtail, bdytail, splitter)
            bctt[3], bctt[2], bctt[1], bctt[0] = Two_Two_Diff(ti1, ti0, tj1,
                                                              tj0)
            bcttlen = 4
        else:
            bct[0] = 0.0
            bctlen = 1
            bctt[0] = 0.0
            bcttlen = 1

        if adxtail != 0.0:
            temp16alen = scale_expansion_zeroelim(axtbclen, axtbc, adxtail,
                                                  temp16a, splitter)
            axtbctlen = scale_expansion_zeroelim(bctlen, bct, adxtail, axtbct,
                                                 splitter)
            temp32alen = scale_expansion_zeroelim(axtbctlen, axtbct, 2.0 * adx,
                                                  temp32a, splitter)
            temp48len = fast_expansion_sum_zeroelim(temp16alen, temp16a,
                                                    temp32alen, temp32a,
                                                    temp48)
            finlength = fast_expansion_sum_zeroelim(finlength, finnow,
                                                    temp48len, temp48,
                                                    finother)
            finswap = finnow
            finnow = finother
            finother = finswap

            if bdytail != 0.0:
                temp8len = scale_expansion_zeroelim(4, cc, adxtail, temp8,
                                                    splitter)
                temp16alen = scale_expansion_zeroelim(temp8len, temp8, bdytail,
                                                      temp16a, splitter)
                finlength = fast_expansion_sum_zeroelim(finlength, finnow,
                                                        temp16alen, temp16a,
                                                        finother)
                finswap = finnow
                finnow = finother
                finother = finswap

            if cdytail != 0.0:
                temp8len = scale_expansion_zeroelim(4, bb, -adxtail, temp8,
                                                    splitter)
                temp16alen = scale_expansion_zeroelim(temp8len, temp8, cdytail,
                                                      temp16a, splitter)
                finlength = fast_expansion_sum_zeroelim(finlength, finnow,
                                                        temp16alen, temp16a,
                                                        finother)
                finswap = finnow
                finnow = finother
                finother = finswap

            temp32alen = scale_expansion_zeroelim(axtbctlen, axtbct, adxtail,
                                                temp32a, splitter)
            axtbcttlen = scale_expansion_zeroelim(bcttlen, bctt, adxtail,
                                                  axtbctt, splitter)
            temp16alen = scale_expansion_zeroelim(axtbcttlen, axtbctt,
                                                  2.0 * adx, temp16a, splitter)
            temp16blen = scale_expansion_zeroelim(axtbcttlen, axtbctt, adxtail,
                                                  temp16b, splitter)
            temp32blen = fast_expansion_sum_zeroelim(temp16alen, temp16a,
                                                     temp16blen, temp16b,
                                                     temp32b)
            temp64len = fast_expansion_sum_zeroelim(temp32alen, temp32a,
                                                    temp32blen, temp32b,
                                                    temp64)
            finlength = fast_expansion_sum_zeroelim(finlength, finnow,
                                                    temp64len, temp64,
                                                    finother)
            finswap = finnow
            finnow = finother
            finother = finswap

        if adytail != 0.0:
            temp16alen = scale_expansion_zeroelim(aytbclen, aytbc, adytail,
                                                  temp16a, splitter)
            aytbctlen = scale_expansion_zeroelim(bctlen, bct, adytail, aytbct,
                                                 splitter)
            temp32alen = scale_expansion_zeroelim(aytbctlen, aytbct, 2.0 * ady,
                                                  temp32a, splitter)
            temp48len = fast_expansion_sum_zeroelim(temp16alen, temp16a,
                                                    temp32alen, temp32a,
                                                    temp48)
            finlength = fast_expansion_sum_zeroelim(finlength, finnow,
                                                    temp48len, temp48,
                                                    finother)
            finswap = finnow
            finnow = finother
            finother = finswap

            temp32alen = scale_expansion_zeroelim(aytbctlen, aytbct, adytail,
                                                  temp32a, splitter)
            aytbcttlen = scale_expansion_zeroelim(bcttlen, bctt, adytail,
                                                  aytbctt, splitter)
            temp16alen = scale_expansion_zeroelim(aytbcttlen, aytbctt,
                                                  2.0 * ady, temp16a, splitter)
            temp16blen = scale_expansion_zeroelim(aytbcttlen, aytbctt, adytail,
                                                  temp16b, splitter)
            temp32blen = fast_expansion_sum_zeroelim(temp16alen, temp16a,
                                                     temp16blen, temp16b,
                                                     temp32b)
            temp64len = fast_expansion_sum_zeroelim(temp32alen, temp32a,
                                                    temp32blen, temp32b,
                                                    temp64)
            finlength = fast_expansion_sum_zeroelim(finlength, finnow,
                                                    temp64len, temp64,
                                                    finother)
            finswap = finnow
            finnow = finother
            finother = finswap

    if (bdxtail != 0.0) or (bdytail != 0.0):
        if (cdxtail != 0.0) or (cdytail != 0.0) or \
                (adxtail != 0.0) or (adytail != 0.0):
            ti1, ti0 = Two_Product(cdxtail, ady, splitter)
            tj1, tj0 = Two_Product(cdx, adytail, splitter)
            u[3], u[2], u[1], u[0] = Two_Two_Sum(ti1, ti0, tj1, tj0)
            negate = -cdy
            ti1, ti0 = Two_Product(adxtail, negate, splitter)
            negate = -cdytail
            tj1, tj0 = Two_Product(adx, negate, splitter)
            v[3], v[2], v[1], v[0] = Two_Two_Sum(ti1, ti0, tj1, tj0)
            catlen = fast_expansion_sum_zeroelim(4, u, 4, v, cat)

            ti1, ti0 = Two_Product(cdxtail, adytail, splitter)
            tj1, tj0 = Two_Product(adxtail, cdytail, splitter)
            catt[3], catt[2], catt[1], catt[0] = Two_Two_Diff(ti1, ti0, tj1,
                                                              tj0)
            cattlen = 4
        else:
            cat[0] = 0.0
            catlen = 1
            catt[0] = 0.0
            cattlen = 1

        if bdxtail != 0.0:
            temp16alen = scale_expansion_zeroelim(bxtcalen, bxtca, bdxtail,
                                                  temp16a, splitter)
            bxtcatlen = scale_expansion_zeroelim(catlen, cat, bdxtail, bxtcat,
                                                 splitter)
            temp32alen = scale_expansion_zeroelim(bxtcatlen, bxtcat, 2.0 * bdx,
                                                  temp32a, splitter)
            temp48len = fast_expansion_sum_zeroelim(temp16alen, temp16a,
                                                    temp32alen, temp32a,
                                                    temp48)
            finlength = fast_expansion_sum_zeroelim(finlength, finnow,
                                                    temp48len, temp48,
                                                    finother)
            finswap = finnow
            finnow = finother
            finother = finswap
            if cdytail != 0.0:
                temp8len = scale_expansion_zeroelim(4, aa, bdxtail, temp8,
                                                    splitter)
                temp16alen = scale_expansion_zeroelim(temp8len, temp8, cdytail,
                                                      temp16a, splitter)
                finlength = fast_expansion_sum_zeroelim(finlength, finnow,
                                                        temp16alen, temp16a,
                                                        finother)
                finswap = finnow
                finnow = finother
                finother = finswap
            if adytail != 0.0:
                temp8len = scale_expansion_zeroelim(4, cc, -bdxtail, temp8,
                                                    splitter)
                temp16alen = scale_expansion_zeroelim(temp8len, temp8, adytail,
                                                      temp16a, splitter)
                finlength = fast_expansion_sum_zeroelim(finlength, finnow,
                                                        temp16alen, temp16a,
                                                        finother)
                finswap = finnow
                finnow = finother
                finother = finswap

            temp32alen = scale_expansion_zeroelim(bxtcatlen, bxtcat, bdxtail,
                                                  temp32a, splitter)
            bxtcattlen = scale_expansion_zeroelim(cattlen, catt, bdxtail,
                                                  bxtcatt, splitter)
            temp16alen = scale_expansion_zeroelim(bxtcattlen, bxtcatt,
                                                  2.0 * bdx, temp16a, splitter)
            temp16blen = scale_expansion_zeroelim(bxtcattlen, bxtcatt, bdxtail,
                                                  temp16b, splitter)
            temp32blen = fast_expansion_sum_zeroelim(temp16alen, temp16a,
                                                     temp16blen, temp16b,
                                                     temp32b)
            temp64len = fast_expansion_sum_zeroelim(temp32alen, temp32a,
                                                    temp32blen, temp32b,
                                                    temp64)
            finlength = fast_expansion_sum_zeroelim(finlength, finnow,
                                                    temp64len, temp64,
                                                    finother)
            finswap = finnow
            finnow = finother
            finother = finswap

        if bdytail != 0.0:
            temp16alen = scale_expansion_zeroelim(bytcalen, bytca, bdytail,
                                                  temp16a, splitter)
            bytcatlen = scale_expansion_zeroelim(catlen, cat, bdytail, bytcat,
                                                 splitter)
            temp32alen = scale_expansion_zeroelim(bytcatlen, bytcat, 2.0 * bdy,
                                                  temp32a, splitter)
            temp48len = fast_expansion_sum_zeroelim(temp16alen, temp16a,
                                                    temp32alen, temp32a,
                                                    temp48)
            finlength = fast_expansion_sum_zeroelim(finlength, finnow,
                                                    temp48len, temp48,
                                                    finother)
            finswap = finnow
            finnow = finother
            finother = finswap

            temp32alen = scale_expansion_zeroelim(bytcatlen, bytcat, bdytail,
                                                  temp32a, splitter)
            bytcattlen = scale_expansion_zeroelim(cattlen, catt, bdytail,
                                                  bytcatt, splitter)
            temp16alen = scale_expansion_zeroelim(bytcattlen, bytcatt,
                                                  2.0 * bdy, temp16a, splitter)
            temp16blen = scale_expansion_zeroelim(bytcattlen, bytcatt, bdytail,
                                                  temp16b, splitter)
            temp32blen = fast_expansion_sum_zeroelim(temp16alen, temp16a,
                                                     temp16blen, temp16b,
                                                     temp32b)
            temp64len = fast_expansion_sum_zeroelim(temp32alen, temp32a,
                                                    temp32blen, temp32b,
                                                    temp64)
            finlength = fast_expansion_sum_zeroelim(finlength, finnow,
                                                    temp64len, temp64,
                                                    finother)
            finswap = finnow
            finnow = finother
            finother = finswap

    if (cdxtail != 0.0) or (cdytail != 0.0):
        if (adxtail != 0.0) or (adytail != 0.0) or \
                (bdxtail != 0.0) or (bdytail != 0.0):
            ti1, ti0 = Two_Product(adxtail, bdy, splitter)
            tj1, tj0 = Two_Product(adx, bdytail, splitter)
            u[3], u[2], u[1], u[0] = Two_Two_Sum(ti1, ti0, tj1, tj0)
            negate = -ady
            ti1, ti0 = Two_Product(bdxtail, negate, splitter)
            negate = -adytail
            tj1, tj0 = Two_Product(bdx, negate, splitter)
            v[3], v[2], v[1], v[0] = Two_Two_Sum(ti1, ti0, tj1, tj0)
            abtlen = fast_expansion_sum_zeroelim(4, u, 4, v, abt)

            ti1, ti0 = Two_Product(adxtail, bdytail, splitter)
            tj1, tj0 = Two_Product(bdxtail, adytail, splitter)
            abtt[3], abtt[2], abtt[1], abtt[0] = Two_Two_Diff(ti1, ti0, tj1,
                                                              tj0)
            abttlen = 4
        else:
            abt[0] = 0.0
            abtlen = 1
            abtt[0] = 0.0
            abttlen = 1

        if cdxtail != 0.0:
            temp16alen = scale_expansion_zeroelim(cxtablen, cxtab, cdxtail,
                                                  temp16a, splitter)
            cxtabtlen = scale_expansion_zeroelim(abtlen, abt, cdxtail, cxtabt,
                                                 splitter)
            temp32alen = scale_expansion_zeroelim(cxtabtlen, cxtabt, 2.0 * cdx,
                                                  temp32a, splitter)
            temp48len = fast_expansion_sum_zeroelim(temp16alen, temp16a,
                                                    temp32alen, temp32a,
                                                    temp48)
            finlength = fast_expansion_sum_zeroelim(finlength, finnow,
                                                    temp48len, temp48,
                                                    finother)
            finswap = finnow
            finnow = finother
            finother = finswap
      
            if adytail != 0.0:
                temp8len = scale_expansion_zeroelim(4, bb, cdxtail, temp8,
                                                    splitter)
                temp16alen = scale_expansion_zeroelim(temp8len, temp8, adytail,
                                                      temp16a, splitter)
                finlength = fast_expansion_sum_zeroelim(finlength, finnow,
                                                        temp16alen, temp16a,
                                                        finother)
                finswap = finnow
                finnow = finother
                finother = finswap

            if bdytail != 0.0:
                temp8len = scale_expansion_zeroelim(4, aa, -cdxtail, temp8,
                                                    splitter)
                temp16alen = scale_expansion_zeroelim(temp8len, temp8, bdytail,
                                                      temp16a, splitter)
                finlength = fast_expansion_sum_zeroelim(finlength, finnow,
                                                        temp16alen, temp16a,
                                                        finother)
                finswap = finnow
                finnow = finother
                finother = finswap

            temp32alen = scale_expansion_zeroelim(cxtabtlen, cxtabt, cdxtail,
                                                  temp32a, splitter)
            cxtabttlen = scale_expansion_zeroelim(abttlen, abtt, cdxtail,
                                                  cxtabtt, splitter)
            temp16alen = scale_expansion_zeroelim(cxtabttlen, cxtabtt,
                                                  2.0 * cdx, temp16a, splitter)
            temp16blen = scale_expansion_zeroelim(cxtabttlen, cxtabtt, cdxtail,
                                                  temp16b, splitter)
            temp32blen = fast_expansion_sum_zeroelim(temp16alen, temp16a,
                                                     temp16blen, temp16b,
                                                     temp32b)
            temp64len = fast_expansion_sum_zeroelim(temp32alen, temp32a,
                                                    temp32blen, temp32b,
                                                    temp64)
            finlength = fast_expansion_sum_zeroelim(finlength, finnow,
                                                    temp64len, temp64,
                                                    finother)
            finswap = finnow
            finnow = finother
            finother = finswap

        if cdytail != 0.0:
            temp16alen = scale_expansion_zeroelim(cytablen, cytab, cdytail,
                                                  temp16a, splitter)
            cytabtlen = scale_expansion_zeroelim(abtlen, abt, cdytail, cytabt,
                                                 splitter)
            temp32alen = scale_expansion_zeroelim(cytabtlen, cytabt, 2.0 * cdy,
                                                  temp32a, splitter)
            temp48len = fast_expansion_sum_zeroelim(temp16alen, temp16a,
                                                    temp32alen, temp32a,
                                                    temp48)
            finlength = fast_expansion_sum_zeroelim(finlength, finnow,
                                                    temp48len, temp48,
                                                    finother)
            finswap = finnow
            finnow = finother
            finother = finswap

            temp32alen = scale_expansion_zeroelim(cytabtlen, cytabt, cdytail,
                                                  temp32a, splitter)
            cytabttlen = scale_expansion_zeroelim(abttlen, abtt, cdytail,
                                                  cytabtt, splitter)
            temp16alen = scale_expansion_zeroelim(cytabttlen, cytabtt,
                                                  2.0 * cdy, temp16a, splitter)
            temp16blen = scale_expansion_zeroelim(cytabttlen, cytabtt, cdytail,
                                                  temp16b, splitter)
            temp32blen = fast_expansion_sum_zeroelim(temp16alen, temp16a,
                                                     temp16blen, temp16b,
                                                     temp32b)
            temp64len = fast_expansion_sum_zeroelim(temp32alen, temp32a,
                                                    temp32blen, temp32b,
                                                    temp64)
            finlength = fast_expansion_sum_zeroelim(finlength, finnow,
                                                    temp64len, temp64,
                                                    finother)
            finswap = finnow
            finnow = finother
            finother = finswap

    return finnow[finlength - 1]


@njit(cache=True)
def incircle(
        pa_x, pa_y, pb_x, pb_y, pc_x, pc_y, pd_x, pd_y, res_arr, global_arr):
    '''
    len(bc) = 4
    len(ca) = 4
    len(ab) = 4
    len(axbc) = 8
    len(axxbc) = 16
    len(aybc) = 8
    len(ayybc) = 16
    len(adet) = 32
    len(bxca) = 8
    len(bxxca) = 16
    len(byca) = 8
    len(byyca) = 16
    len(bdet) = 32
    len(cxab) = 8
    len(cxxab) = 16
    len(cyab) = 8
    len(cyyab) = 16
    len(cdet) = 32
    len(abdet) = 64
    len(fin1) = 1152
    len(fin2) = 1152
    len(aa) = 4
    len(bb) = 4
    len(cc) = 4
    len(u) = 4
    len(v) = 4
    len(temp8) = 8
    len(temp16a) = 16
    len(temp16b) = 16
    len(temp16c) = 16
    len(temp32a) = 32
    len(temp32b) = 32
    len(temp48) = 48
    len(temp64) = 64
    len(axtbb) = 8
    len(axtcc) = 8
    len(aytbb) = 8
    len(aytcc) = 8
    len(bxtaa) = 8
    len(bxtcc) = 8
    len(bytaa) = 8
    len(bytcc) = 8
    len(cxtaa) = 8
    len(cxtbb) = 8
    len(cytaa) = 8
    len(cytbb) = 8
    len(axtbc) = 8
    len(aytbc) = 8
    len(bxtca) = 8
    len(bytca) = 8
    len(cxtab) = 8
    len(cytab) = 8
    len(axtbct) = 16
    len(aytbct) = 16
    len(bxtcat) = 16
    len(bytcat) = 16
    len(cxtabt) = 16
    len(cytabt) = 16
    len(axtbctt) = 8
    len(aytbctt) = 8
    len(bxtcatt) = 8
    len(bytcatt) = 8
    len(cxtabtt) = 8
    len(cytabtt) = 8
    len(abt) = 8
    len(bct) = 8
    len(cat) = 8
    len(abtt) = 4
    len(bctt) = 4
    len(catt) = 4
    '''

    splitter = res_arr[0]
    resulterrbound = res_arr[1]
    iccerrboundA = res_arr[5]
    iccerrboundB = res_arr[6]
    iccerrboundC = res_arr[7]
    static_filter_i2d = res_arr[9]

    adx = pa_x - pd_x
    bdx = pb_x - pd_x
    cdx = pc_x - pd_x
    ady = pa_y - pd_y
    bdy = pb_y - pd_y
    cdy = pc_y - pd_y

    bdxcdy = bdx * cdy
    cdxbdy = cdx * bdy
    alift = adx * adx + ady * ady

    cdxady = cdx * ady
    adxcdy = adx * cdy
    blift = bdx * bdx + bdy * bdy

    adxbdy = adx * bdy
    bdxady = bdx * ady
    clift = cdx * cdx + cdy * cdy

    det = alift * (bdxcdy - cdxbdy) + \
          blift * (cdxady - adxcdy) + \
          clift * (adxbdy - bdxady)
    if np.abs(det) > static_filter_i2d:
        return det

    permanent = (np.abs(bdxcdy) + np.abs(cdxbdy)) * alift + \
                (np.abs(cdxady) + np.abs(adxcdy)) * blift + \
                (np.abs(adxbdy) + np.abs(bdxady)) * clift
    errbound = iccerrboundA * permanent
    if np.abs(det) > errbound:
        return det

    return incircleadapt(
        pa_x, pa_y, pb_x, pb_y, pc_x, pc_y, pd_x, pd_y, permanent, global_arr,
        splitter, iccerrboundB, iccerrboundC, resulterrbound)


#*****************************************************************************#
#                                                                             #
#   insphere()   Adaptive exact 3D insphere test.  Robust.                    #
#                                                                             #
#                Return a positive value if the point pe lies inside the      #
#                sphere passing through pa, pb, pc, and pd; a negative value  #
#                if it lies outside; and zero if the five points are          #
#                cospherical.  The points pa, pb, pc, and pd must be ordered  #
#                so that they have a positive orientation (as defined by      #
#                orient3d()), or the sign of the result will be reversed.     #
#                                                                             #
#   This uses exact arithmetic to ensure a correct answer.  The               #
#   result returned is the determinant of a matrix.  In insphere() only,      #
#   this determinant is computed adaptively, in the sense that exact          #
#   arithmetic is used only to the degree it is needed to ensure that the     #
#   returned value has the correct sign.  Hence, insphere() is usually quite  #
#   fast, but will run more slowly when the input points are cospherical or   #
#   nearly so.                                                                #
#                                                                             #
#*****************************************************************************#


@njit(cache=True)
def insphereexact(pa, pb, pc, pd, pe, ab, bc, cd, de, ea, ac, bd, ce, da, eb,
                  temp8a, temp8b, temp16, abc, bcd, cde, dea, eab, abd, bce,
                  cda, deb, eac, temp48a, temp48b, abcd, bcde, cdea, deab,
                  eabc, temp192, det384x, det384y, det384z, detxy, adet, bdet,
                  cdet, ddet, edet, abdet, cddet, cdedet, deter, splitter):
    '''
    len(ab) = 4
    len(bc) = 4
    len(cd) = 4
    len(de) = 4
    len(ea) = 4
    len(ac) = 4
    len(bd) = 4
    len(ce) = 4
    len(da) = 4
    len(eb) = 4
    len(temp8a) = 8
    len(temp8b) = 8
    len(temp16) = 16
    len(abc) = 24
    len(bcd) = 24
    len(cde) = 24
    len(dea) = 24
    len(eab) = 24
    len(abd) = 24
    len(bce) = 24
    len(cda) = 24
    len(deb) = 24
    len(eac) = 24
    len(temp48a) = 48
    len(temp48b) = 48
    len(abcd) = 96
    len(bcde) = 96
    len(cdea) = 96
    len(deab) = 96
    len(eabc) = 96
    len(temp192) = 192
    len(det384x) = 384
    len(det384y) = 384
    len(det384z) = 384
    len(detxy) = 768
    len(adet) = 1152
    len(bdet) = 1152
    len(cdet) = 1152
    len(ddet) = 1152
    len(edet) = 1152
    len(abdet) = 2304
    len(cddet) = 2304
    len(cdedet) = 3456
    len(deter) = 5760
    '''

    axby1, axby0 = Two_Product(pa[0], pb[1], splitter)
    bxay1, bxay0 = Two_Product(pb[0], pa[1], splitter)
    ab[3], ab[2], ab[1], ab[0] = Two_Two_Diff(axby1, axby0, bxay1, bxay0)

    bxcy1, bxcy0 = Two_Product(pb[0], pc[1], splitter)
    cxby1, cxby0 = Two_Product(pc[0], pb[1], splitter)
    bc[3], bc[2], bc[1], bc[0] = Two_Two_Diff(bxcy1, bxcy0, cxby1, cxby0)

    cxdy1, cxdy0 = Two_Product(pc[0], pd[1], splitter)
    dxcy1, dxcy0 = Two_Product(pd[0], pc[1], splitter)
    cd[3], cd[2], cd[1], cd[0] = Two_Two_Diff(cxdy1, cxdy0, dxcy1, dxcy0)

    dxey1, dxey0 = Two_Product(pd[0], pe[1], splitter)
    exdy1, exdy0 = Two_Product(pe[0], pd[1], splitter)
    de[3], de[2], de[1], de[0] = Two_Two_Diff(dxey1, dxey0, exdy1, exdy0)

    exay1, exay0 = Two_Product(pe[0], pa[1], splitter)
    axey1, axey0 = Two_Product(pa[0], pe[1], splitter)
    ea[3], ea[2], ea[1], ea[0] = Two_Two_Diff(exay1, exay0, axey1, axey0)

    axcy1, axcy0 = Two_Product(pa[0], pc[1], splitter)
    cxay1, cxay0 = Two_Product(pc[0], pa[1], splitter)
    ac[3], ac[2], ac[1], ac[0] = Two_Two_Diff(axcy1, axcy0, cxay1, cxay0)

    bxdy1, bxdy0 = Two_Product(pb[0], pd[1], splitter)
    dxby1, dxby0 = Two_Product(pd[0], pb[1], splitter)
    bd[3], bd[2], bd[1], bd[0] = Two_Two_Diff(bxdy1, bxdy0, dxby1, dxby0)

    cxey1, cxey0 = Two_Product(pc[0], pe[1], splitter)
    excy1, excy0 = Two_Product(pe[0], pc[1], splitter)
    ce[3], ce[2], ce[1], ce[0] = Two_Two_Diff(cxey1, cxey0, excy1, excy0)

    dxay1, dxay0 = Two_Product(pd[0], pa[1], splitter)
    axdy1, axdy0 = Two_Product(pa[0], pd[1], splitter)
    da[3], da[2], da[1], da[0] = Two_Two_Diff(dxay1, dxay0, axdy1, axdy0)

    exby1, exby0 = Two_Product(pe[0], pb[1], splitter)
    bxey1, bxey0 = Two_Product(pb[0], pe[1], splitter)
    eb[3], eb[2], eb[1], eb[0] = Two_Two_Diff(exby1, exby0, bxey1, bxey0)

    temp8alen = scale_expansion_zeroelim(4, bc, pa[2], temp8a, splitter)
    temp8blen = scale_expansion_zeroelim(4, ac, -pb[2], temp8b, splitter)
    temp16len = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp8blen,
                                            temp8b, temp16)
    temp8alen = scale_expansion_zeroelim(4, ab, pc[2], temp8a, splitter)
    abclen = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp16len, temp16,
                                         abc)

    temp8alen = scale_expansion_zeroelim(4, cd, pb[2], temp8a, splitter)
    temp8blen = scale_expansion_zeroelim(4, bd, -pc[2], temp8b, splitter)
    temp16len = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp8blen,
                                            temp8b, temp16)
    temp8alen = scale_expansion_zeroelim(4, bc, pd[2], temp8a, splitter)
    bcdlen = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp16len, temp16,
                                         bcd)

    temp8alen = scale_expansion_zeroelim(4, de, pc[2], temp8a, splitter)
    temp8blen = scale_expansion_zeroelim(4, ce, -pd[2], temp8b, splitter)
    temp16len = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp8blen,
                                            temp8b, temp16)
    temp8alen = scale_expansion_zeroelim(4, cd, pe[2], temp8a, splitter)
    cdelen = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp16len, temp16,
                                         cde)

    temp8alen = scale_expansion_zeroelim(4, ea, pd[2], temp8a, splitter)
    temp8blen = scale_expansion_zeroelim(4, da, -pe[2], temp8b, splitter)
    temp16len = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp8blen,
                                            temp8b, temp16)
    temp8alen = scale_expansion_zeroelim(4, de, pa[2], temp8a, splitter)
    dealen = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp16len, temp16,
                                         dea)

    temp8alen = scale_expansion_zeroelim(4, ab, pe[2], temp8a, splitter)
    temp8blen = scale_expansion_zeroelim(4, eb, -pa[2], temp8b, splitter)
    temp16len = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp8blen,
                                            temp8b, temp16)
    temp8alen = scale_expansion_zeroelim(4, ea, pb[2], temp8a, splitter)
    eablen = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp16len, temp16,
                                         eab)

    temp8alen = scale_expansion_zeroelim(4, bd, pa[2], temp8a, splitter)
    temp8blen = scale_expansion_zeroelim(4, da, pb[2], temp8b, splitter)
    temp16len = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp8blen,
                                            temp8b, temp16)
    temp8alen = scale_expansion_zeroelim(4, ab, pd[2], temp8a, splitter)
    abdlen = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp16len, temp16,
                                         abd)

    temp8alen = scale_expansion_zeroelim(4, ce, pb[2], temp8a, splitter)
    temp8blen = scale_expansion_zeroelim(4, eb, pc[2], temp8b, splitter)
    temp16len = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp8blen,
                                            temp8b, temp16)
    temp8alen = scale_expansion_zeroelim(4, bc, pe[2], temp8a, splitter)
    bcelen = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp16len, temp16,
                                         bce)

    temp8alen = scale_expansion_zeroelim(4, da, pc[2], temp8a, splitter)
    temp8blen = scale_expansion_zeroelim(4, ac, pd[2], temp8b, splitter)
    temp16len = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp8blen,
                                            temp8b, temp16)
    temp8alen = scale_expansion_zeroelim(4, cd, pa[2], temp8a, splitter)
    cdalen = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp16len, temp16,
                                         cda)

    temp8alen = scale_expansion_zeroelim(4, eb, pd[2], temp8a, splitter)
    temp8blen = scale_expansion_zeroelim(4, bd, pe[2], temp8b, splitter)
    temp16len = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp8blen,
                                            temp8b, temp16)
    temp8alen = scale_expansion_zeroelim(4, de, pb[2], temp8a, splitter)
    deblen = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp16len, temp16,
                                         deb)

    temp8alen = scale_expansion_zeroelim(4, ac, pe[2], temp8a, splitter)
    temp8blen = scale_expansion_zeroelim(4, ce, pa[2], temp8b, splitter)
    temp16len = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp8blen,
                                            temp8b, temp16)
    temp8alen = scale_expansion_zeroelim(4, ea, pc[2], temp8a, splitter)
    eaclen = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp16len, temp16,
                                         eac)

    temp48alen = fast_expansion_sum_zeroelim(cdelen, cde, bcelen, bce, temp48a)
    temp48blen = fast_expansion_sum_zeroelim(deblen, deb, bcdlen, bcd, temp48b)
    for i in range(0, temp48blen):
        temp48b[i] = -temp48b[i]

    bcdelen = fast_expansion_sum_zeroelim(temp48alen, temp48a, temp48blen,
                                          temp48b, bcde)
    xlen = scale_expansion_zeroelim(bcdelen, bcde, pa[0], temp192, splitter)
    xlen = scale_expansion_zeroelim(xlen, temp192, pa[0], det384x, splitter)
    ylen = scale_expansion_zeroelim(bcdelen, bcde, pa[1], temp192, splitter)
    ylen = scale_expansion_zeroelim(ylen, temp192, pa[1], det384y, splitter)
    zlen = scale_expansion_zeroelim(bcdelen, bcde, pa[2], temp192, splitter)
    zlen = scale_expansion_zeroelim(zlen, temp192, pa[2], det384z, splitter)
    xylen = fast_expansion_sum_zeroelim(xlen, det384x, ylen, det384y, detxy)
    alen = fast_expansion_sum_zeroelim(xylen, detxy, zlen, det384z, adet)

    temp48alen = fast_expansion_sum_zeroelim(dealen, dea, cdalen, cda, temp48a)
    temp48blen = fast_expansion_sum_zeroelim(eaclen, eac, cdelen, cde, temp48b)
    for i in range(0, temp48blen):
        temp48b[i] = -temp48b[i]

    cdealen = fast_expansion_sum_zeroelim(temp48alen, temp48a, temp48blen,
                                          temp48b, cdea)
    xlen = scale_expansion_zeroelim(cdealen, cdea, pb[0], temp192, splitter)
    xlen = scale_expansion_zeroelim(xlen, temp192, pb[0], det384x, splitter)
    ylen = scale_expansion_zeroelim(cdealen, cdea, pb[1], temp192, splitter)
    ylen = scale_expansion_zeroelim(ylen, temp192, pb[1], det384y, splitter)
    zlen = scale_expansion_zeroelim(cdealen, cdea, pb[2], temp192, splitter)
    zlen = scale_expansion_zeroelim(zlen, temp192, pb[2], det384z, splitter)
    xylen = fast_expansion_sum_zeroelim(xlen, det384x, ylen, det384y, detxy)
    blen = fast_expansion_sum_zeroelim(xylen, detxy, zlen, det384z, bdet)

    temp48alen = fast_expansion_sum_zeroelim(eablen, eab, deblen, deb, temp48a)
    temp48blen = fast_expansion_sum_zeroelim(abdlen, abd, dealen, dea, temp48b)
    for i in range(0, temp48blen):
        temp48b[i] = -temp48b[i]

    deablen = fast_expansion_sum_zeroelim(temp48alen, temp48a, temp48blen,
                                          temp48b, deab)
    xlen = scale_expansion_zeroelim(deablen, deab, pc[0], temp192, splitter)
    xlen = scale_expansion_zeroelim(xlen, temp192, pc[0], det384x, splitter)
    ylen = scale_expansion_zeroelim(deablen, deab, pc[1], temp192, splitter)
    ylen = scale_expansion_zeroelim(ylen, temp192, pc[1], det384y, splitter)
    zlen = scale_expansion_zeroelim(deablen, deab, pc[2], temp192, splitter)
    zlen = scale_expansion_zeroelim(zlen, temp192, pc[2], det384z, splitter)
    xylen = fast_expansion_sum_zeroelim(xlen, det384x, ylen, det384y, detxy)
    clen = fast_expansion_sum_zeroelim(xylen, detxy, zlen, det384z, cdet)

    temp48alen = fast_expansion_sum_zeroelim(abclen, abc, eaclen, eac, temp48a)
    temp48blen = fast_expansion_sum_zeroelim(bcelen, bce, eablen, eab, temp48b)
    for i in range(0, temp48blen):
        temp48b[i] = -temp48b[i]

    eabclen = fast_expansion_sum_zeroelim(temp48alen, temp48a, temp48blen,
                                          temp48b, eabc)
    xlen = scale_expansion_zeroelim(eabclen, eabc, pd[0], temp192, splitter)
    xlen = scale_expansion_zeroelim(xlen, temp192, pd[0], det384x, splitter)
    ylen = scale_expansion_zeroelim(eabclen, eabc, pd[1], temp192, splitter)
    ylen = scale_expansion_zeroelim(ylen, temp192, pd[1], det384y, splitter)
    zlen = scale_expansion_zeroelim(eabclen, eabc, pd[2], temp192, splitter)
    zlen = scale_expansion_zeroelim(zlen, temp192, pd[2], det384z, splitter)
    xylen = fast_expansion_sum_zeroelim(xlen, det384x, ylen, det384y, detxy)
    dlen = fast_expansion_sum_zeroelim(xylen, detxy, zlen, det384z, ddet)

    temp48alen = fast_expansion_sum_zeroelim(bcdlen, bcd, abdlen, abd, temp48a)
    temp48blen = fast_expansion_sum_zeroelim(cdalen, cda, abclen, abc, temp48b)
    for i in range(0, temp48blen):
        temp48b[i] = -temp48b[i]

    abcdlen = fast_expansion_sum_zeroelim(temp48alen, temp48a, temp48blen,
                                          temp48b, abcd)
    xlen = scale_expansion_zeroelim(abcdlen, abcd, pe[0], temp192, splitter)
    xlen = scale_expansion_zeroelim(xlen, temp192, pe[0], det384x, splitter)
    ylen = scale_expansion_zeroelim(abcdlen, abcd, pe[1], temp192, splitter)
    ylen = scale_expansion_zeroelim(ylen, temp192, pe[1], det384y, splitter)
    zlen = scale_expansion_zeroelim(abcdlen, abcd, pe[2], temp192, splitter)
    zlen = scale_expansion_zeroelim(zlen, temp192, pe[2], det384z, splitter)
    xylen = fast_expansion_sum_zeroelim(xlen, det384x, ylen, det384y, detxy)
    elen = fast_expansion_sum_zeroelim(xylen, detxy, zlen, det384z, edet)

    ablen = fast_expansion_sum_zeroelim(alen, adet, blen, bdet, abdet)
    cdlen = fast_expansion_sum_zeroelim(clen, cdet, dlen, ddet, cddet)
    cdelen = fast_expansion_sum_zeroelim(cdlen, cddet, elen, edet, cdedet)
    deterlen = fast_expansion_sum_zeroelim(ablen, abdet, cdelen, cdedet, deter)

    return deter[deterlen - 1]


@njit(cache=True)
def insphereadapt(pa, pb, pc, pd, pe, permanent, ab, bc, cd, de, ea, ac, bd,
                  ce, da, eb, temp8a, temp8b, temp8c, temp16, temp24, temp48,
                  xdet, ydet, zdet, xydet, adet, bdet, cdet, ddet, abdet,
                  cddet, fin1, abc, bcd, cde, dea, eab, abd, bce, cda, deb,
                  eac, temp48a, temp48b, abcd, bcde, cdea, deab, eabc, temp192,
                  det384x, det384y, det384z, detxy, adet1152, bdet1152,
                  cdet1152, ddet1152, edet1152, abdet2304, cddet2304, cdedet,
                  deter, splitter, isperrboundB, isperrboundC, resulterrbound):
    '''
    len(ab) = 4
    len(bc) = 4
    len(cd) = 4
    len(de) = 4
    len(ea) = 4
    len(ac) = 4
    len(bd) = 4
    len(ce) = 4
    len(da) = 4
    len(eb) = 4
    len(temp8a) = 8
    len(temp8b) = 8
    len(temp8c) = 8
    len(temp16) = 16
    len(temp24) = 24
    len(temp48) = 48
    len(xdet) = 96
    len(ydet) = 96
    len(zdet) = 96
    len(ydet) = 192
    len(adet) = 288
    len(bdet) = 288
    len(cdet) = 288
    len(ddet) = 288
    len(abdet) = 576
    len(cddet) = 576
    len(fin1) = 1152
    len(abc) = 24
    len(bcd) = 24
    len(cde) = 24
    len(dea) = 24
    len(eab) = 24
    len(abd) = 24
    len(bce) = 24
    len(cda) = 24
    len(deb) = 24
    len(eac) = 24
    len(temp48a) = 48
    len(temp48b) = 48
    len(abcd) = 96
    len(bcde) = 96
    len(cdea) = 96
    len(deab) = 96
    len(eabc) = 96
    len(temp192) = 192
    len(det384x) = 384
    len(det384y) = 384
    len(det384z) = 384
    len(detxy) = 768
    len(adet1152) = 1152
    len(bdet1152) = 1152
    len(cdet1152) = 1152
    len(ddet1152) = 1152
    len(edet1152) = 1152
    len(abdet2304) = 2304
    len(cddet2304) = 2304
    len(cdedet) = 3456
    len(deter) = 5760
    '''

    aex = pa[0] - pe[0]
    bex = pb[0] - pe[0]
    cex = pc[0] - pe[0]
    dex = pd[0] - pe[0]
    aey = pa[1] - pe[1]
    bey = pb[1] - pe[1]
    cey = pc[1] - pe[1]
    dey = pd[1] - pe[1]
    aez = pa[2] - pe[2]
    bez = pb[2] - pe[2]
    cez = pc[2] - pe[2]
    dez = pd[2] - pe[2]

    aexbey1, aexbey0 = Two_Product(aex, bey, splitter)
    bexaey1, bexaey0 = Two_Product(bex, aey, splitter)
    ab[3], ab[2], ab[1], ab[0] = Two_Two_Diff(aexbey1, aexbey0, bexaey1,
                                              bexaey0)

    bexcey1, bexcey0 = Two_Product(bex, cey, splitter)
    cexbey1, cexbey0 = Two_Product(cex, bey, splitter)
    bc[3], bc[2], bc[1], bc[0] = Two_Two_Diff(bexcey1, bexcey0, cexbey1,
                                              cexbey0)

    cexdey1, cexdey0 = Two_Product(cex, dey, splitter)
    dexcey1, dexcey0 = Two_Product(dex, cey, splitter)
    cd[3], cd[2], cd[1], cd[0] = Two_Two_Diff(cexdey1, cexdey0, dexcey1,
                                              dexcey0)

    dexaey1, dexaey0 = Two_Product(dex, aey, splitter)
    aexdey1, aexdey0 = Two_Product(aex, dey, splitter)
    da[3], da[2], da[1], da[0] = Two_Two_Diff(dexaey1, dexaey0, aexdey1,
                                              aexdey0)

    aexcey1, aexcey0 = Two_Product(aex, cey, splitter)
    cexaey1, cexaey0 = Two_Product(cex, aey, splitter)
    ac[3], ac[2], ac[1], ac[0] = Two_Two_Diff(aexcey1, aexcey0, cexaey1,
                                              cexaey0)

    bexdey1, bexdey0 = Two_Product(bex, dey, splitter)
    dexbey1, dexbey0 = Two_Product(dex, bey, splitter)
    bd[3], bd[2], bd[1], bd[0] = Two_Two_Diff(bexdey1, bexdey0, dexbey1,
                                              dexbey0)

    temp8alen = scale_expansion_zeroelim(4, cd, bez, temp8a, splitter)
    temp8blen = scale_expansion_zeroelim(4, bd, -cez, temp8b, splitter)
    temp8clen = scale_expansion_zeroelim(4, bc, dez, temp8c, splitter)
    temp16len = fast_expansion_sum_zeroelim(temp8alen, temp8a,
                                            temp8blen, temp8b, temp16)
    temp24len = fast_expansion_sum_zeroelim(temp8clen, temp8c,
                                            temp16len, temp16, temp24)
    temp48len = scale_expansion_zeroelim(temp24len, temp24, aex, temp48,
                                         splitter)
    xlen = scale_expansion_zeroelim(temp48len, temp48, -aex, xdet, splitter)
    temp48len = scale_expansion_zeroelim(temp24len, temp24, aey, temp48,
                                         splitter)
    ylen = scale_expansion_zeroelim(temp48len, temp48, -aey, ydet, splitter)
    temp48len = scale_expansion_zeroelim(temp24len, temp24, aez, temp48,
                                         splitter)
    zlen = scale_expansion_zeroelim(temp48len, temp48, -aez, zdet, splitter)
    xylen = fast_expansion_sum_zeroelim(xlen, xdet, ylen, ydet, xydet)
    alen = fast_expansion_sum_zeroelim(xylen, xydet, zlen, zdet, adet)

    temp8alen = scale_expansion_zeroelim(4, da, cez, temp8a, splitter)
    temp8blen = scale_expansion_zeroelim(4, ac, dez, temp8b, splitter)
    temp8clen = scale_expansion_zeroelim(4, cd, aez, temp8c, splitter)
    temp16len = fast_expansion_sum_zeroelim(temp8alen, temp8a,
                                            temp8blen, temp8b, temp16)
    temp24len = fast_expansion_sum_zeroelim(temp8clen, temp8c,
                                            temp16len, temp16, temp24)
    temp48len = scale_expansion_zeroelim(temp24len, temp24, bex, temp48,
                                         splitter)
    xlen = scale_expansion_zeroelim(temp48len, temp48, bex, xdet, splitter)
    temp48len = scale_expansion_zeroelim(temp24len, temp24, bey, temp48,
                                         splitter)
    ylen = scale_expansion_zeroelim(temp48len, temp48, bey, ydet, splitter)
    temp48len = scale_expansion_zeroelim(temp24len, temp24, bez, temp48,
                                         splitter)
    zlen = scale_expansion_zeroelim(temp48len, temp48, bez, zdet, splitter)
    xylen = fast_expansion_sum_zeroelim(xlen, xdet, ylen, ydet, xydet)
    blen = fast_expansion_sum_zeroelim(xylen, xydet, zlen, zdet, bdet)

    temp8alen = scale_expansion_zeroelim(4, ab, dez, temp8a, splitter)
    temp8blen = scale_expansion_zeroelim(4, bd, aez, temp8b, splitter)
    temp8clen = scale_expansion_zeroelim(4, da, bez, temp8c, splitter)
    temp16len = fast_expansion_sum_zeroelim(temp8alen, temp8a,
                                            temp8blen, temp8b, temp16)
    temp24len = fast_expansion_sum_zeroelim(temp8clen, temp8c,
                                            temp16len, temp16, temp24)
    temp48len = scale_expansion_zeroelim(temp24len, temp24, cex, temp48,
                                         splitter)
    xlen = scale_expansion_zeroelim(temp48len, temp48, -cex, xdet, splitter)
    temp48len = scale_expansion_zeroelim(temp24len, temp24, cey, temp48,
                                         splitter)
    ylen = scale_expansion_zeroelim(temp48len, temp48, -cey, ydet, splitter)
    temp48len = scale_expansion_zeroelim(temp24len, temp24, cez, temp48,
                                         splitter)
    zlen = scale_expansion_zeroelim(temp48len, temp48, -cez, zdet, splitter)
    xylen = fast_expansion_sum_zeroelim(xlen, xdet, ylen, ydet, xydet)
    clen = fast_expansion_sum_zeroelim(xylen, xydet, zlen, zdet, cdet)

    temp8alen = scale_expansion_zeroelim(4, bc, aez, temp8a, splitter)
    temp8blen = scale_expansion_zeroelim(4, ac, -bez, temp8b, splitter)
    temp8clen = scale_expansion_zeroelim(4, ab, cez, temp8c, splitter)
    temp16len = fast_expansion_sum_zeroelim(temp8alen, temp8a,
                                            temp8blen, temp8b, temp16)
    temp24len = fast_expansion_sum_zeroelim(temp8clen, temp8c,
                                            temp16len, temp16, temp24)
    temp48len = scale_expansion_zeroelim(temp24len, temp24, dex, temp48,
                                         splitter)
    xlen = scale_expansion_zeroelim(temp48len, temp48, dex, xdet, splitter)
    temp48len = scale_expansion_zeroelim(temp24len, temp24, dey, temp48,
                                         splitter)
    ylen = scale_expansion_zeroelim(temp48len, temp48, dey, ydet, splitter)
    temp48len = scale_expansion_zeroelim(temp24len, temp24, dez, temp48,
                                         splitter)
    zlen = scale_expansion_zeroelim(temp48len, temp48, dez, zdet, splitter)
    xylen = fast_expansion_sum_zeroelim(xlen, xdet, ylen, ydet, xydet)
    dlen = fast_expansion_sum_zeroelim(xylen, xydet, zlen, zdet, ddet)

    ablen = fast_expansion_sum_zeroelim(alen, adet, blen, bdet, abdet)
    cdlen = fast_expansion_sum_zeroelim(clen, cdet, dlen, ddet, cddet)
    finlength = fast_expansion_sum_zeroelim(ablen, abdet, cdlen, cddet, fin1)

    det = estimate(finlength, fin1)
    errbound = isperrboundB * permanent
    if (det >= errbound) or (-det >= errbound):
        return det

    aextail = Two_Diff_Tail(pa[0], pe[0], aex)
    aeytail = Two_Diff_Tail(pa[1], pe[1], aey)
    aeztail = Two_Diff_Tail(pa[2], pe[2], aez)
    bextail = Two_Diff_Tail(pb[0], pe[0], bex)
    beytail = Two_Diff_Tail(pb[1], pe[1], bey)
    beztail = Two_Diff_Tail(pb[2], pe[2], bez)
    cextail = Two_Diff_Tail(pc[0], pe[0], cex)
    ceytail = Two_Diff_Tail(pc[1], pe[1], cey)
    ceztail = Two_Diff_Tail(pc[2], pe[2], cez)
    dextail = Two_Diff_Tail(pd[0], pe[0], dex)
    deytail = Two_Diff_Tail(pd[1], pe[1], dey)
    deztail = Two_Diff_Tail(pd[2], pe[2], dez)
    if (aextail == 0.0) and (aeytail == 0.0) and (aeztail == 0.0) and \
            (bextail == 0.0) and (beytail == 0.0) and (beztail == 0.0) and \
            (cextail == 0.0) and (ceytail == 0.0) and (ceztail == 0.0) and \
            (dextail == 0.0) and (deytail == 0.0) and (deztail == 0.0):
        return det


    errbound = isperrboundC * permanent + resulterrbound * np.abs(det);
    abeps = (aex * beytail + bey * aextail) - \
            (aey * bextail + bex * aeytail)
    bceps = (bex * ceytail + cey * bextail) - \
            (bey * cextail + cex * beytail)
    cdeps = (cex * deytail + dey * cextail) - \
            (cey * dextail + dex * ceytail)
    daeps = (dex * aeytail + aey * dextail) - \
            (dey * aextail + aex * deytail)
    aceps = (aex * ceytail + cey * aextail) - \
            (aey * cextail + cex * aeytail)
    bdeps = (bex * deytail + dey * bextail) - \
            (bey * dextail + dex * beytail)
    det += (((bex * bex + bey * bey + bez * bez)
             * ((cez * daeps + dez * aceps + aez * cdeps)
                + (ceztail * da3 + deztail * ac3 + aeztail * cd3))
             + (dex * dex + dey * dey + dez * dez)
             * ((aez * bceps - bez * aceps + cez * abeps)
                + (aeztail * bc3 - beztail * ac3 + ceztail * ab3)))
            - ((aex * aex + aey * aey + aez * aez)
             * ((bez * cdeps - cez * bdeps + dez * bceps)
                + (beztail * cd3 - ceztail * bd3 + deztail * bc3))
             + (cex * cex + cey * cey + cez * cez)
             * ((dez * abeps + aez * bdeps + bez * daeps)
                + (deztail * ab3 + aeztail * bd3 + beztail * da3)))) \
         + 2.0 * (((bex * bextail + bey * beytail + bez * beztail)
                   * (cez * da3 + dez * ac3 + aez * cd3)
                   + (dex * dextail + dey * deytail + dez * deztail)
                   * (aez * bc3 - bez * ac3 + cez * ab3))
                  - ((aex * aextail + aey * aeytail + aez * aeztail)
                   * (bez * cd3 - cez * bd3 + dez * bc3)
                   + (cex * cextail + cey * ceytail + cez * ceztail)
                   * (dez * ab3 + aez * bd3 + bez * da3)))
    if (det >= errbound) or (-det >= errbound):
        return det

    return insphereexact(pa, pb, pc, pd, pe, ab, bc, cd, de, ea, ac, bd, ce,
                         da, eb, temp8a, temp8b, temp16, abc, bcd, cde, dea,
                         eab, abd, bce, cda, deb, eac, temp48a, temp48b, abcd,
                         bcde, cdea, deab, eabc, temp192, det384x, det384y,
                         det384z, detxy, adet1152, bdet1152, cdet1152,
                         ddet1152, edet1152, abdet2304, cddet2304, cdedet,
                         deter, splitter)


@njit(cache=True)
def insphere(pa, pb, pc, pd, pe, ab, bc, cd, de, ea, ac, bd, ce, da, eb,
             temp8a, temp8b, temp8c, temp16, temp24, temp48, xdet, ydet, zdet,
             xydet, adet, bdet, cdet, ddet, abdet, cddet, fin1, abc, bcd, cde,
             dea, eab, abd, bce, cda, deb, eac, temp48a, temp48b, abcd, bcde,
             cdea, deab, eabc, temp192, det384x, det384y, det384z, detxy,
             adet1152, bdet1152, cdet1152, ddet1152, edet1152, abdet2304,
             cddet2304, cdedet, deter, splitter, isperrboundA, isperrboundB,
             isperrboundC, resulterrbound):

    aex = pa[0] - pe[0]
    bex = pb[0] - pe[0]
    cex = pc[0] - pe[0]
    dex = pd[0] - pe[0]
    aey = pa[1] - pe[1]
    bey = pb[1] - pe[1]
    cey = pc[1] - pe[1]
    dey = pd[1] - pe[1]
    aez = pa[2] - pe[2]
    bez = pb[2] - pe[2]
    cez = pc[2] - pe[2]
    dez = pd[2] - pe[2]

    aexbey = aex * bey
    bexaey = bex * aey
    ab = aexbey - bexaey
    bexcey = bex * cey
    cexbey = cex * bey
    bc = bexcey - cexbey
    cexdey = cex * dey
    dexcey = dex * cey
    cd = cexdey - dexcey
    dexaey = dex * aey
    aexdey = aex * dey
    da = dexaey - aexdey

    aexcey = aex * cey
    cexaey = cex * aey
    ac = aexcey - cexaey
    bexdey = bex * dey
    dexbey = dex * bey
    bd = bexdey - dexbey

    abc = aez * bc - bez * ac + cez * ab
    bcd = bez * cd - cez * bd + dez * bc
    cda = cez * da + dez * ac + aez * cd
    dab = dez * ab + aez * bd + bez * da

    alift = aex * aex + aey * aey + aez * aez
    blift = bex * bex + bey * bey + bez * bez
    clift = cex * cex + cey * cey + cez * cez
    dlift = dex * dex + dey * dey + dez * dez

    det = (dlift * abc - clift * dab) + (blift * cda - alift * bcd)

    aezplus = np.abs(aez)
    bezplus = np.abs(bez)
    cezplus = np.abs(cez)
    dezplus = np.abs(dez)
    aexbeyplus = np.abs(aexbey)
    bexaeyplus = np.abs(bexaey)
    bexceyplus = np.abs(bexcey)
    cexbeyplus = np.abs(cexbey)
    cexdeyplus = np.abs(cexdey)
    dexceyplus = np.abs(dexcey)
    dexaeyplus = np.abs(dexaey)
    aexdeyplus = np.abs(aexdey)
    aexceyplus = np.abs(aexcey)
    cexaeyplus = np.abs(cexaey)
    bexdeyplus = np.abs(bexdey)
    dexbeyplus = np.abs(dexbey)
    permanent = ((cexdeyplus + dexceyplus) * bezplus
                 + (dexbeyplus + bexdeyplus) * cezplus
                 + (bexceyplus + cexbeyplus) * dezplus) \
              * alift \
              + ((dexaeyplus + aexdeyplus) * cezplus
                 + (aexceyplus + cexaeyplus) * dezplus
                 + (cexdeyplus + dexceyplus) * aezplus) \
              * blift \
              + ((aexbeyplus + bexaeyplus) * dezplus
                 + (bexdeyplus + dexbeyplus) * aezplus
                 + (dexaeyplus + aexdeyplus) * bezplus) \
              * clift \
              + ((bexceyplus + cexbeyplus) * aezplus
                 + (cexaeyplus + aexceyplus) * bezplus
                 + (aexbeyplus + bexaeyplus) * cezplus) \
              * dlift
    errbound = isperrboundA * permanent
    if (det > errbound) or (-det > errbound):
        return det

    return insphereadapt(pa, pb, pc, pd, pe, permanent, ab, bc, cd, de, ea, ac,
                         bd, ce, da, eb, temp8a, temp8b, temp8c, temp16,
                         temp24, temp48, xdet, ydet, zdet, xydet, adet, bdet,
                         cdet, ddet, abdet, cddet, fin1, abc, bcd, cde, dea,
                         eab, abd, bce, cda, deb, eac, temp48a, temp48b, abcd,
                         bcde, cdea, deab, eabc, temp192, det384x, det384y,
                         det384z, detxy, adet1152, bdet1152, cdet1152,
                         ddet1152, edet1152, abdet2304, cddet2304, cdedet,
                         deter, splitter, isperrboundB, isperrboundC,
                         resulterrbound)


def test2d():

    B = np.empty(shape=4, dtype=np.float64)
    C1 = np.empty(shape=8, dtype=np.float64)
    C2 = np.empty(shape=12, dtype=np.float64)
    D = np.empty(shape=16, dtype=np.float64)
    u = np.empty(shape=4, dtype=np.float64)
    bc = np.empty(shape=4, dtype=np.float64)
    ca = np.empty(shape=4, dtype=np.float64)
    ab = np.empty(shape=4, dtype=np.float64)
    axbc = np.empty(shape=8, dtype=np.float64)
    axxbc = np.empty(shape=16, dtype=np.float64)
    aybc = np.empty(shape=8, dtype=np.float64)
    ayybc = np.empty(shape=16, dtype=np.float64)
    adet = np.empty(shape=32, dtype=np.float64)
    bxca = np.empty(shape=8, dtype=np.float64)
    bxxca = np.empty(shape=16, dtype=np.float64)
    byca = np.empty(shape=8, dtype=np.float64)
    byyca = np.empty(shape=16, dtype=np.float64)
    bdet = np.empty(shape=32, dtype=np.float64)
    cxab = np.empty(shape=8, dtype=np.float64)
    cxxab = np.empty(shape=16, dtype=np.float64)
    cyab = np.empty(shape=8, dtype=np.float64)
    cyyab = np.empty(shape=16, dtype=np.float64)
    cdet = np.empty(shape=32, dtype=np.float64)
    abdet = np.empty(shape=64, dtype=np.float64)
    fin1 = np.empty(shape=1152, dtype=np.float64)
    fin2 = np.empty(shape=1152, dtype=np.float64)
    temp8 = np.empty(shape=8, dtype=np.float64)
    temp16a = np.empty(shape=16, dtype=np.float64)
    temp16b = np.empty(shape=16, dtype=np.float64)
    temp16c = np.empty(shape=16, dtype=np.float64)
    temp32a = np.empty(shape=32, dtype=np.float64)
    temp32b = np.empty(shape=32, dtype=np.float64)
    temp48 = np.empty(shape=48, dtype=np.float64)
    temp64 = np.empty(shape=64, dtype=np.float64)
    axtbb = np.empty(shape=8, dtype=np.float64)
    axtcc = np.empty(shape=8, dtype=np.float64)
    aytbb = np.empty(shape=8, dtype=np.float64)
    aytcc = np.empty(shape=8, dtype=np.float64)
    bxtaa = np.empty(shape=8, dtype=np.float64)
    bxtcc = np.empty(shape=8, dtype=np.float64)
    bytaa = np.empty(shape=8, dtype=np.float64)
    bytcc = np.empty(shape=8, dtype=np.float64)
    cxtaa = np.empty(shape=8, dtype=np.float64)
    cxtbb = np.empty(shape=8, dtype=np.float64)
    cytaa = np.empty(shape=8, dtype=np.float64)
    cytbb = np.empty(shape=8, dtype=np.float64)
    axtbc = np.empty(shape=8, dtype=np.float64)
    aytbc = np.empty(shape=8, dtype=np.float64)
    bxtca = np.empty(shape=8, dtype=np.float64)
    bytca = np.empty(shape=8, dtype=np.float64)
    cxtab = np.empty(shape=8, dtype=np.float64)
    cytab = np.empty(shape=8, dtype=np.float64)
    axtbct = np.empty(shape=16, dtype=np.float64)
    aytbct = np.empty(shape=16, dtype=np.float64)
    bxtcat = np.empty(shape=16, dtype=np.float64)
    bytcat = np.empty(shape=16, dtype=np.float64)
    cxtabt = np.empty(shape=16, dtype=np.float64)
    cytabt = np.empty(shape=16, dtype=np.float64)
    axtbctt = np.empty(shape=8, dtype=np.float64)
    aytbctt = np.empty(shape=8, dtype=np.float64)
    bxtcatt = np.empty(shape=8, dtype=np.float64)
    bytcatt = np.empty(shape=8, dtype=np.float64)
    cxtabtt = np.empty(shape=8, dtype=np.float64)
    cytabtt = np.empty(shape=8, dtype=np.float64)
    abt = np.empty(shape=8, dtype=np.float64)
    bct = np.empty(shape=8, dtype=np.float64)
    cat = np.empty(shape=8, dtype=np.float64)
    abtt = np.empty(shape=4, dtype=np.float64)
    bctt = np.empty(shape=4, dtype=np.float64)
    catt = np.empty(shape=4, dtype=np.float64)
    resulterrbound, ccwerrboundA, ccwerrboundB, ccwerrboundC, \
    iccerrboundA, iccerrboundB, iccerrboundC, splitter = exactinit2d()


    a_x = 0.0
    a_y = 0.0
    b_x = 1.0
    b_y = 0.0
    c_x = 3.0
    c_y = 1.0
    d_x = 0.0
    d_y = 0.5
    print(orient2d(
            a_x, a_y, b_x, b_y, c_x, c_y, splitter, B, C1, C2, D, u,
            ccwerrboundA, ccwerrboundB, ccwerrboundC, resulterrbound))

    print(incircle(
            a_x, a_y, b_x, b_y, c_x, c_y, d_x, d_y, bc, ca, ab, axbc, axxbc,
            aybc, ayybc, adet, bxca, bxxca, byca, byyca, bdet, cxab, cxxab,
            cyab, cyyab, cdet, abdet, fin1, fin2, temp8, temp16a, temp16b,
            temp16c, temp32a, temp32b, temp48, temp64, axtbb, axtcc, aytbb,
            aytcc, bxtaa, bxtcc, bytaa, bytcc, cxtaa, cxtbb, cytaa, cytbb,
            axtbc, aytbc, bxtca, bytca, cxtab, cytab, axtbct, aytbct, bxtcat,
            bytcat, cxtabt, cytabt, axtbctt, aytbctt, bxtcatt, bytcatt,
            cxtabtt, cytabtt, abt, bct, cat, abtt, bctt, catt, splitter,
            iccerrboundA, iccerrboundB, iccerrboundC, resulterrbound))


if __name__ == "__main__":

    test2d()