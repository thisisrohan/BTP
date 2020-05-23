import numpy as np
import math


###############################################################################

# computing machine epsilon for np.float64
one = np.float64(1.0)
half = np.float64(0.5)
epsilon = np.float64(1.0)
while one + epsilon != one:
    epsilon_old = epsilon
    epsilon *= half

###############################################################################

def ulp(m):
    exp = np.float64(math.frexp(m)[1])
    return epsilon*(2.0**exp)


class number:
    '''
    special class that helps compute the static bounds on a predicate's
    round-off error
    '''
    def __init__(self, m=1.0, e=0.0):
        self.m = m
        self.e = np.abs(e)
    def __add__(self, num):
        m = self.m + num.m
        e = self.e + num.e + 0.5*ulp(m)
        return number(m, e)
    def __sub__(self, num):
        m = self.m + num.m
        e = self.e + num.e + 0.5*ulp(m)
        return number(m, e)
    def __mul__(self, num):
        m = self.m*num.m
        e = np.abs(self.e*num.m) + np.abs(self.m*num.e) + 0.5*ulp(m)
        return number(m, e)

###############################################################################

def orient2d(ax, ay, bx, by, cx, cy):
    detleft = (ax - cx) * (by - cy)
    detright = (ay - cy) * (bx - cx)
    det = detleft - detright
    return det


def orient3d(ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz):
    adx = ax - dx
    ady = ay - dy
    adz = az - dz

    bdx = bx - dx
    bdy = by - dy
    bdz = bz - dz

    cdx = cx - dx
    cdy = cy - dy
    cdz = cz - dz

    m1 = adx*bdy - ady*bdx
    m2 = adx*cdy - ady*cdx
    m3 = bdx*cdy - bdy*cdx

    det = cdz*m1 - bdz*m2 + adz*m3

    return det


def incircle(ax, ay, bx, by, cx, cy, dx, dy):
    adx = ax - dx
    bdx = bx - dx
    cdx = cx - dx
    ady = ay - dy
    bdy = by - dy
    cdy = cy - dy

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

    return det


def insphere(ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz, ex, ey, ez):
    aex = ax - ex
    bex = bx - ex
    cex = cx - ex
    dex = dx - ex
    aey = ay - ey
    bey = by - ey
    cey = cy - ey
    dey = dy - ey
    aez = az - ez
    bez = bz - ez
    cez = cz - ez
    dez = dz - ez

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

    return det

###############################################################################

res = orient2d(number(), number(), number(), number(), number(), number())
print("orient2d")
print(res.e)
print(res.e/epsilon)
print("")

res = orient3d(number(), number(), number(), number(), number(), number(),
               number(), number(), number(), number(), number(), number())
print("orient3d")
print(res.e)
print(res.e/epsilon)
print("")

res = incircle(number(), number(), number(), number(), number(), number(),
               number(), number())
print("incircle")
print(res.e)
print(res.e/epsilon)
print("")

res = insphere(number(), number(), number(), number(), number(), number(),
               number(), number(), number(), number(), number(), number(),
               number(), number(), number())
print("insphere")
print(res.e)
print(res.e/epsilon)
print("")

###############################################################################
