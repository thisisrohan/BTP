import numpy as np

def make_data():
    N = 10
    r = 2.0
    points = np.empty(shape=(N, 2), dtype=np.float64)
    points[:, 0] = r*np.cos(2*np.pi*np.arange(N)/N)
    points[:, 1] = r*np.sin(2*np.pi*np.arange(N)/N)

    nr = 8 # 7 per edge
    rpoints = np.empty(shape=(nr*nr, 2), dtype = np.float64)
    temp = np.linspace(-r, r, nr)
    for i in range(nr):
        rpoints[nr*i:nr*(i+1), 0] = temp[i]
        rpoints[i::nr, 1] = temp[i]
    # print(rpoints)


    points = np.append(points, rpoints, axis=0)

    segments = np.array([[i, (i+1)%N] for i in range(N)])

    return points, segments

