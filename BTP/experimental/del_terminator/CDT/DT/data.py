import numpy as np

def make_data():
    points_1 = np.array([
        0.0, 0.0,
        .0075, .0176,
        .0125, .0215,
        .0250, .0276,
        .0375, .0316,
        .0500, .0347,
        .0750, .0394,
        .1000, .0428,
        .1250, .0455,
        .1500, .0476,
        .1750, .0493,
        .2000, .0507,
        .2500, .0528,
        .3000, .0540,
        .3500, .0547,
        .4000, .0550,
        .4500, .0548,
        .5000, .0543,
        .5500, .0533,
        .5750, .0527,
        .6000, .0519,
        .6250, .0511,
        .6500, .0501,
        .6750, .0489,
        .7000, .0476,
        .7250, .0460,
        .7500, .0442,
        .7750, .0422,
        .8000, .0398,
        .8250, .0370,
        .8500, .0337,
        .8750, .0300,
        .9000, .0255,
        .9250, .0204,
        .9500, .0144,
        .9750, .0074,
        1.0000, -.0008,
    ])
    points_2 = np.array([
        .0075, -.0176,
        .0125, -.0216,
        .0250, -.0281,
        .0375, -.0324,
        .0500, -.0358,
        .0750, -.0408,
        .1000, -.0444,
        .1250, -.0472,
        .1500, -.0493,
        .1750, -.0510,
        .2000, -.0522,
        .2500, -.0540,
        .3000, -.0548,
        .3500, -.0549,
        .4000, -.0541,
        .4500, -.0524,
        .5000, -.0497,
        .5500, -.0455,
        .5750, -.0426,
        .6000, -.0389,
        .6250, -.0342,
        .6500, -.0282,
        .6750, -.0215,
        .7000, -.0149,
        .7250, -.0090,
        .7500, -.0036,
        .7750, .0012,
        .8000, .0053,
        .8250, .0088,
        .8500, .0114,
        .8750, .0132,
        .9000, .0138,
        .9250, .0131,
        .9500, .0106,
        .9750, .0060,
        1.000, -.0013,
    ])
    points_2[0::2] = points_2[0::2][::-1]
    points_2[1::2] = points_2[1::2][::-1]

    points = np.append(points_1, points_2)
    num_points = int(0.5*len(points))
    points = points.reshape((num_points, 2))

    segments = np.array([[i, (i + 1) % num_points] for i in range(num_points)])

    points = np.append(
        points,
        np.array([
            [-1, -1.],
            [2, -1.],
            [2, 1.],
            [-1, 1.],
        ]),
        axis=0
    )
    num_points = len(points)
    segments = np.append(
        segments,
        [
            [num_points-1-3, num_points-1-2],
            [num_points-1-2, num_points-1-1],
            [num_points-1-1, num_points-1-0],
            [num_points-1-0, num_points-1-3]
        ],
        axis=0
    )


    # N = 10
    # xpts = np.linspace(-1.0, 2.0, N)
    # ypts = np.linspace(-1.0, -1.0, N)
    # tpoints = np.empty(shape=(N*N, 2), dtype=np.float64)
    # for i in range(N):
    #     tpoints[i*N:(i+1)*N, 0] = xpts
    #     tpoints[i*N:(i+1)*N:, 1] = ypts[i]
    # points = np.append(points, tpoints, axis=0)

    # rand_pts = np.random.rand(100, 2)
    # rand_pts[:, 0] = 3*(3*rand_pts[:, 0] - 1.0)
    # rand_pts[:, 1] = 3*(3*rand_pts[:, 1] - 1.5)

    # rand_pts = 3*(3*np.random.rand(100, 2) - 1.5)
    # points = np.append(points, rand_pts, axis=0)

    return points, segments

