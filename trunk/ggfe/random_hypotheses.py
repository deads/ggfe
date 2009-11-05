import numpy as np

from numpy import pi, zeros, cos, sin, vstack, hstack
from numpy.random import rand, permutation

def artificial_hits(sz, centers, hit_spread_max, dr=0.9, far=0.1, hits_per_detection=1):
    (num_centers, dummy) = centers.shape
    #out = np.zeros(sz, dtype='f')
    num_misses = num_centers - int(dr * num_centers)
    center_hits = np.random.permutation(num_centers)
    center_hits = center_hits[0:(num_centers-num_misses)]
    artificial_hits = []
    confidences = []
    num_false_alarms = int(far * num_centers)
    false_alarms = []
    for (y, x) in centers[center_hits, :]:
        for i in xrange(0, hits_per_detection):
            rho = rand() * hit_spread_max
            theta = rand() * 2. * pi
            rx = np.int_(max(0, x - rho * cos(theta)))
            ry = np.int_(max(0, y - rho * sin(theta)))
            rx = min(sz[1], rx)
            ry = min(sz[0], ry)
            confidence = rand() * 0.7 + 0.3
            artificial_hits.append([ry, rx])
            confidences.append(confidence)

    false_alarms = np.int_(rand(num_false_alarms, 2) * np.array(sz))
    fa_confidences = rand(num_false_alarms)

    artificial_hits = vstack([artificial_hits, false_alarms])
    confidences = hstack([confidences, fa_confidences])

    print "SSS", confidences.shape
    return (np.array(artificial_hits, dtype='i'), np.array(confidences, dtype='d'))
