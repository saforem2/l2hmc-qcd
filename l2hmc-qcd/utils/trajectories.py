import numpy as np
from .data_utils import calc_avg_vals_errors


def _calc_distances(trajectories, normalize=True):
    """Caclulate the (Euclidean) distance traveled for each
    trajectory.

    Args:
        trajectories (array-like):
            Array containing self.params['num_samples'] unique
            trajectories, where each trajectory is
            self.params['test_trajectory_length'] steps long.
    Returns:
        distances_arr (np.ndarray):
            Array containing the total distance traveled during each
            trajectory in trajectories.
    """
    distances_arr = []
    for trajectory in trajectories:
        diffs = trajectory[:-1, :] - trajectory[1:, :]
        distance = sum([np.sqrt(np.dot(d, d.T)) for d in diffs])
        # normalize to calculate distance / step
        if normalize:
            distances_arr.append(distance / len(trajectory))
        else:
            distances_arr.append(distance)
    return np.array(distances_arr)


def calc_avg_distances(trajectories, normalize=True):
    """Calculate average (Euclidean) distance traveled by each trajectory
    (with errors) using block jackknife resampling from
    self.calc_avg_vals_errors method. """
    distances = _calc_distances(trajectories, normalize)
    distances_avg_err = calc_avg_vals_errors(distances)
    return distances_avg_err

def _match_distribution(pt, means, num_distributions):
    """Given a point pt and multiple distributions (each with their own
    respective mean, contained in `means`), try to identify which distribution
    the point pt belongs to.

    Args:
        pt (scalar or array-like):
            Point belonging to some unknown distribution.
        means (array-like):
            Array containing the mean vectors of different normal
            distributions.
    Returns:
        Index in `means` corresponding to the distribution `x` is closest  to.
    """
    norm_diff_arr = []
    #  for mean in means:
    for row in range(num_distributions):
        #  diff = x - meaan
        diff = pt - means[row]
        norm_diff = np.sqrt(np.dot(diff.T, diff))
        norm_diff_arr.append(norm_diff)
    return np.argmin(np.array(norm_diff_arr))


def calc_tunneling_rate(trajectory, means, num_distributions):
    """
    Calculate the tunneling rate of `trajectory` which is equal to the total
    number of tunneling events divided by the length of the trajectory - 1.

    Given a point along the trajectory, we say that the point `belongs` to the
    distribution its closest to. A tunneling event, then, is said to occur if
    the next point in a trajectory belongs to a different distribution than the
    previous point.

    Args:
        trajectory (array-like): 
            Array containing time ordered positions.
        means (array-like): 
            Array containing the locations of the means of each distribution.
        num_distributions (int):
            Number of unique target distributions for the model.
    Returns:
        tunneling_rate (float)
    """
    idxs = [(i, i+1) for i in range(len(trajectory) - 1)]
    #  tunneling_events = {}
    num_events = 0
    for pair in idxs:
        x0 = trajectory[pair[0]]
        x1 = trajectory[pair[1]]
        dist0 = _match_distribution(x0, means, num_distributions)
        dist1 = _match_distribution(x1, means, num_distributions)
        if dist1 != dist0:
            #  tunneling_events[pair] = [x0, x1]
            num_events += 1
    tunneling_rate = num_events / (len(trajectory) - 1)
    return tunneling_rate

#  def calc_min_distance(means, cov):
#      idxs = [(i, j) for i in range(len(means) - 1, 0, -1)
#              for j in range(len(means) - 1) if i > j]
#      min_dist = []
#      for pair in idxs:
#          _vec = means[pair[1]] - means[pair[0]]
#          _dist = np.sqrt(_vec.T.dot(_vec))
#          _unit_vec = np.sqrt(cov) * _vec / _dist
#          p0 = means[pair[0]] + _unit_vec
#          p1 = means[pair[1]] - _unit_vec
#          _diff = p1 - p0
#          _min_dist = np.sqrt(_diff.T.dot(_diff))
#          min_dist.append(_min_dist)
#      return min(min_dist)
#
#  def calc_tunneling_rate(trajectory, min_distance):
#      idxs = [(i, i+1) for i in range(len(trajectory) - 1)]
#      #min_distance = calc_min_dist(means, cov)
#      tunneling_events = {}
#      num_events = 0
#      for pair in idxs:
#          _distance = distance(trajectory[pair[0]], trajectory[pair[1]])
#          if _distance >= min_distance:
#              tunneling_events[pair] = _distance
#              num_events += 1
#      tunneling_rate = num_events / (len(trajectory) - 1)
#      return tunneling_events, tunneling_rate

