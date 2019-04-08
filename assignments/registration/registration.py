import numpy as np
import scipy as sp
from scipy import spatial

def paired_points_matching(source, target):
    """
    Calculates the transformation T that maps the source to the target
    :param source: A N x 3 matrix with N 3D points
    :param target: A N x 3 matrix with N 3D points
    :return:
        T: 4x4 transformation matrix mapping source onto target
        R: 3x3 rotation matrix part of T
        t: 1x3 translation vector part of T
    """
    assert source.shape == target.shape
    T = np.eye(4)
    R = np.eye(3)
    t = np.zeros((1, 3))

    N = source.shape[0]

    centroidSource = np.mean(source, axis=0)
    centroidTarget = np.mean(target, axis=0)

    srcCentrAlign = source - np.tile(centroidSource, (N, 1))
    trgCentrAlign = target - np.tile(centroidTarget, (N, 1))

    covMatrix = np.dot(srcCentrAlign.T, trgCentrAlign)

    uMat, sMat, vMat = np.linalg.svd(covMatrix)

    R = np.dot(vMat.T, uMat.T)

    t = - (np.dot(R, centroidSource.T)) + centroidTarget

    T[:3, :3] = R
    T[:3, 3] = t

    return T, R, t


def find_nearest_neighbor(src, dst):
    """
    Finds the nearest neighbor of every point in src in dst
    :param src: A N x 3 point cloud
    :param dst: A N x 3 point cloud
    :return: the
    """
    tree = sp.spatial.KDTree(src)
    distance, index = tree.query(dst)

    return distance, index


def icp(source, target, init_pose=None, max_iterations=10, tolerance=0.0001):
    """
    Iteratively finds the best transformation that mapps the source points onto the target
    :param source: A N x 3 point cloud
    :param target: A N x 3 point cloud
    :param init_pose: A 4 x 4 transformation matrix for the initial pose
    :param max_iterations: default 10
    :param tolerance: maximum allowed error
        :return: A 4 x 4 rigid transformation matrix mapping source to target
            the distances and the error
    """
    #T = np.eye(4)
    distances = 0
    error = 0

    # Your code goes here

    src_init = np.dot(init_pose[:3, :3], source.T).T
    src_init = src_init + np.tile(init_pose[:3, 3], (source.shape[0], 1))

    tmp_trg = np.zeros_like(source)

    if init_pose is None:
        T = np.eye(4)
    else:
        T = init_pose

    tmp_tol = np.inf
    #error = np.inf
    k = 0

    for i in range(max_iterations):
        while tmp_tol > tolerance:
            distances, index = find_nearest_neighbor(src_init, target)
            for ii, el in enumerate(index):
                tmp_trg[ii] = target[el]

            T_tmp, R_tmp, t_tmp = paired_points_matching(src_init, tmp_trg)

            src_init = np.dot(R_tmp, src_init.T).T
            src_init = src_init + np.tile(t_tmp, (source.shape[0], 1))
            T = np.dot(T_tmp, T)

            err_tmp = error
            error = np.sum(distances) / distances.shape[0]
            error = np.sqrt(error)
            tmp_tol = err_tmp - error
            # print(tmp_tol)

            k += 1

    print("Iterations: ", k)
    return T, distances, error


def get_initial_pose(template_points, target_points):
    """
    Calculates an initial rough registration
    (Optionally you can also return a hand picked initial pose)
    :param source:
    :param target:
    :return: A transformation matrix
    """
    T = np.eye(4)

    # Your code goes here

    centr_tmpl = np.mean(template_points, axis=0)
    centr_target = np.mean(target_points, axis=0)
    t = centr_target - centr_tmpl

    T[:3, 3] = t

    return T

