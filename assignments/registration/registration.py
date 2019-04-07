import numpy as np
import scipy as sp
import cv2
import sys
from sklearn.neighbors import NearestNeighbors

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

    centerSource = np.zeros_like(source)
    centerTarget = np.zeros_like(target)

    centroidSource = (1/source.shape[0] * np.sum(source, axis=0))
    centroidTarget = (1 / source.shape[0] * np.sum(target, axis=0))

    for i in range(0, source.shape[0]):
        centerSource[i] = source[i] - centroidSource

    for i in range(0, source.shape[0]):
        centerTarget[i] = target[i] - centroidTarget

    transposedTargetCenter = centerTarget.transpose()

    multCenter = np.matmul(centerSource, transposedTargetCenter)

    svd = np.linalg.svd(multCenter)

    uMatrix = svd[0]
    sMatrix = svd[1]
    vMatrix = svd[2]

    uMatrixTrans = uMatrix.transpose()

    rotation = np.matmul(vMatrix, uMatrixTrans)



    return T, R, t


def find_nearest_neighbor(src, dst):
    """
    Finds the nearest neighbor of every point in src in dst
    :param src: A N x 3 point cloud
    :param dst: A N x 3 point cloud
    :return: the
    """
    tree = sp.spacial.KDTree(src)
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
    T = np.eye(4)
    distances = 0
    error = 0

    # Your code goes here

    # src = np.array([source.T], copy=True).astype(np.float32)
    # dst = np.array([target.T], copy=True).astype(np.float32)
    #
    # knn = cv2.KNearest()
    # responses = np.array(range(len(dst[0]))).astype(np.float32)
    # knn.train(src[0], responses)
    #
    # Tr = np.array([[np.cos(0), -np.sin(0), 0],
    #                [np.sin(0), np.cos(0), 0],
    #                [0, 0, 1]])
    #
    # dst = cv2.transform(dst, Tr[0:2])
    #
    # scale_x = np.max(src[0]) - np.min(src[0])
    # scale_y = np.max(src[1]) - np.min(src[1])
    # scale = max(scale_x, scale_y)
    #
    # for i in range(max_iterations):
    #     ret, results, neighbours, dist = knn.find_nearest(dst[0], 1)
    #
    #     indeces = results.astype(np.int32).T
    #     indeces = del_miss(indeces, dist, distances, tolerance)
    #
    #     T = cv2.estimateRigidTransform(dst[0, indeces], src[0, indeces], True)
    #
    #     distances = np.max(dist)
    #     dst = cv2.transform(dst, T)
    #     Tr = np.dot(np.vstack((T, [0, 0, 1])), Tr)
    #
    #     if (is_converge(T, scale)):
    #         break

    src = np.array([source.T], copy=True).astype(np.float32)
    dst = np.array([target.T], copy=True).astype(np.float32)

    # Initialise with the initial pose estimation
    Tr = np.array([[np.cos(init_pose[2]), -np.sin(init_pose[2]), init_pose[0]],
                   [np.sin(init_pose[2]), np.cos(init_pose[2]), init_pose[1]],
                   [0, 0, 1]])

    src = cv2.transform(src, Tr[0:2])

    for i in range(max_iterations):
        # Find the nearest neighbours between the current source and the
        # destination cloudpoint
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto',
                                warn_on_equidistant=False).fit(dst[0])
        distances, indices = nbrs.kneighbors(src[0])

        # Compute the transformation between the current source
        # and destination cloudpoint
        T = cv2.estimateRigidTransform(src, dst[0, indices.T], False)
        # Transform the previous source and update the
        # current source cloudpoint
        src = cv2.transform(src, T)
        # Save the transformation from the actual source cloudpoint
        # to the destination
        Tr = np.dot(Tr, np.vstack((T, [0, 0, 1])))

    return T, distances, error


def del_miss(indeces, dist, max_dist, tolerance):
    th_dist = max_dist * tolerance
    return np.array([indeces[0][np.where(dist.T[0] < th_dist)]])

def is_converge(Tr, scale):
    delta_angle = 0.0001
    delta_scale = scale * 0.0001

    min_cos = 1 - delta_angle
    max_cos = 1 + delta_angle
    min_sin = -delta_angle
    max_sin = delta_angle
    min_move = -delta_scale
    max_move = delta_scale

    return min_cos < Tr[0, 0] and Tr[0, 0] < max_cos and \
           min_cos < Tr[1, 1] and Tr[1, 1] < max_cos and \
           min_sin < -Tr[1, 0] and -Tr[1, 0] < max_sin and \
           min_sin < Tr[0, 1] and Tr[0, 1] < max_sin and \
           min_move < Tr[0, 2] and Tr[0, 2] < max_move and \
           min_move < Tr[1, 2] and Tr[1, 2] < max_move


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

    return T

