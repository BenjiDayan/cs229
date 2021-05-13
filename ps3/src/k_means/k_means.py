from __future__ import division, print_function
import argparse
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os

from functools import reduce
prod = lambda iterable: reduce(lambda x, y: x*y, iterable)


def init_centroids(num_clusters, image):
    """
    Initialize a `num_clusters` x image_shape[-1] nparray to RGB
    values of randomly chosen pixels of`image`

    Parameters
    ----------
    num_clusters : int
        Number of centroids/clusters
    image : nparray
        (H, W, C) image represented as an nparray

    Returns
    -------
    centroids_init : nparray
        Randomly initialized centroids
    """

    # *** START YOUR CODE ***
    # raise NotImplementedError('init_centroids function not implemented')
    # *** END YOUR CODE ***
    H, W, C = image.shape
    N = H * W

    points = np.random.choice(N, num_clusters, replace=False)
    point_2_pair = lambda x: (x // W , x % W )
    centroids_init = [image[a, b] for a, b in map(point_2_pair, points)]

    return np.array(centroids_init)


def update_centroids(centroids, image, max_iter=30, print_every=10):
    """
    Carry out k-means centroid update step `max_iter` times

    Parameters
    ----------
    centroids : nparray
        (num_clusters, C) shape - The centroids stored as an nparray
    image : nparray
        (H, W, C) image represented as an nparray
    max_iter : int
        Number of iterations to run
    print_every : int
        Frequency of status update

    Returns
    -------
    new_centroids : nparray
        Updated centroids
    """

    # *** START YOUR CODE ***
    # raise NotImplementedError('update_centroids function not implemented')
        # Usually expected to converge long before `max_iter` iterations
                # Initialize `dist` vector to keep track of distance to every centroid
                # Loop over all centroids and store distances in `dist`
                # Find closest centroid and update `new_centroids`
        # Update `new_centroids`
    # *** END YOUR CODE ***
    H, W, C = image.shape
    for i in range(max_iter):
        # Extract just the final C-vec (i.e. 3-vector for RGB)
        points = image.reshape(H*W, C).astype(np.float64)
        point_centroid_distances, points_closest_centroid = get_centroid_distances(points, centroids)

        new_centroids = np.zeros((len(centroids), C))
        for centroid_i in range(len(centroids)):
            centroid_points = points[points_closest_centroid == centroid_i]
            new_centroids[centroid_i, :] = centroid_points.sum(axis=0) / centroid_points.shape[0]

        centroids = new_centroids

        if i % print_every == 0:
            i, j = 0,1
            num_points = 200
            sub_points = np.random.choice(len(points), num_points)
            sub_points = points[sub_points]
            # plt.figure()
            # plt.scatter(sub_points[:, i], sub_points[:, j], s=2)
            # plt.scatter(centroids[:, i], centroids[:, j], s=10, marker='x')
            min_dists = point_centroid_distances[points_closest_centroid, np.arange(point_centroid_distances.shape[1])]
            avg_dist = min_dists.sum()/len(min_dists)
            print(f'Step {i} of {max_iter}.\n Average point distance from nearest centroid: {avg_dist}')

    return new_centroids


def get_centroid_distances(points, centroids):
    # k centroids. N x C points.

    # k lots of N x C arrays
    point_centroid_distances = [points - centroid for centroid in centroids]

    # k x N array of point distance to each centroid
    point_centroid_distances = np.array(
        [(distances * distances).sum(axis=1) for distances in point_centroid_distances]
    )
    # N-array of index of closest centroid
    points_closest_centroid = point_centroid_distances.argmin(axis=0)
    return point_centroid_distances, points_closest_centroid

def update_image(image, centroids):
    """
    Update RGB values of pixels in `image` by finding
    the closest among the `centroids`

    Parameters
    ----------
    image : nparray
        (H, W, C) image represented as an nparray
    centroids : int
        The centroids stored as an nparray

    Returns
    -------
    image : nparray
        Updated image
    """

    # *** START YOUR CODE ***
    # raise NotImplementedError('update_image function not implemented')
            # Initialize `dist` vector to keep track of distance to every centroid
            # Loop over all centroids and store distances in `dist`
            # Find closest centroid and update pixel value in `image`
    # *** END YOUR CODE ***

    points = image.reshape(-1, image.shape[-1])
    point_centroid_distances, points_closest_centroid = get_centroid_distances(points, centroids)

    replacement_image = centroids[points_closest_centroid, :].reshape(image.shape)
    return np.round(replacement_image.astype(np.uint8))


def main(args):

    # Setup
    max_iter = args.max_iter
    print_every = args.print_every
    image_path_small = args.small_path
    image_path_large = args.large_path
    num_clusters = args.num_clusters
    figure_idx = 0

    # Load small image
    image = np.copy(mpimg.imread(image_path_small))
    print('[INFO] Loaded small image with shape: {}'.format(np.shape(image)))
    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image)
    plt.title('Original small image')
    plt.axis('off')
    savepath = os.path.join('.', 'orig_small.png')
    plt.savefig(savepath, transparent=True, format='png', bbox_inches='tight')

    # Initialize centroids
    print('[INFO] Centroids initialized')
    centroids_init = init_centroids(num_clusters, image)

    # Update centroids
    print(25 * '=')
    print('Updating centroids ...')
    print(25 * '=')
    centroids = update_centroids(centroids_init, image, max_iter, print_every)

    # Load large image
    image = np.copy(mpimg.imread(image_path_large))
    image.setflags(write=1)
    print('[INFO] Loaded large image with shape: {}'.format(np.shape(image)))
    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image)
    plt.title('Original large image')
    plt.axis('off')
    savepath = os.path.join('.', 'orig_large.png')
    plt.savefig(fname=savepath, transparent=True, format='png', bbox_inches='tight')

    # Update large image with centroids calculated on small image
    print(25 * '=')
    print('Updating large image ...')
    print(25 * '=')
    image_clustered = update_image(image, centroids)

    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image_clustered)
    plt.title('Updated large image')
    plt.axis('off')
    savepath = os.path.join('.', 'updated_large.png')
    plt.savefig(fname=savepath, transparent=True, format='png', bbox_inches='tight')

    print('\nCOMPLETE')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--small_path', default='./peppers-small.tiff',
                        help='Path to small image')
    parser.add_argument('--large_path', default='./peppers-large.tiff',
                        help='Path to large image')
    parser.add_argument('--max_iter', type=int, default=150,
                        help='Maximum number of iterations')
    parser.add_argument('--num_clusters', type=int, default=16,
                        help='Number of centroids/clusters')
    parser.add_argument('--print_every', type=int, default=10,
                        help='Iteration print frequency')
    args = parser.parse_args()
    main(args)
