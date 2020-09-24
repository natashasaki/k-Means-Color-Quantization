from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os


# initialize centroids to rgb values of randomly chose pixels of image
def init_centroids(num_clusters, image):
    centroids_init = np.zeros((num_clusters, 3))
    for i in range(num_clusters): # set centroid to RGB values of random pixels
        h, w = np.random.randint(image.shape[0]), np.random.randint(image.shape[1])
        centroids_init[i] = image[h][w]

    return centroids_init


# k-means update step for centroids carried out a maximum of max_iter times
def update_centroids(centroids, image):
    max_iter = 300
    k = centroids.shape[0]
    prev_labels = np.zeros((image.shape[0], image.shape[1])) #initialize labels for all pix in image

    # loop a max time in case no "convergence" in clustering
    for i in range(max_iter):
        # rgb distance for all clusters
        dist = np.array([np.linalg.norm(c - image, axis=2) for c in centroids])
        labels = np.argmin(dist, axis=0)

        if (prev_labels == labels).all():
            print(i)
            return centroids
        else:
            for c in range(k): #update each cluster center to average
                centroids[c] = np.mean(image[c == labels], axis=0)

        prev_labels = labels
    print("updated centroids")
    return centroids


# updates the image pixel colors to closest one in centroids
def update_image(image, centroids):
    centroids = np.around(centroids)
    dist = np.array([np.linalg.norm(c - image, axis=2) for c in centroids])
    labels = np.argmin(dist, axis=0)
    Cost = np.sum(np.square(image - centroids[labels]))

    return Cost, centroids[labels].astype(np.uint8)


### MAIN ###
image_path = './HokusaiBeautiful.gif'
figure_idx = 0
Costs = []
# Load image
image = np.copy(mpimg.imread(image_path))

image.setflags(write=1)
print('[INFO] Loaded image with shape: {}'.format(np.shape(image)))
plt.figure(figure_idx)
plt.imshow(image)
plt.title('Original image')
plt.axis('off')
savepath = os.path.join('.', 'original.png')
plt.savefig(fname=savepath, transparent=True, format='png', bbox_inches='tight')

# images and calculations for c = 1,2,...10
for i in range(10):
    num_clusters = i + 1
    print(25 * '=')
    print('Updating image ' + str(num_clusters) + ' clusters ...')
    print(25 * '=')
    image = np.copy(mpimg.imread(image_path))
    image = image[..., :3]
    centroids_init = init_centroids(num_clusters, image)
    centroids = update_centroids(centroids_init, image)

    cost, image_clustered = update_image(image, centroids)
    figure_idx += 1
    Costs.append(cost)
    plt.figure(figure_idx)

    plt.imshow(image_clustered)
    plt.title('Updated image c = ' + str(num_clusters))
    plt.axis('off')
    savepath = os.path.join('.', 'updated_' + str(num_clusters) + '.png')
    plt.savefig(fname=savepath, transparent=True, format='png', bbox_inches='tight')
    plt.show()

# output Costs plot
figure_idx += 1
plt.figure(figure_idx)
x = np.arange(10) + 1
plt.plot(x, Costs)
plt.ylabel('Cluster Cost')
plt.xlabel('# of Clusters')
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
plt.title("# of Clusters vs Cluster Cost")
savepath = os.path.join('.', 'Costs.png')
plt.savefig(fname=savepath, transparent=True, format='png', bbox_inches='tight')
plt.show()
