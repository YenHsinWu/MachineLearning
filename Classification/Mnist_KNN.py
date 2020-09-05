import numpy as np

from tensorflow.keras.datasets import mnist

def ComputeDistance(x_train, data):
	"""
	Compute the distances between given data and all training data.
	The distance is defined as the inner product of data and training data.
	Return an array of all 60000 distances.
	"""
	distances = []

	for i in range(x_train.shape[0]):
		distances.append(np.inner(x_train[i], data))

	return np.array(distances)

def FindKNearestNeighbors(distances, K):
	"""
	Find the indice of K nearest distances and return them as an array.
	"""
	neighbor_idx = []

	while len(neighbor_idx) < K:
		neighbor_idx.append(np.argmax(distances))
		distances[neighbor_idx[-1]] = 0

	return np.array(neighbor_idx)

def RecognizeNumber(y_train, indices):
	"""
	Count the appearance of each digits and return the one with maximum appearance.
	"""
	counter = np.zeros(10)

	for idx in indices:
		counter[int(y_train[idx])] += 1

	return np.argmax(counter)

def TestAccuracy(x_train, x_test, y_train, y_test, K):
	"""
	Compute the accuracy of the KNN learning model.
	"""
	correct_cnt = 0

	for i in range(x_test.shape[0]):
		distances = ComputeDistance(x_train, x_test[i])
		indices = FindKNearestNeighbors(distances, K)
		y_pred = RecognizeNumber(y_train, indices)

		if (y_pred == y_test[i]):
			correct_cnt += 1

	return correct_cnt / x_test.shape[0]

if __name__ == "__main__":
	
	image_size = 28
	K = 5 # number of nearest neighbors

	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	x_train = x_train.reshape(-1, image_size*image_size)
	x_test = x_test.reshape(-1, image_size*image_size)

	accuracy = TestAccuracy(x_train, x_test, y_train, y_test, K)
	print("{:.3f}%".format(accuracy))
