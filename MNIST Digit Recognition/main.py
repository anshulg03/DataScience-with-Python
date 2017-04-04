import pandas as pd
import numpy as np
import gzip
import matplotlib.pyplot as plt


def load(path_img, path_lbl):
  from array import array
  import struct

  with gzip.open(path_lbl, 'rb') as file:
    magic, size = struct.unpack(">II", file.read(8))
    if magic != 2049:
      raise ValueError('Magic number mismatch, expected 2049, got {0}'.format(magic))
    labels = array("B", file.read())

  with gzip.open(path_img, 'rb') as file:
    magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
    if magic != 2051:
      raise ValueError('Magic number mismatch, expected 2051, got {0}'.format(magic))
    image_data = array("B", file.read())

  images = []
  for i in range(size): images.append([0] * rows * cols)
  for i in range(size): images[i] = np.array(image_data[i * rows * cols:(i + 1) * rows * cols]).reshape(28,28)[::3,::3].reshape(-1)
  return pd.DataFrame(images), pd.Series(labels)

def peekData(X_train):
  # The 'targets' or labels are stored in y. The 'samples' or data is stored in X
  print ("Peeking your data...")
  fig = plt.figure()

  cnt = 0
  for col in range(5):
    for row in range(10):
      plt.subplot(5, 10, cnt + 1)
      plt.imshow(X_train.ix[cnt,:].values.reshape(10,10), cmap=plt.cm.gray_r, interpolation='nearest')
      plt.axis('off')
      cnt += 1
  fig.set_tight_layout(True)
  plt.show()
  

dir = 'C:/Users/anshangu/Documents/GitHub/Python_LogisticReg/DataScience-with-Python/MNIST Digit Recognition/'

X_train, y_train = load(dir + 'train-images-idx3-ubyte.gz', dir + 'train-labels-idx1-ubyte.gz')

X_test, y_test = load(dir + 't10k-images-idx3-ubyte.gz', dir + 't10k-labels-idx1-ubyte.gz')


X_train = X_train.ix[:39999]
y_train = y_train[:40000]

peekData(X_train)


print('Training SVC Classifier...')
from sklearn.svm import SVC


model = SVC(kernel = 'linear', cache_size = 7000)
model.fit(X_train, y_train)

print('Scoring model....')
score = model.score(X_test,y_test)

print('Score \n', score)


true_1000th_test_value = y_test[1000]

print ("1000th test label: ", true_1000th_test_value)

guess_1000th_test_value = model.predict(X_test.iloc[[1000]])
print ("1000th test prediction: ", guess_1000th_test_value)
