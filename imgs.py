from sklearn.datasets import load_digits
from sklearn import svm
import matplotlib.pyplot as plt
import numpy

digits_data = load_digits()
img_samples = len(digits_data.images)

img = digits_data.images.reshape(img_samples, -1)
labels = digits_data.target

classify = svm.SVC(gamma=0.001)

#*  Train model
img_half = img[:img_samples // 2]
labels_half = labels[:img_samples // 2]
classify.fit(img_half, labels_half)

#*  Predict
#labels_expected = digits_data.target[img_samples // 2:]
img_predicted = classify.predict(img[img_samples // 2:])
image_predictions = list(zip(digits_data.images[img_samples // 2:], img_predicted))

#*  Loop predicted (images, labels) tuple
for i, (image, predict) in enumerate(image_predictions[:24]):
    plt.subplot(4, 6, i+1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title("Predict: %i" % predict)

plt.show()