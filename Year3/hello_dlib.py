'''
An example using opencv dlib.
'''

import dlib

x = dlib.vectors()
y = dlib.array()

# Make a training dataset.
# For binary classification, the y labels should all be either +1 or -1.
x.append(dlib.vector([1, 2, 3, -1, -2, -3]))
y.append(+1)

x.append(dlib.vector([-1, -2, -3, 1, 2, 3]))
y.append(-1)


# Make a training object. This object is responsible for turning a
# training dataset into a prediction model. This one here is a SVM trainer
# that uses a linear kernel.
svm = dlib.svm_c_trainer_linear()
svm.be_verbose()
svm.set_c(10)

# Train the model. The return value is the trained model capable of making predictions.
classifier = svm.train(x, y)

# Run the model on our data and look at the results.
print("prediction for first sample:  {}".format(classifier(x[0])))
print("prediction for second sample: {}".format(classifier(x[1])))
