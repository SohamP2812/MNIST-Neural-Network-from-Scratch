# MNIST-Neural-Network-from-Scratch

A deep neural network model with one hidden layer that can classify digits from 0-9 (MNIST Dataset)

How can you test it out?

After you have cloned this repo, you must download the MNIST dataset file and save it to the same folder as the python script.
Please find this CSV File in the URL below:
https://www.dropbox.com/s/t4i83ccpdqe9vhy/mnist_train.csv?dl=0

You then must install the required packages.
Run "pip3 install -r requirements.txt"

You are now set to run the model!
Run "python3 MNIST_NN.py"

It will automatically train the model and notify you when it is finished. You can see the cost going down as time passes (this means it's getting more accurate!).
When it is finished, it will prompt you to pick a number of visual tests to perform. This is just to show you how the model perceives a few sample images. Pick a number like 10.

It will then run the tests on the rest of the testing data, and come up with an accuracy.

Be sure to play around with the learning rate and iterations and try to improve the accuracy!
