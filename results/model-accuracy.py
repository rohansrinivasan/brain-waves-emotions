# Base Model Accuracy
model_acc = model.evaluate(x_test, y_test, verbose=0)[1]
print("Test Accuracy: {:.3f}%".format(model_acc * 100))

