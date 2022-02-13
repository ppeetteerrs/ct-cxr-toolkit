# Confusing Dimensions of OpenCV, Numpy and Matplotlib

In numpy:
img.shape => H (no. of rows), W (no. of cols)

In matplotlib
plt.imshow => height and width follows numpy
plt figsize => width, height ...

In OpenCV
cv.resize(img, (X, Y)) => X is no. of cols (in x-direction), Y is no. of rows (in y-direction)



choose cxr positive samples with no annotations