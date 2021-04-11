#Nonlinearly separable data
#Kernels: Takes two inputs and outputs their similarity
#Done using inner product. Inner product and dot product are the same thing
#Trying to get to some new dimensional space. Initially deal in X space. Trying to get to z space
#Can we interchange x and z? Is every interaction with the optimization and svm a dot product?
#y = sign(w.x+b)        Could we interchange x with z? yes. Returns a scalar if it's 2D or 50D
#for constraints: yi(xi.w+b)-1>=0   yes.can replace
#W = sum(alphai . yi . xi)          yes.
#L = sum(alpha) - 1/2sum(sum alphai . alphaij . yi . yj (xi . xj)
#Kernel takes two inputs and outputs the similarity using inner products. Inner product is a projection of x1 onto x2.
#   How much overlapping do we have going on? Degree of similarity. Can use kernel to transform feature space, because
#   every interaction is an inner product.

# K(x,x') = z . z'          z is a function applied to its x counterpart    z' = function(x')   Identical function
# Convert x to z. then dot product between z and z'. And that's a kernel
# Can we calculate the inner product of the z space without knowing/processing that z space
# X = [x1, x2] => 2nd order polynomial      results in six degree
# Z = [1,x1,x2,x1^2,x2^2,x1x2] Z' = [1,x1',x2',x1'^2,x2'^2,x1x2']
# K(x,x') = z.z' = 1 + x1x1' + x2x2' + x1^2x1'^2 + x2^2x2'^2 + x1x1'x2x2'
# K(x,x') = (1 + x.x')^p    p=2 n=2     (1+x1x1' + ... + xnxn')^p     if we increase p=1xdif-100 or n=15? Not much more difficult
#
# Radial basis function Kernel (RBF)    Default kernel
# K(x,x') = exp(-gamma||x-x'||^2)       exp(x) = e^x        may force Data into linear separability
# Soft margin SVM       Caution. Support vectors use almost all Data points. Problem of overfitting. Can query to look for #SVs / #Samples. >10-20% signal of overfit.
# Soft margin - some degree of error. distance from hyperplane to misclassified points. Allow for "slack"
# Slack = Epsilon. yi(xiw+b)>=1-Epsilon     Epsilon >= 0    =0 would be hard margin
# Would like to minimize slack      = 1/2 ||w||^2 + C * sum(epsilon)    If we raise C, we want less violations. If we lower C, more allowing of violation. Smaller C, less slack (error) matters
#                                                                           Decide how important slack is in relation to minimizing vector w
# In most cases running a soft margin classifier


