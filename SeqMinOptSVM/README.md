SeqMinOptSvm is an implementation of SVM classification which uses sequential minimal optimizations to find support vectors
and their associated weights (alphas).

I am by no means an expert on the topic. The purpose of the code was to better understand how SMO worked while in the process
i get some experience programming in scala.

The code is not complete. Instead of using a scientific package (equivalent to matlab/numpy in scala) in the spirit of learning
chose to implement most of the matrix algebra i needed myself. See MatrixAlgraUtil.scala

ATM The code supports linear and rbf kernels and will expand it to take a file larger than 3 columns :)