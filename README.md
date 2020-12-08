# GENE-CLASSIFICATION
This is an in class Kaggle Competition in Kernel Method for Machine Learning at AMMI


# INTRODUCTION

Transcription factors (TFs) are regulatory proteins that bind specific sequence motifs in the genome to activate or repress transcription of target genes.
Genome-wide protein-DNA binding maps can be profiled using some experimental techniques and thus all genomics can be classified into two classes for a TF of interest: bound or unbound.

The main task of this project is to classify gene sequence: thus
predicting whether a DNA sequence region is binding site to a specific
transcription factor.

# DATA SET

The data is of two form: the principal files and the optional files.

The principal files contain data that has 2000 training points and
1000 test sequence.


# MODELS USED

1. Ridge Regression

2. Kernel Ridge Regression

3. Naive Bayes Model

4. Logistic Regression

5. Kernel Logistic Regression

6. Weighted Kernel Logistic Regression

7. Kernel Support Vector Machine

# KERNELS IMPLIMENTED 

1. Linear Kernel

2. Quadratic Kernel

3. Polynomial Kernel

4. Exponential Kernel

5. Radial Basis Kernel (RBF)

6. Laplacian Kernel

# RESULT AND FINDING

* On the Private score, the three best accuracies are: 0.684, 0.662 and
0.648 which were obtained by kernel logistic regression (polynomial
kernel), Kernel ridge Regression(Polynomial kernel) and SVM with
RBF kernel respectively.

* This indicates that, these kernels work well on the data set.

* In addition, simple models performed better than Support Vector Machine (SVM) in general.
