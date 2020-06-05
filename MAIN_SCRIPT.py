
if  __name__ =="__main__":

    import pandas as pd
    import seaborn as sns
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn import model_selection

    #!pip  install cvxopt


    # insert the data 
    X_train = pd.read_csv("Xtr.csv")
    Y_train = pd.read_csv("Ytr.csv")
    X_test = pd.read_csv("Xte.csv")


    # read the first 5 lines from the data
    X_train.head()

    # use Kmers Method to encode the data, by define spectrum function
    def spectrum (DNA, k):
        return [DNA[x:x+k].lower() for x in range(len(DNA)-k+1)]

    # do one hot encoder for the sequence data
    from sklearn.preprocessing import OneHotEncoder

    def on_hot(data):
        onehot_encoder = OneHotEncoder(sparse=False,categories='auto')
        data = onehot_encoder.fit_transform(data)
        return data

    # we choose k=3
    def spectrum_on_data(data,k=3):
        data = data
        d = 101 - k +1
        outs = []
        cols = ['word'+str(i) for i in range(d)]
        for ind in range(len(data)):
            seq = data.iloc[ind]['seq']
            seq = spectrum(seq,k)
            outs.append(seq)
        outs_df = pd.DataFrame(data = outs,columns=cols)
        outs_one_hot = on_hot(np.array(outs))
        return outs_one_hot

    #combine x train with x test data into one train set
    train_test = [X_train,X_test]
    train_test = pd.concat(train_test)

    train_test_onehot = spectrum_on_data(train_test)

    # split the data into train and test 
    x_train = train_test_onehot[:2000,:]
    x_test = train_test_onehot[2000:,:]

    #drop double index
    y = Y_train.drop(['Id'],axis = 1)

    # drop the first row
    y = y['Bound']

    # convert y to numpy array
    y = np.array(y)

    # convert the 0 class in y to -1
    y_svm = y
    for i in range(len(y_svm)):
        if y_svm[i] ==  0:y_svm[i] = -1

    #split the data
    from sklearn.model_selection import train_test_split
    X_train_, X_test_, y_train_, y_test_ = train_test_split( x_train, y, test_size=0.15, random_state=42,shuffle=True)

    # import cvxopt to solve convex optimization problems
    import cvxopt

    def cvxopt_qp(P, q, G, h, A, b):
        P = .5 * (P + P.T)
        cvx_matrices = [
            cvxopt.matrix(M) if M is not None else None for M in [P, q, G, h, A, b] 
        ]
        solution = cvxopt.solvers.qp(*cvx_matrices, options={'show_progress': False})
        return np.array(solution['x']).flatten()

    solve_qp = cvxopt_qp

      #define some kernels:
        
    from numpy import linalg
    import cvxopt
    import cvxopt.solvers

    def  Exponential_Kernel(x, y, sigma=5):

        return np.exp(-linalg.norm(x-y) / (2 * (sigma ** 2)))

    def quadratic_kernel(X1, X2):

        return (1 + linear_kernel(X1, X2))**2

    def  Laplacian_Kernel(x, y, sigma=10.0):

        return np.exp(-linalg.norm(x-y) / (sigma ))

    def linear_kernel(x1, x2):
        return np.dot(x1, x2)

    def polynomial_kernel(x, y, p=4):
        return (1 + np.dot(x, y.T)) ** p
    # =rbf
    def gaussian_kernel(x, y, sigma=10):
        return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))
    
    def rbf_kernel(X1, X2, sigma=10.0):

        X2_norm = np.sum(X2 ** 2, axis = -1)
        X1_norm = np.sum(X1 ** 2, axis = -1)
        gamma = 1 / (2 * sigma ** 2)
        K = np.exp(- gamma * (X1_norm[:, None] + X2_norm[None, :] - 2 * np.dot(X1, X2.T)))
        return K



    # SVM CLASS

    class SVM(object):

        def __init__(self, kernel=gaussian_kernel, C=20):
            self.kernel = kernel
            self.C = C
            if self.C is not None: self.C = float(self.C)

        def fit(self, X, y):
            n_samples = X.shape[0]
            n_features = X.shape[1]

            # define gram matrix
            K = np.zeros((n_samples, n_samples))
            for i in range(n_samples):
                for j in range(n_samples):
                    K[i,j] = self.kernel(X[i], X[j])

            P = cvxopt.matrix(np.outer(y,y) * K)
            q = cvxopt.matrix(np.ones(n_samples) * -1)
            A = y.reshape(1,n_samples)

            A = A.astype('float')
            A = cvxopt.matrix(A)

            b = cvxopt.matrix(0.0)

            if self.C is None:
                G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
                h = cvxopt.matrix(np.zeros(n_samples))
            else:
                tmp1 = np.diag(np.ones(n_samples) * -1)
                tmp2 = np.identity(n_samples)
                G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
                tmp1 = np.zeros(n_samples)
                tmp2 = np.ones(n_samples) * self.C
                h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

            # solve Quadratic Programing problem
            print('P',P.size,'q',q.size,'G',G.size,'h',h.size,'A',A.size,'b',b.size)
            solution = cvxopt.solvers.qp(P, q, G, h, A, b)

            # Lagrange multipliers
            a = np.ravel(solution['x'])

            sv = a > 1e-12
            ind = np.arange(len(a))[sv]
            self.a = a[sv]
            self.sv = X[sv]
            self.sv_y = y[sv]
            print("%d support vectors out of %d points" % (len(self.a), n_samples))

            # Intercept
            self.b = 0
            for n in range(len(self.a)):
                print(len(self.a))
                self.b += self.sv_y[n]
                self.b -= np.sum(self.a * self.sv_y * K[ind[n],sv])

            self.b /= len(self.a)

            # Weight vector
            if self.kernel == linear_kernel:
                self.w = np.zeros(n_features)
                for n in range(len(self.a)):
                    self.w += self.a[n] * self.sv_y[n] * self.sv[n]
            else:
                self.w = None

        def project(self, X):
            if self.w is not None:
                return np.dot(X, self.w) + self.b
            else:
                y_predict = np.zeros(len(X))
                for i in range(len(X)):
                    s = 0
                    for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                        s += a * sv_y * self.kernel(X[i], sv)
                    y_predict[i] = s
                return y_predict + self.b

        def predict(self, X):
            return np.sign(self.project(X))

#     #create instance from svm class
#     svm_model = SVM()

#     #fit the model
#     svm_model.fit(X_train_,y_train_)

#     # predict on splitted test data
#     ypredict = svm_model.predict(X_test_)

#     # calculate the model accuracy
#     accuracy = np.mean(ypredict == y_test_)
#     print('Accuracy is ' ,accuracy )

#     # predict on the test data
#     pred = svm_model.predict(x_test)


    # ****************Cross Validation Class***************************

    # to evaluate the model performance we can validate it using k_fold cross validation

    import random

    class k_fold_cross_validation():
        def __init__(self,model, k=3):
            self.k = k
            self.model = model

        def split_(self, X, y, i, l):

            n = len(X)//self.k
            validation_indices = l[i*n:(i+1)*n]
            training_indices = [x  for x in l if x not in validation_indices]

            validation_set = X[validation_indices]
            training_set = X[training_indices]
            validation_target = y[validation_indices]
            training_target = y[training_indices]
            return training_set, validation_set, training_target, validation_target

        def validate(self, X, y):
            l  = [j for j in range(len(X))]
            random.shuffle(l)
            accs = []
            for i in range(self.k): 
                X_train, X_test, y_train, y_test = self.split_(X,y,i,l)
                self.model.fit(X_train, y_train)
                prediction = self.model.predict(X_test)
                acc = np.mean(prediction==y_test)
                accs.append(acc)
                print("acc {} : {}".format(i,acc))
            average_loss = sum(accs)/len(accs)
            print("accuracy loss: {}".format(average_loss))

#     #choose the model you want to validate

#     kfcv=k_fold_cross_validation(svm_model)

#     # fit whole data into the k_fold cross validation class
#     kfcv.validate(x_train, y)

#     # predict in test data
#     pred = svm_model.predict(x_test)

 

    ## Kernel Ridge Regression

    class KernelMethodBase(object):

        kernels_ = {
            'linear': linear_kernel,
            'polynomial': polynomial_kernel,
            'rbf': rbf_kernel

        }
        def __init__(self, kernel='polynomial', **kwargs):
            self.kernel_name = kernel
            self.kernel_function_ = self.kernels_[kernel]
            self.kernel_parameters = self.get_kernel_parameters(**kwargs)

        def get_kernel_parameters(self, **kwargs):
            params = {}
            if self.kernel_name == 'rbf':
                params['sigma'] = kwargs.get('sigma', 1)
            if self.kernel_name == 'polynomial':
                params['p'] = kwargs.get('p', 2)
            #if self.kernel_name == 'Laplacian_Kernel':
             #     params['sigma'] = kwargs.get('sigma', 1)
            #     params['parameter_2'] = kwargs.get('parameter_2', None)
            return params

        def fit(self, X, y, **kwargs):
            return self

        def decision_function(self, X):
            pass

        def predict(self, X):
            pass
        
# ******************* Kernel Ridge Regression *********************

    class KernelRidgeRegression(KernelMethodBase):

        def __init__(self, lambd=1.10, **kwargs):
            self.lambd = lambd
            super(KernelRidgeRegression, self).__init__(**kwargs)

        def fit(self, X, y,  sample_weights=None):
            n=X.shape[0]
            p = X.shape[1]
            assert (n == len(y))

            self.X_train = X
            self.y_train = y

            if sample_weights is not None:
                w_sqrt = np.sqrt(sample_weights)
                self.X_train = self.X_train * w_sqrt[:, None]
                self.y_train = self.y_train * w_sqrt

            A = self.kernel_function_(X, X, **self.kernel_parameters) 

            A[np.diag_indices_from(A)] = np.add(A[np.diag_indices_from(A)] ,n*self.lambd)
            self.alpha = np.linalg.solve(A , self.y_train)

            return self

        def decision_function(self, X):
            K_x = self.kernel_function_(X, self.X_train,**self.kernel_parameters )
            return  np.sign(K_x.dot(self.alpha))


        def predict(self, X):
            return self.decision_function(X)    

        def Accuracy_check(self,X,y):
            return np.mean(self.predict(X)==y)

    # create instance from the class and fit the data
    KR = KernelRidgeRegression(p=5)
    KR.fit(X_train_,y_train_)

    pred = KR.predict(X_test_)

    acc = KR.Accuracy_check(X_test_,y_test_)
    acc

    # cross validation

    kfc=k_fold_cross_validation(KR)

    kfc.validate(x_train, y)

    fpred = KR.predict(x_test)



    ## ******************** Logistic Regression *************************

    class LogisticRegression():
        def __init__(self, lr=0.3, num_iter=100000, batch_size=1000, verbose=True):
            self.lr = lr
            self.num_iter = num_iter
            self.batch_size = batch_size
            self.verbose = verbose

        def __add_intercept(self, X):
            intercept = np.ones((X.shape[0], 1))
            return np.concatenate((intercept, X), axis=1)

        def sigmoid(self,x):
            if (x>0).any():
                return 1 / (1 + np.exp(-x))               
            else:
                return np.exp(x) / (1 + np.exp(x)) 

        def __loss(self, h, y):
            return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

        def fit(self, X, y):
            y = self.trans_y(y)

            X = self.__add_intercept(X)
            self.theta = np.zeros(X.shape[1])

            for i in range(self.num_iter):
                z = np.dot(X, self.theta)
                h = self.sigmoid(z)
                rand = np.random.choice(y.size, self.batch_size).squeeze()

                gradient = np.dot(X[rand].T, (h[rand].reshape(-1,1) - y[rand].reshape(-1,1)))/y.size   
                self.theta =  self.theta.reshape(-1,1)
                self.theta -= (self.lr * gradient)


                if(self.verbose == True and i % 10000 == 0):
                    z = np.dot(X, self.theta)
                    h = self.sigmoid(z)
                    print(f'loss: {self.__loss(h, y)} \t')

        def predict_probability(self, X):
            X = self.__add_intercept(X)

            return self.sigmoid(np.dot(X, self.theta))

        def predict(self, X, threshold=0.5):
              return np.where(self.predict_probability(X) >= 0.5, 1, -1)


        def Accuracy_check(self,X,y):
            return np.mean(self.predict(X)==y)

        def trans_y(self, y):
            if isinstance(y, pd.Series):
                y = y.values
            if isinstance(y, list):
                y = np.array(y)
            return y

#     LR=LogisticRegression()

#     LR.fit(X_train_,np.array(y_train_))

#     acc = LR.Accuracy_check(X_test_,y_test_)
#     acc

#     pred = LR.predict(x_test)

#     # cross validation
#     kfcLR=k_fold_cross_validation(LR)
#     kfcLR.validate(x_train, y)



  ## ******************** Kernel Logistic Regression *************************
    
# class to weight Logistic model
    class WeightedKernelRidgeRegression(KernelRidgeRegression):

        def fit(self, K, y, sample_weights=None):

            self.y_train = y
            n = len(self.y_train)

            w = np.ones_like(self.y_train) if sample_weights is None else sample_weights
            W = np.diag(np.sqrt(w))

            A = W.dot(K).dot(W)
            A[np.diag_indices_from(A)] += self.lambd * n
            self.alpha = W.dot(np.linalg.solve(A , W.dot(self.y_train)))

            return self


            
    def sigmoid(x):
        if (x>0).any():
            return 1 / (1 + np.exp(-x))               
        else:
            return np.exp(x) / (1 + np.exp(x)) 
    class KernelLogisticRegression(KernelMethodBase):

        def __init__(self, lambd=0.1, **kwargs):
            self.lambd = lambd
            # Python 3: replace the following line by
            super().__init__(**kwargs)
            #super(KernelLogisticRegression, self).__init__(**kwargs)

        def fit(self, X, y, max_iter=100, tol=0.0001):

            self.X_train = X
            self.y_train = y

            K = self.kernel_function_(X, X, **self.kernel_parameters)

            # IRLS
            WKRR = WeightedKernelRidgeRegression(
                lambd=self.lambd,
                kernel=self.kernel_name,
                **self.kernel_parameters
            )
            # Initialize
            alpha = np.zeros_like(self.y_train)
            # Iterate until convergence or max iterations
            for n_iter in range(max_iter):
                alpha_old = alpha
                f = K.dot(alpha_old)
                w = sigmoid(f) * sigmoid(-f)
               # print(sigmoid(f) )
                z = f + y / sigmoid(-y*f)
                alpha = WKRR.fit(K, z, sample_weights=w).alpha
                # Break condition (achieved convergence)
                if np.sum((alpha-alpha_old)**2) < tol:
                    break
            self.n_iter = n_iter
            self.alpha = alpha

            return self

        def decision_function(self, X):
            K_x = self.kernel_function_(X, self.X_train, **self.kernel_parameters)    
            return sigmoid(K_x.dot(self.alpha))


        def predict(self, X):
            decisions = self.decision_function(X)
            #print(decisions)
            predicted_classes = np.where(decisions < 0.5, -1, 1)
            return predicted_classes
        


    model=KernelLogisticRegression(lambd=0.01, kernel='polynomial', sigma=10, degree=5)
    print('')
    print(' Starting fitting kernal logistic model ...')
    
    print('')

    model.fit(X_train_,y_train_)

    print('')
    print(' ... predict  ...')  
    print('')
    predy=model.predict(X_test_)
    print('')
    print(' ... calculate the acurracy  ...')
    print('')
    acc = np.mean(predy==y_test_)
    print('')
    print('the accuracy = ' , acc)

    print(' ... predict on test data  ...')
    print('')
    fpred = model.predict(x_test)
    print('')
    # cross validation
    print(' ... Validate the model  ...')
    print('')
    kfc=k_fold_cross_validation(model)
    kfc.validate(x_train, y)

    print(' ... we are making submission  ...')
    print('')  
    
    

           
    def sigmoid(x):
        if (x>0).any():
            return 1 / (1 + np.exp(-x))               
        else:
            return np.exp(x) / (1 + np.exp(x)) 

    class KernelLogisticRegression(KernelMethodBase):
        def __init__(self, lambd=0.00001, **kwargs):
            self.lambd = lambd
            # Python 3: replace the following line by
            super().__init__(**kwargs)
            #super(KernelLogisticRegression, self).__init__(**kwargs)

        def fit(self, X, y, max_iter=1000, tol=0.0001):
            n, p = X.shape
            assert (n == len(y))

            self.X_train = X
            self.y_train = y

            K = self.kernel_function_(X, X, **self.kernel_parameters)

            # IRLS
            KRR = KernelRidgeRegression(
                lambd=2*self.lambd,
                kernel=self.kernel_name,
                **self.kernel_parameters
            )

        # Initialize
            alpha = np.zeros(n)

            for n_iter in range(max_iter):
                alpha_old = alpha
                m = K.dot(alpha_old)
                w = sigmoid(m) * sigmoid(-m)
                z = m + self.y_train / sigmoid(self.y_train * m)
                alpha = KRR.fit(self.X_train, z, sample_weights=w).alpha

                if np.sum((alpha-alpha_old)**2) < tol:
                    break

            self.n_iter = n_iter
            self.alpha = alpha

            return self

        def decision_function(self, X_test):
            K_x = self.kernel_function_(X_test, self.X_train, **self.kernel_parameters)

            return sigmoid(K_x.dot(self.alpha))

        def predict(self, X):
            probas = self.decision_function(X)
            predicted_classes = np.where(probas < 0.5, -1, 1)
            return predicted_classes

#     model=KernelLogisticRegression(lambd=0.01, kernel='polynomial', sigma=10, degree=5)
#     print('')
#     print(' Starting fitting kernal logistic model ...')
    
#     print('')

#     model.fit(X_train_,y_train_)

#     print('')
#     print(' ... predict  ...')  
#     print('')
#     predy=model.predict(X_test_)
#     print('')
#     print(' ... calculate the acurracy  ...')
#     print('')
#     acc = np.mean(predy==y_test_)
#     print('')
#     print('the accuracy = ' , acc)

#     print(' ... predict on test data  ...')
#     print('')
#     fpred = model.predict(x_test)
#     print('')
#     # cross validation
#     print(' ... Validate the model  ...')
#     print('')
#     kfc=k_fold_cross_validation(model)
#     kfc.validate(x_train, y)

#     print(' ... we are making submission  ...')
#     print('')
    #make submission
    #convert  y to 0 and 1 class
    output = []
    for i in range(len(fpred)):
        if fpred[i]==-1: 
            fpred[i]==0
            output.append(0)
        else:output.append(1)

    output  = np.array(output)

    # put the output in the form of dictionary and then data frames
    Id = np.array([i for i in range(len(output))])
    sub = {'Id':Id,'Bound':output}
    submision = pd.DataFrame(sub)
    
    print(' ... we are Done!  ...')
    print('')
    submision.to_csv('ssubmission1.csv',index=False)