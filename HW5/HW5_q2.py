import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC

df = pd.read_csv('q2_train.csv')

X = df.iloc[:, 0:2].values
y = df.iloc[:, 2].values
r,_ = df.shape



#Types of SVC kernels:
kernel_list = ["linear", "poly", "rbf", "sigmoid"]


#Train and predict four different SVC kernels:
for i in range(len(kernel_list)):
    #Make a linear (or some other type) of SVM
    model = SVC(kernel=kernel_list[i])
    model.fit(X,y)

    #Function to plot the SVM boundaries and lines - cool!
    def plot_svc_decision_function(model, ax=None, plot_support=True):
        if ax is None:
            ax=plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        #Create grid to evaluate model
        x = np.linspace(xlim[0], xlim[1], 30)
        y = np.linspace(ylim[0], ylim[1], 30)
        Y, X = np.meshgrid(y, x)
        xy = np.vstack([X.ravel(), Y.ravel()]).T
        P = model.decision_function(xy).reshape(X.shape)

        #Plot decision boundary and margins
        ax.contour(X, Y, P, colors='k', levels=[-1,0,1], alpha=0.5, linestyles=['--','-','--'])

        #Plot support vectors
        if plot_support:
            ax.scatter(model.support_vectors_[:, 0],
                model.support_vectors_[:, 1],
                s=300, linewidth=1, facecolors='none')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)


    #Call the function to show points and SV boundaries
    plt.scatter(X[:,0], X[:,1], c=y, s=50, cmap='winter')
    plot_svc_decision_function(model)
    print("(4,5), (2,2), (0,-1), (-3,-3)")
    print(model.predict([[4,5],[2,2],[0,-1],[-3,-3]]))
    print()
    print()
    print()

    plt.show()