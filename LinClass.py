import numpy as np ;
import requests ;
import os ;
import matplotlib.pyplot as plt;

class LinClass(object):

  ## allocate weight and bias arrays and store ref 2 train data/labels
  def __init__(self, n_in, n_out, X, T):
    self.W = np.ones([n_in,n_out]) * 0.1 ;
    self.b = np.ones([1,n_out])* 0.1 ;
    self.X = X ;
    self.T = T ;
    self.N = X.shape[0] ;

  # nomal cross-entropy loss. Fill in your code here!
  def loss(self, Y, T): 
    return 0.0 ;

  def dLdb(self,Y,T):
    return np.zeros(self.b.shape) ;

  # fill in your code here!
  def dLdW(self, X, Y, T):
    return np.zeros(self.W.shape) ;
    
  # softmax: fill in your code here!
  def S(self,X):
    return X ;

  # dummy model, fill in your code!
  def f(self,X):
    return np.zeros([X.shape[0],10]) ;

  # performs a single gradient descent step
  # works with any size of X and T
  def train_step(self, X, T, eps):
    Y = self.f(X) ;
    loss = self.loss(Y,T) ;
    dLdb = self.dLdb(Y,T) ;      
    dLdW = self.dLdW(X, Y, T) ;  
    self.b -= eps * dLdb ;       ## b(i+1) = b(i) - eps * gradL
    self.W -= eps * dLdW ;       ## same
    return loss ;

  # perform multiple gradient descent steps and display loss. Does it go down??
  def train(self,max_it,eps):
    for it in range(0,max_it):
      print ("iut=", it, "loss=", self.train_step(self.X, self.T, eps)) ;

if __name__ == "__main__":

  ## download MNIST if not present in current dir!
  if os.path.exists("./mnist.npz") == False:
    print ("Downloading MNIST...") ;
    fname = 'mnist.npz'
    url = 'http://www.gepperth.net/alexander/downloads/'
    r = requests.get(url+fname)
    open(fname , 'wb').write(r.content)
  
  ## read it into 
  data = np.load("mnist.npz")
  traind = data["arr_0"] ;
  trainl = data["arr_2"] ;
  traind = traind.reshape(60000,784)

  # train the linear classifier with 300 MNIST samples. Try with
  # 1000, 10000, 60000 and see your memory go down the drain :-)
  lc = LinClass(784,10, traind[0:300], trainl[0:300]) ;

  # ------------------
  # here, test your softmax, f, and loss implementations!!
  # ....

  # ------------------

  # the actual training 
  lc.train(500, 0.01) ;

  # simple way of testing the classifier
  print(lc.f(traind[0:10]).argmax(axis=1),trainl[0:10].argmax(axis=1))
      
    
    
  #--------------------------------------------------------------------
  # Exercise 3.1
  #---------------------------------------------------------------------
  print(traind.shape)
  train_01 = traind[0:5,:]
  print(train_01.shape)
  # plot first 5 images of digit 0 and 1
for i in range(5):
    img = train_01[i,:].reshape(28,28)
    plt.subplot(1,5,i+1)
    plt.imshow(img, cmap='gray')
    plt.axis('off')

  #  Compute the mean over each image in traind and display all means in a scatter plot
means = np.mean(traind, axis=1)
plt.figure()
plt.scatter(range(len(means)), means)
plt.xlabel('Image Index')
plt.ylabel('Mean Pixel Value')
plt.title('Mean Pixel Values of MNIST Images')
plt.show()

#Compute the “average image” and display it
average_image = np.mean(traind, axis=0).reshape(28, 28)
plt.figure()
plt.imshow(average_image, cmap='gray')
plt.title('Average Image of MNIST Dataset')
plt.axis('off')
plt.show()

#---------------------------------------------------------------------
# Exercise 3.2
#---------------------------------------------------------------------

