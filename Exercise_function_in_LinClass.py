# runs with python 3.10.19

import numpy as np ;
import requests ;
import os ;
import matplotlib.pyplot as plt ;

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
  traind = traind.reshape(60000,28,28)
  print(traind)
  
  fig, axes = plt.subplots(1,10)
  
  for i, axes in enumerate(axes.ravel()):
    axes.imshow(traind[i].reshape(28,28))

#--------------------------------------------------------------
# Mathplotlib first 5 Images of traind
#--------------------------------------------------------------

fig, axes = plt.subplots(1,5)

for i, axes in enumerate(axes.ravel()):
    axes.imshow(traind[i])
    
#-------------------------------------------------------------
# ReLU
#-------------------------------------------------------------

x  = np.array([[-1,1,5],[1,1,2]])
def ReLU(x):
    return (x > 0.) * x
ReLU(x)

#-------------------------------------------------------------
# Softmax
#-------------------------------------------------------------
def Softmax(x):
    exp_x= np.exp(x)
    sum_exp_x = exp_x.sum(axis = 1)
    print(sum_exp_x)
# Van chua xongs

Softmax(x)