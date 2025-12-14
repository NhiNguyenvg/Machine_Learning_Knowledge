# keras cnn demo on mnist
# Author: A.Gepperth, UAS Fulda, 2022
import numpy as np ;
import requests ;
import os, sys ;
import matplotlib.pyplot as plt ;
import tensorflow as tf ;


if __name__ == "__main__":
  ## download MNIST if not present in current dir!
  if os.path.exists("./mnist.npz") == False:
    print ("Downloading MNIST...") ;
    fname = 'mnist.npz'
    url = 'http://www.gepperth.net/alexander/downloads/'
    r = requests.get(url+fname)
    open(fname , 'wb').write(r.content)
  
  ## load data into numpy array and reshape appropriately
  data = np.load("mnist.npz")
  traind = data["arr_0"] ;
  testd = data["arr_1"] ;
  trainl = data["arr_2"] ;
  testl = data["arr_3"] ;
  print ("Train data shape: ", traind.shape, "test data shape:", testd.shape) ; # Train data shape:  (60000, 28, 28) test data shape: (10000, 784)
  print ("Train label shape: ", trainl.shape, "test label shape:", testl.shape) ;# Train label shape:  (60000, 10) test label shape: (10000, 10)
  traind = traind.reshape(-1,784)
  testd = testd.reshape(-1,784)
      
  # reshape to NHWC format for initial convLayers!
  traind = traind.reshape(-1,28,28,1)
  testd = testd.reshape(-1,28,28,1)
    
  # This is a complete DNN, should get ~98% on MNIST test set
  model = tf.keras.Sequential() ;
  model.add(tf.keras.layers.Conv2D(64, 3)) ; # 26,26,64
  model.add(tf.keras.layers.ReLU()) ; # 
  model.add(tf.keras.layers.MaxPool2D()) ; # 13,13,64
  model.add(tf.keras.layers.Reshape(target_shape=(13*13*64,))) ;     # 25*64        
  model.add(tf.keras.layers.Dense(100)) ; # 
  model.add(tf.keras.layers.ReLU()) ; # 
  model.add(tf.keras.layers.Dense(10)) ; # 
  model.add(tf.keras.layers.Softmax()) ;
    
  model.compile(optimizer=tf.keras.optimizers.Adam(0.01), 
      loss = tf.keras.losses.CategoricalCrossentropy(), 
      metrics = [tf.keras.metrics.CategoricalAccuracy()]) ;
  model.fit(traind,trainl,epochs=1, batch_size=100) ;
  model.save_weights("cnn.weights.h5") ;
    
  # evaluate classification error on MNIST test data
  class_acc = model.evaluate(x=testd,y=testl) ;  # compute f(X) 
  print ("classification error on test is ", class_acc) ;
  
  samples = testd[0:10] ;
  # compute raw model output on some test samples, __call__ only works for small batches!
  output = model(samples) ;
  print ("Decisions on samples 0-9", output) ;
  # compute raw model output on all test data using batching: .predict()
  full_output = model.predict(testd) ;
        

    
  
          
    
