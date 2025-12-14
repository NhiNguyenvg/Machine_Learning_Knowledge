import numpy as np
import tensorflow as tf


class_names = [
    'Pants', 'Pullover', 'Dress', 'Winter_Jacket', 'Sandal', 
    'Jacket',      'Sport_Shoe',   'Purse',  'Winter_Shoe'
]

#------------------------------------------------------------------------------
# Load the Dataset
#------------------------------------------------------------------------------
train_size = 0.8
test_size=0.2
dataset = np.load("data1/fashion_mnist_dataset.npz")


X = dataset["images"]
Y = dataset["labels"]
print("X_Shape: ", X.shape) #X_Shape:  (48200, 28, 28)
print("Y_Shape: ", Y.shape) #Y_Shape:  (48200,)

# shuffle dataset, because dataset now is with label 1,2... , it is now good to train
#np.random.shuffle(dataset)
index = np.arange(len(X))
print("len index: ", len(index))
np.random.shuffle(index)
print("index schuffle: ", index)

# Updated the X and Y with method shuffle 
X = X[index]
Y= Y[index]

# 80 train dataset, 20 test dataset
train_x= X[0:int(len(X)*train_size),:,:] 
test_x= X[int(len(X)*train_size):,:,:]

train_y = Y[0:int(len(Y)*train_size)]
test_y = Y[int(len(Y)*train_size):]

print("len train x:", len(train_x), "shape train_x:", train_x.shape) #len train x: 38560 shape train_x: (38560, 28, 28)
print("len test x:", len(test_x), "shape test x: ", test_x.shape) #len train x: 38560 shape train_x: (38560, 28, 28)

# reshape to NHWC format for initial convLayers
# -1 hier means, because shape at begin ist (38560, 28, 28) -> (38560, 28, 28,1), Neu khong cho -1 reshape cho nay se doi truc tiep 38560,28,28 -> 28,28,1 -> se xay ra loi
train_x = train_x.reshape(-1,28,28,1)
test_x = test_x.reshape(-1,28,28,1)

#------------------------------------------------------------------------------
# Build Model
#------------------------------------------------------------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, 3, input_shape=(28, 28, 1)),
    tf.keras.layers.ReLU(),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile with explicit metric name
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss='sparse_categorical_crossentropy',  # String format
    metrics=['accuracy']  # Simplified - will automatically use sparse_categorical_accuracy
)

print("\nModel compiled. Metrics:", model.metrics_names)
model.summary()
#------------------------------------------------------------------------------
# Train
#------------------------------------------------------------------------------
print("\n" + "="*70)
print("TRAINING")
print("="*70)

history = model.fit(
    train_x, train_y,
    epochs=10,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)
#------------------------------------------------------------------------------
# Evaluate
#------------------------------------------------------------------------------
print("\n" + "="*70)
print("EVALUATION")
print("="*70)

# Method 1: Using evaluate
eval_results = model.evaluate(test_x, test_y, verbose=1)
print(f"\nEvaluate results: {eval_results}")
print(f"Metric names: {model.metrics_names}")

if len(eval_results) == 2:
    test_loss, test_accuracy = eval_results
    print(f"\nTest Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
else:
    print(f"Unexpected number of metrics: {len(eval_results)}")

# Example: Predict the first 5 images in the test dataset
predictions = model.predict(test_x[:5])

# Output the predicted class for each image
predicted_classes = np.argmax(predictions, axis=1)

# Print the predicted class labels for the first 5 images
print("Predicted class labels:", predicted_classes)

# Print the actual class labels for the same images
print("Actual class labels:", test_y[:5])

# If you want to see the predicted probabilities for each class:
print("Predicted probabilities:", predictions)
    
model.save_weights("cnn.weights.h5") ;
