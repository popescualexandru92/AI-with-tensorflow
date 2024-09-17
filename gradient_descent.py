import tensorflow as tf
import matplotlib.pyplot as plt

x = tf.Variable(3.0,tf.int32)

with tf.GradientTape() as tape:
  
  y = x * x

dy_dx = tape.gradient(y, x)



# Initialize a random value for our initial x
x = tf.Variable(tf.constant([0],tf.float64))

print("Initializing x={}".format(x.numpy()))

learning_rate = 0.05 # learning rate for SGD
history = []
# Define the target value
x_f = 10

# We will run SGD for a number of iterations. At each iteration, we compute the loss,
#   compute the derivative of the loss with respect to x, and perform the SGD update.
for i in range(100):
  with tf.GradientTape() as tape:
    loss = (x - x_f)**2 # "forward pass": record the current loss on the tape
  # loss minimization using gradient tape
  grad = tape.gradient(loss, x) # compute the derivative of the loss with respect to x
  new_x = x - learning_rate*grad # sgd update
  x.assign(new_x) # update the value of x
  history.append(x.numpy()[0])
    
# Plot the evolution of x as we optimize towards x_f!
print(history)
plt.plot(history)
plt.plot([0, 100],[x_f,x_f])
plt.legend(('Predicted', 'True'))
plt.xlabel('Iteration')
plt.ylabel('x value')
plt.show()