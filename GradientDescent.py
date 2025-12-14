import numpy as np


step_e = 0.1

def function_f(x1,x2):
    f_x = np.square(x1)+np.square(x2)
    print(f_x)
    return f_x 

def gradient_fx(x1,x2):
    grad_fx= np.array([[2*x1,2*x2]])
    print(grad_fx)
    return grad_fx

def gradient_descent(start_x, num_steps=3):
    x = start_x.copy()  # [[1, 3]]
    result = np.zeros((num_steps, 2))  
    result[0] = x[0] 
    
    for i in range(1, num_steps):
        grad = gradient_fx(x0[0,0],x0[0,1])
        x = x - step_e * grad
        result[i] = x[0]
        print(f"Step {i}: x = {x}, f(x) = {function_f(x0[0,0],x0[0,1])}")
    
    return result
        
x0= np.array([[1,3]])
print(x0.shape)
print(x0[0,1])

gradient_fx(x0[0,0],x0[0,1])

print(gradient_descent(x0,3))


    
    