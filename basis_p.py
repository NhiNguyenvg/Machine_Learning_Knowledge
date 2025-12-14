import numpy as np
import matplotlib.pyplot as plt

print("=== 2.1 Basics numpy ===\n")

# a) First array - 2D array with specific structure
x = np.array([5,1,4,2,8,4])
x = np.array([x[0],x[2],x[3]])
print(x)

# b) Second array - appears to be a 4x4 array with sequential numbers
b = np.arange(1,10,1).reshape(3, 3)
z= np.array([1,0,9])
x= np.add(b,z)
print(b)
print(z)
print(x)

# c) Third array - appears to show transformation/manipulation
c = np.array([1,2,3,4,4,11,99,5])
print("c)")
print(c[c>=5])

#----------------------------------------------------------------------------------#

print("\n=== 2.2 Advanced numpy ===\n")

x = np.random.uniform(0,5,size=(1,100))
print(x)
T = x.T

#-----------------------------------------------------------------------------------#

print(" 2.3 Mathplotlib")

# Create 100 support (x/y) pairs in range [0, 3]
x1 = np.linspace(0, 3, 100)
x2 =  np.linspace(0, 1, 100)

# Create figure with 2 subplots as shown in fig. 3
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left subplot: Plot of f(x)
# Based on the sketch, appears to be a simple function
y1 = x1  # or could be another function
ax1.plot(x1, y1, 'b-', linewidth=2)
ax1.set_title('Plot of f(x)')
ax1.set_xlabel('x in [0, 3] range')
ax1.set_ylabel('f(x)')
ax1.grid(True)

# Right subplot: Scatter plot of f(x) = x²
y2 = x2 ** 2
ax2.scatter(x2, y2, c='red', s=20, alpha=0.6)
ax2.set_title('Scatter plot of f(x) = x²')
ax2.set_xlabel('x in [0, 3] range')
ax2.set_ylabel('f(x) = x²')
ax2.grid(True)

plt.tight_layout()
plt.savefig('matplotlib_functions.png', dpi=300, bbox_inches='tight')
print("Matplotlib figure saved as 'matplotlib_functions.png'")
plt.show()

#-----------------------------------------------------------------------------------#
print("Gradient Decent")


