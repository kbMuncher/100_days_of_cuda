
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

# Step 1: Set the original (true) values
true_m = 2.5
true_c = 1.0

# Generate random x values
np.random.seed(0)
x = np.linspace(0, 10, 50)
noise = np.random.randn(50)  # random noise
y = true_m * x + true_c + noise  # true y with noise

# Step 2: Initialize m and c for gradient descent
m = 0.0
c = 0.0
learning_rate = 0.01
epochs = 1000
n = float(len(x))

# Step 3: Gradient Descent
for _ in range(epochs):
    y_pred = m * x + c
    error = y_pred - y
    cost = (error ** 2).mean()

    # Gradients
    m_grad = (2/n) * np.dot(error, x)
    c_grad = (2/n) * error.sum()

    # Update weights
    m -= learning_rate * m_grad
    c -= learning_rate * c_grad

# Final predicted values
y_pred = m * x + c

# Step 4: Visualization
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Noisy data')
plt.plot(x, true_m * x + true_c, color='green', linestyle='--', label='True line')
plt.plot(x, y_pred, color='red', label='Predicted line (Gradient Descent)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression using Gradient Descent')
plt.legend()
plt.grid(True)
plt.show()

# Output learned values
print(f"Learned slope (m): {m:.3f}")
print(f"Learned intercept (c): {c:.3f}")
