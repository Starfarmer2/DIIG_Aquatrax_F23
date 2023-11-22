import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(MLP, self).__init__()
        # First hidden layer
        self.fc1 = nn.Linear(input_size, hidden_size1)
        # Second hidden layer
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        # Output layer
        self.fc3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        # Apply ReLU activation function after each hidden layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Output layer does not have an activation function here
        # This can be changed depending on the problem
        x = self.fc3(x)
        return x

# Example usage
input_size = 10    # number of input features
hidden_size1 = 50  # number of neurons in the first hidden layer
hidden_size2 = 50  # number of neurons in the second hidden layer
output_size = 3    # number of output classes

# Create an instance of the MLP
mlp = MLP(input_size, hidden_size1, hidden_size2, output_size)

# Example input
example_input = torch.randn(1, input_size)  # batch size of 1

# Forward pass
output = mlp(example_input)
print(output)






# Back propagation
import torch.optim as optim

# Hypothetical dataset
# Assuming inputs are of size 10 and there are 3 classes
X_train = torch.randn(100, 10)  # 100 training examples
y_train = torch.randint(0, 3, (100,))  # 100 labels

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer (e.g., Stochastic Gradient Descent)
optimizer = optim.SGD(mlp.parameters(), lr=0.01)

# Training loop
for epoch in range(100):  # number of epochs
    for i in range(len(X_train)):
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = mlp(X_train[i])

        # Compute the loss
        loss = criterion(outputs.unsqueeze(0), y_train[i].unsqueeze(0))

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

print(mlp(torch.Tensor([1,2,3,4,5,6,7,0,0,0])))