import matplotlib.pyplot as plt

# Read data from the file
file_path = 'weights_history.txt'  # Replace with the actual path to your file
data1 = []
data2 = []
data3 = []

with open(file_path, 'r') as file:
    for line in file:
        values = [float(val) for val in line.strip().split(',')]
        data1.append(values[0])  # Extract the first element from each line
        data2.append(values[2])
        data3.append(values[3])
        # Plotting
plt.plot(data1, label='weight1', color='blue')
plt.plot(data2, label='weight2', color='green')
plt.plot(data3, label='weight3', color='red')
plt.xlabel('Training iterations')
plt.ylabel('The value of parameters')
plt.title('Changes in 3 weights Over iterations')
plt.savefig('weights_plot.png')
plt.show()




