import numpy as np
import matplotlib.pyplot as plt

"""
# Problem 1

- Create a 1D NumPy array of shape $1{\\times}5$ with random values drawn from a uniform distribution
- Compute the mean and standard deviation of the array
- Reshape the array into a 2D array with $5$ rows and $1$ column
- Add $5$ to each element in the array and print the result
- Compute the dot product of this reshaped array with itself
"""
def problem1():
    values = np.random.random(5)
    print(f"Values: {values}")

    mean = np.mean(values)
    standard_deviation = np.std(values)
    print(f"Mean: {mean}")
    print(f"Standard Deviation: {standard_deviation}")

    values2 = np.reshape(values, (5, 1))
    print(f"Reshaped values:\n {values2}")

    values2 += 5
    print(f"Adjusted values:\n {values2}")

    # Type annotation to make LSP shut up
    # Have to transpose or else we get an error
    dot: float = np.dot(values2.transpose(), values2)[0][0]
    print(f"Dot product: {dot}")

def main():
    problem1()
if __name__ == "__main__":
    main()
