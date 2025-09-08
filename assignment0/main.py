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

"""
# Problem 2

- Generate a set of $x$ values ranging from $0$ to $100$ with an increment of $0.1$ using NumPy
- Compute the corresponding $y$ values using the function $y=sin(x)$
- Plot the sine wave using Matplotlib, and add appropriate labels for the $x$-axis, $y$-axis, and a title for the plot
- Save the plot as a PNG file named `sine_wave.png`
"""
def problem2():
    xs = np.arange(0, 100.1, 0.1)
    ys = np.sin(xs)

    ax = plt.axes()
    ax.set_title("Sine Wave")
    ax.set_xlabel("Input")
    ax.set_ylabel("Output")
    ax.plot(xs, ys)
    plt.savefig("sine_wave.png")
    plt.close()

"""
# Problem 3

- Create two NumPy arrays:
    - One for $x$ values ranging from $0$ to $100$ with an increment of $1$
    - Another for $y$ values that represent a quadratic function $y=0.5x^2+2x+1$
- Plot the quadratic function using Matplotlib with appropriate labels and a legend
- Add gridlines to the plot and display it with a line style of your choice
- Save the plot as a PDF file named `quadratic_function.pdf`
"""
def problem3():
    xs = np.arange(0, 101)
    ys = np.array([0.5 * x**2 + 2*x + 1 for x in xs])
    
    ax = plt.axes()
    ax.set_title("Quadratic")
    ax.set_xlabel("Input")
    ax.set_ylabel("Output")
    ax.plot(xs, ys, label="0.5x^2+2x+1", color='orange')
    ax.legend()
    ax.grid()
    plt.savefig("quadratic_function.pdf")
    plt.close()

def main():
    problem1()
    problem2()
    problem3()
if __name__ == "__main__":
    main()
