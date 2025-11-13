import numpy as np

# Follow the tasks below to practice basic Python concepts.
# Write your code in between the dashed lines.
# Don't import additional packages. Numpy suffices.


# Task 1: Compute Output Size for 1D Convolution
# Instructions:
# Write a function that takes two one-dimensional numpy arrays (input_array, kernel_array) as arguments.
# The function should return the length of the convolution output (assuming no padding and a stride of one).
# The output length can be computed as follows:
# (input_length - kernel_length + 1)

# Your code here:
# -----------------------------------------------

def compute_output_size_1d(input_array, kernel_array):
    out_len = input_array.shape[0] - kernel.array.shape[0] +1 # compute output length
    return max(0,out_len) # in case kernel is larger than input, return 0 (bc convolution cannot happen in that scenario)
    # .shape[0] gives the length of 1D array
# -----------------------------------------------
# Example:
input_array = np.array([1, 2, 3, 4, 5])
kernel_array = np.array([1, 0, -1])
print(compute_output_size_1d(input_array, kernel_array))


# Task 2: 1D Convolution
# Instructions:
# Write a function that takes a one-dimensional numpy array (input_array) and a one-dimensional kernel array (kernel_array)
# and returns their convolution (no padding, stride 1).

# Your code here:
# -----------------------------------------------

def convolve_1d(input_array, kernel_array):
    # Tip: start by initializing an empty output array (you can use your function above to calculate the correct size).
    # Then fill the cells in the array with a loop.
    out_length = compute_output_size_1d(input_array, kernel_array)
    output = np.zeros(out_length) # array full of zeros with the right length

    kernel_size = len(kernel_array)

    for i in range(out_length): # loop over indices 
        window = input_array[i : i + kernel_size]
        output[i] = np.sum(window * kernel_array)
    
    return output
        
    
# -----------------------------------------------
# Another tip: write test cases like this, so you can easily test your function.
input_array = np.array([1, 2, 3, 4, 5])
kernel_array = np.array([1, 0, -1])
print(convolve_1d(input_array, kernel_array))

# Task 3: Compute Output Size for 2D Convolution
# Instructions:
# Write a function that takes two two-dimensional numpy matrices (input_matrix, kernel_matrix) as arguments.
# The function should return a tuple with the dimensions of the convolution of both matrices.
# The dimensions of the output (assuming no padding and a stride of one) can be computed as follows:
# (input_height - kernel_height + 1, input_width - kernel_width + 1)

# Your code here:
# -----------------------------------------------

def compute_output_size_2d(input_matrix, kernel_matrix):
    # get dimensions of input and kernel
    h, w = input_matrix.shape # height is nrow, width is n of columns
    kh, kw = kernel_matrix.shape

    out_h = h - kh + 1
    out_w = w - kw + 1

    return(out_h, out_w)

input_matrix = np.array([[1,2,3],
                         [4,5,6]])
kernel_matrix = np.array([[1,0,-1],
                          [0,1,-1]])
print(compute_output_size_2d(input_matrix, kernel_matrix))

kernel_matrix2 = np.array([[1,0],
                           [0,1]])
print(compute_output_size_2d(input_matrix, kernel_matrix2))

# -----------------------------------------------


# Task 4: 2D Convolution
# Instructions:
# Write a function that computes the convolution (no padding, stride 1) of two matrices (input_matrix, kernel_matrix).
# Your function will likely use lots of looping and you can reuse the functions you made above.

# Your code here:
# -----------------------------------------------
def convolute_2d(input_matrix, kernel_matrix):
    # Tip: same tips as above, but you might need a nested loop here in order to
    # define which parts of the input matrix need to be multiplied with the kernel matrix.
    out_h, out_w = compute_output_size_2d(input_matrix, kernel_matrix)
    kh, kw = kernel_matrix.shape

    output = np.zeros((out_h, out_w)) # initialize output matrix with zeros
    for i in range(out_h):
        for j in range(out_w):
            window = input_matrix[i : i + kh, j : j + kw]
            output[i, j] = np.sum(window * kernel_matrix)
    return output
    





# -----------------------------------------------
