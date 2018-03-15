#coding: utf-8
# from: https://www.machinelearningplus.com/101-numpy-exercises-python/
import numpy as np
import sys


class Excercise:
    def __init__(self, num):
        self.num = num

    def run(self):
        attr = "exec{}".format(self.num)
        getattr(self, attr)()

    def exec1(self):
        # Import numpy as np and see the version
        import numpy as np

    def exec2(self):
        # Create a 1D array of numbers from 0 to 9
        print np.arange(10)

    def exec3(self):
        # Create a 3×3 numpy array of all True’s
        print np.full((3, 3), True)
        #print np.ones((3, 3)).astype("bool")

    def exec4(self):
        # Extract all odd numbers from arr
        arr = np.arange(10)
        print arr[arr % 2 == 1]

    def exec5(self):
        # Replace all odd numbers in arr with -1
        arr = np.arange(10)
        arr[arr % 2 == 1] = -1
        print arr

    def exec6(self):
        # Replace all odd numbers in arr with -1 without changing arr
        arr = np.arange(10)
        out = np.where(arr % 2 == 1, -1, arr)
        print out
        print arr

    def exec7(self):
        # Convert a 1D array to a 2D array with 2 rows
        arr = np.arange(10)
        print arr.reshape((2, -1))

    def exec8(self):
        # Stack arrays a and b vertically
        a = np.arange(10).reshape(2,-1)
        b = np.repeat(1, 10).reshape(2,-1)
        print np.r_[(a, b)]
        print np.vstack((a, b))
        print np.concatenate([a, b])

    def exec9(self):
        # Stack the arrays a and b horizontally.
        a = np.arange(10).reshape(2,-1)
        b = np.repeat(1, 10).reshape(2,-1)
        print np.c_[(a, b)]
        print np.hstack((a, b))
        print np.concatenate([a, b], axis=1)

    def exec10(self):
        # Create the following pattern without hardcoding. Use only numpy functions and the below input array a.
        a = np.array([1,2,3])
        # output: array([1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3])
        print np.hstack((np.repeat(a, 3), np.tile(a, 3)))
        print np.r_[(np.repeat(a, 3), np.tile(a, 3))]

    def exec11(self):
        # Get the common items between a and b
        a = np.array([1,2,3,2,3,4,3,4,5,6])
        b = np.array([7,2,10,2,7,4,9,4,9,8])
        print np.intersect1d(a, b)

    def exec12(self):
        # From array a remove all items present in array b
        a = np.array([1,2,3,4,5])
        b = np.array([5,6,7,8,9])
        #print np.array([x for x in a if x not in b])
        print np.setdiff1d(a, b)

    def exec13(self):
        # Get the positions where elements of a and b match
        a = np.array([1,2,3,2,3,4,3,4,5,6])
        b = np.array([7,2,10,2,7,4,9,4,9,8])
        print np.where(a == b)

    def exec14(self):
        # Get all items between 5 and 10 from a.
        a = np.arange(15)
        print a[np.where(((a >= 5) & (a <= 10)))]
        print a[(a >= 5) & (a <= 10)]

    def exec15(self):
        # Convert the function maxx that works on two scalars, to work on two arrays.
        def maxx(x, y):
            """Get the maximum of two items"""
            if x >= y:
                return x
            else:
                return y
        print maxx(1, 5)
        pair_max = np.vectorize(maxx)
        a = np.array([5, 7, 9, 8, 6, 4, 5])
        b = np.array([6, 3, 4, 8, 9, 7, 1])
        print pair_max(a, b)

    def exec16(self):
        # Swap columns 1 and 2 in the array arr.
        arr = np.arange(9).reshape(3,3)
        print arr
        print arr[:, [1, 0, 2]]

    def exec17(self):
        # Swap rows 1 and 2 in the array arr:
        arr = np.arange(9).reshape(3,3)
        print arr
        print arr[[1, 0, 2]]

    def exec18(self):
        # Reverse the rows of a 2D array arr.
        arr = np.arange(9).reshape(3, 3)
        print arr
        print arr[::-1, :]

    def exec19(self):
        # Reverse the columns of a 2D array arr.
        arr = np.arange(9).reshape(3, 3)
        print arr
        print arr[:, ::-1]

    def exec20(self):
        # Create a 2D array of shape 5x3 to contain random decimal numbers between 5 and 10.
        R = 5
        C = 3
        print np.random.uniform(5, 10, R*C).reshape((R, -1))
        print np.random.uniform(5, 10, (R, C))

    def exec21(self):
        # Print or show only 3 decimal places of the numpy array rand_arr.
        rand_arr = np.random.random((5,3))
        np.set_printoptions(precision=3)
        print rand_arr

    def exec22(self):
        # Pretty print rand_arr by suppressing the scientific notation (like 1e10)
        np.random.seed(100)
        rand_arr = np.random.random([3,3])/1e3
        np.set_printoptions(suppress=True)
        print rand_arr

    def exec23(self):
        # Limit the number of items printed in python numpy array a to a maximum of 6 elements.
        a = np.arange(15)
        np.set_printoptions(threshold=6)
        print a

    def exec24(self):
        # Print the full numpy array a without truncating.
        a = np.arange(15)
        np.set_printoptions(threshold=6)
        print a
        np.set_printoptions(threshold=a.size)
        print a

    def exec25(self):
        # Import the iris dataset keeping the text intact.
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
        iris = np.genfromtxt(url, delimiter=',', dtype='object')
        print iris[:3]

    def exec26(self):
        # Extract the text column species from the 1D iris imported in previous question.
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
        iris_1d = np.genfromtxt(url, delimiter=',', dtype=None)
        print np.array([x[-1] for x in iris_1d])

    def exec27(self):
        # Convert the 1D iris to 2D array iris_2d by omitting the species text field.
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
        iris_1d = np.genfromtxt(url, delimiter=',', dtype=None)
        #print np.array([[x[0], x[1], x[2], x[3]] for x in iris_1d])
        print np.array([x.tolist()[:-1] for x in iris_1d])

    def exec28(self):
        # Find the mean, median, standard deviation of iris's sepallength (1st column)
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
        iris = np.genfromtxt(url, delimiter=',', dtype='object')
        sepallength = iris[:, 0].astype("float")
        print np.mean(sepallength), np.median(sepallength), np.std(sepallength)

    def exec29(self):
        # Create a normalized form of iris's sepallength whose values range exactly between 0 and 1 so that the minimum has value 0 and maximum has value 1.
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
        sepallength = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0])
        # we might need to cache the min to save the computation
        print (sepallength - sepallength.min()) / (sepallength.max() - sepallength.min())

    def exec30(self):
        # Compute the softmax score of sepallength.
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
        sepallength = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0])
        print (np.e**sepallength) / (np.e**sepallength).sum()

        

if __name__ == "__main__":
    num = sys.argv[-1]
    if not num.isdigit():
        print "You need to specify a digit when running the script"
    else:
        e = Excercise(int(num))
        e.run()
