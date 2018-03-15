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
        

if __name__ == "__main__":
    num = sys.argv[-1]
    if not num.isdigit():
        print "You need to specify a digit when running the script"
    else:
        e = Excercise(int(num))
        e.run()
