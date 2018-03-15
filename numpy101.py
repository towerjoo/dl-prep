#coding: utf-8
# from: https://www.machinelearningplus.com/101-numpy-exercises-python/
import numpy as np
import sys
import PIL
from PIL import Image
import requests
import StringIO


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

    def exec31(self):
        # Find the 5th and 95th percentile of iris's sepallength
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
        sepallength = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0])
        #print np.percentile(sepallength, 5), np.percentile(sepallength, 95)
        print np.percentile(sepallength, [5, 95])

    def exec32(self):
        # Insert np.nan values at 20 random positions in iris_2d dataset
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
        iris_2d = np.genfromtxt(url, delimiter=',', dtype='object')
        # method 1: ensure 20 unique positions
        nans = np.full(iris_2d.size, True, dtype="bool")
        nans[:20] = False
        np.random.shuffle(nans)
        nans = nans.reshape((iris_2d.shape[0], -1))
        #print np.where(nans, iris_2d, np.nan)
        # method 2: not ensure
        r, c = iris_2d.shape
        iris_2d[np.random.choice(r, 20), np.random.choice(c, 20)] = np.nan
        print iris_2d

    def exec33(self):
        #  Find the number and position of missing values in iris_2d's sepallength (1st column)
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
        iris_2d = np.genfromtxt(url, delimiter=',', dtype='float')
        iris_2d[np.random.randint(150, size=20), np.random.randint(4, size=20)] = np.nan
        sepallength = iris_2d[:, 0]
        print np.isnan(sepallength).sum(), np.where(np.isnan(sepallength))

    def exec34(self):
        # Filter the rows of iris_2d that has petallength (3rd column) > 1.5 and sepallength (1st column) < 5.0
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
        iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])
        print iris_2d[(iris_2d[:, 2] > 1.5) & (iris_2d[:, 0] < 5.0)]
        print iris_2d[np.where((iris_2d[:, 2] > 1.5) & (iris_2d[:, 0] < 5.0))]

    def exec35(self):
        # Select the rows of iris_2d that does not have any nan value.
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
        iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])
        iris_2d[np.random.randint(150, size=20), np.random.randint(4, size=20)] = np.nan
        nan_rows = np.array([~np.any(np.isnan(row)) for row in iris_2d])
        #print iris_2d[nan_rows]
        # method 2
        print iris_2d[np.sum(np.isnan(iris_2d),axis=1) == 0]

    def exec36(self):
        # Find the correlation between SepalLength(1st column) and PetalLength(3rd column) in iris_2d
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
        iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])
        print np.corrcoef(iris_2d[:, 0], iris_2d[:, 2])[0, 1]

    def exec37(self):
        # Find out if iris_2d has any missing values.
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
        iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])
        iris_2d[np.random.randint(150, size=20), np.random.randint(4, size=20)] = np.nan
        print np.isnan(iris_2d).sum() > 0
        print np.isnan(iris_2d).any()

    def exec38(self):
        # Replace all ccurrences of nan with 0 in numpy array
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
        iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])
        iris_2d[np.random.randint(150, size=20), np.random.randint(4, size=20)] = np.nan
        print iris_2d[:5]
        # NOT change the original data
        print np.where(np.isnan(iris_2d), 0, iris_2d)[:5]
        # change the original data
        iris_2d[np.isnan(iris_2d)] = 0
        print iris_2d[:5]

    def exec39(self):
        # Find the unique values and the count of unique values in iris's species
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
        iris = np.genfromtxt(url, delimiter=',', dtype='object')
        names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')
        print np.unique(iris[:, -1], return_counts=True)

    def exec40(self):
        # Bin the petal length (3rd) column of iris_2d to form a text array, such that if petal length is:
        # Less than 3 --> 'small'
        # 3-5 --> 'medium'
        # '>=5 --> 'large'
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
        iris = np.genfromtxt(url, delimiter=',', dtype='object')
        names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')
        petallength = iris[:, 2].astype("float")
        print np.where(petallength < 3, "small", np.where(petallength >= 5, "large", "medium"))[:5]
        # or
        bins = np.digitize(petallength, [0, 3, 5, 10])
        label_map = {1: 'small', 2: 'medium', 3: 'large', 4: np.nan}
        cats = [label_map[x] for x in bins]
        print cats[:5]

    def exec41(self):
        # Create a new column for volume in iris_2d, where volume is (pi x petallength x sepal_length^2)/3
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
        iris_2d = np.genfromtxt(url, delimiter=',', dtype='object')
        names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')
        petallength = iris_2d[:, 2].astype("float")
        sepallength = iris_2d[:, 0].astype("float")
        print np.c_[iris_2d, np.pi * petallength * (sepallength ** 2) / 3][:5]

    def exec42(self):
        # Randomly sample iris's species such that setose is twice the number of versicolor and virginica
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
        iris = np.genfromtxt(url, delimiter=',', dtype='object')
        species = ['Iris-setosa','Iris-versicolor','Iris-virginica']
        species_out = np.random.choice(species, 150, p=[0.5, 0.25, 0.25])
        print np.unique(species_out, return_counts=True)

        # or
        probs = np.r_[np.linspace(0, .5, num=50), np.linspace(.501, .75, num=50), np.linspace(.751, 1., num=50)]
        index = np.searchsorted(probs, np.random.random(150))
        species_out = iris[:, -1][index]
        print np.unique(species_out, return_counts=True)

    def exec43(self):
        # What is the value of second longest petallength of species setosa
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
        iris = np.genfromtxt(url, delimiter=',', dtype='object')
        names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')
        petallength = iris[iris[:,-1] == 'Iris-setosa'][:, 2].astype("float")
        print np.unique(np.sort(petallength))[-2]

    def exec44(self):
        # Sort the iris dataset based on sepallength column.
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
        iris = np.genfromtxt(url, delimiter=',', dtype='object')
        names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')
        sepallength = iris[:, 0].astype("float")
        sorted_index = np.argsort(sepallength)
        print iris[sorted_index]

    def exec45(self):
        # Find the most frequent value of petal length (3rd column) in iris dataset.
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
        iris = np.genfromtxt(url, delimiter=',', dtype='object')
        names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')
        values, counter = np.unique(iris[:, 2], return_counts=True)
        print values[np.argsort(counter)[-1]]
        # or
        print values[np.argmax(counter)]

    def exec46(self):
        #  Find the position of the first occurrence of a value greater than 1.0 in petalwidth 4th column of iris dataset.
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
        iris = np.genfromtxt(url, delimiter=',', dtype='object')
        petalwidth = iris[:, 3].astype("float")
        print np.argwhere(petalwidth > 1.0)[0]
        # or
        print np.argmax(petalwidth > 1.0)

    def exec47(self):
        # From the array a, replace all values greater than 30 to 30 and less than 10 to 10.
        np.random.seed(100)
        a = np.random.uniform(1,50, 20)
        print a
        print np.where(a > 30, 30, np.where(a < 10, 10, a))
        # or use clip
        print np.clip(a, 10, 30)
        # change the original array
        a[a > 30] = 30
        a[a < 10] = 10
        print a

    def exec48(self):
        # Get the positions of top 5 maximum values in a given array a.
        np.random.seed(100)
        a = np.random.uniform(1,50, 20)
        print np.argsort(a)[-5:]

    def exec49(self):
        # Compute the counts of unique values row-wise.
        np.random.seed(100)
        arr = np.random.randint(1,11,size=(6, 10))
        out = None
        for r in arr:
            values = np.zeros((1, 10), dtype="int")
            v, c = np.unique(r, return_counts=True)
            values[0, v-1] = c
            if out is None:
                out = values
            else:
                out = np.r_[out, values]
        print out

    def exec50(self):
        # Convert array_of_arrays into a flat linear 1d array.
        arr1 = np.arange(3)
        arr2 = np.arange(3,7)
        arr3 = np.arange(7,10)
        array_of_arrays = np.array([arr1, arr2, arr3])
        print np.concatenate(array_of_arrays)
        # or
        print np.hstack(array_of_arrays)

    def exec51(self):
        # Compute the one-hot encodings (dummy binary variables for each unique value in the array)
        np.random.seed(101)
        arr = np.random.randint(1,4, size=6)
        values = np.unique(arr)
        encoded = np.zeros((arr.size, values.size))
        for row, a in enumerate(arr):
            col = np.where(values == a)
            encoded[row, col] = 1
        print encoded

    def exec52(self):
        # Create row numbers grouped by a categorical variable. Use the following sample from iris species as input.
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
        species = np.genfromtxt(url, delimiter=',', dtype='str', usecols=4)
        species_small = np.sort(np.random.choice(species, size=20))
        print species_small
        names, count = np.unique(species_small, return_counts=True)
        print np.hstack([np.arange(x) for x in count])

    def exec53(self):
        # Create group ids based on a given categorical variable. Use the following sample from iris species as input.
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
        species = np.genfromtxt(url, delimiter=',', dtype='str', usecols=4)
        species_small = np.sort(np.random.choice(species, size=20))
        print species_small
        names, count = np.unique(species_small, return_counts=True)
        print np.repeat(np.arange(count.size), count)

    def exec54(self):
        # Create the ranks for the given numeric array a.
        np.random.seed(10)
        a = np.random.randint(20, size=10)
        print a
        print np.argsort(np.argsort(a))

    def exec55(self):
        # Create a rank array of the same shape as a given numeric array a.
        np.random.seed(10)
        a = np.random.randint(20, size=[2,5])
        print(a)
        print np.argsort(np.argsort(a.ravel())).reshape(a.shape)

    def exec56(self):
        # Compute the maximum for each row in the given array.
        np.random.seed(100)
        a = np.random.randint(1,10, [5,3])
        print a
        print np.amax(a, axis=1)

    def exec57(self):
        # Compute the min-by-max for each row for given 2d numpy array.
        np.random.seed(100)
        a = np.random.randint(1,10, [5,3])
        print np.apply_along_axis(lambda r: r.min() / r.max(), arr=a.astype("float"), axis=1)

    def exec58(self):
        # Find the duplicate entries (2nd occurrence onwards) in the given numpy array and mark them as True. First time occurrences should be False.
        np.random.seed(100)
        a = np.random.randint(0, 5, 10)
        print('Array: ', a)
        #> Array: [0 0 3 0 2 4 2 2 2 2]
        mark = np.ones(a.size, dtype="bool")
        values, index = np.unique(a, return_index=True)
        mark[index] = False
        print mark


        # python approach
        marked = []
        for index, item in enumerate(a):
            a[index] = item in marked
            marked.append(item)
        print a.astype("bool")

    def exec59(self):
        # Find the mean of a numeric column grouped by a categorical column in a 2D numpy array
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
        iris = np.genfromtxt(url, delimiter=',', dtype='object')
        names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')
        species = np.unique(iris[:, -1])
        index = 0 # sepallength
        index = 1 # sepalwidth
        #index = 2 # petallength
        #index = 3 # petalwidth
        print np.vstack([[v, iris[iris[:, -1] == species[i]][:, index].astype("float").mean()] for i, v in enumerate(species)])

    def exec60(self):
        # Import the image from the following URL and convert it to a numpy array.
        # Import image from URL
        URL = 'https://upload.wikimedia.org/wikipedia/commons/8/8b/Denali_Mt_McKinley.jpg'
        response = requests.get(URL)

        # Read it as Image
        I = Image.open(StringIO.StringIO(response.content))

        # Optionally resize
        I = I.resize([150,150])

        # Convert to numpy array
        arr = np.asarray(I)

        # Optionaly Convert it back to an image and show
        im = PIL.Image.fromarray(np.uint8(arr))
        Image.Image.show(im)

            
            

if __name__ == "__main__":
    num = sys.argv[-1]
    if not num.isdigit():
        print "You need to specify a digit when running the script"
    else:
        e = Excercise(int(num))
        e.run()
