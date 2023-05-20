import numpy as np

# Create a clean data set of a specified size
class Sinusoids:
    def __init__(self, DATA_SIZE, ARRAY_SIZE) -> None:
        self.data_size = DATA_SIZE
        self.array_size = ARRAY_SIZE
        pass

    def create_data(self):
        x_vals = [np.linspace(0, np.random.randint(1, 20)*np.pi, self.array_size)
              for i in range(self.data_size)]
        y_vals = [np.sin(x) for x in x_vals]
        classification = ["Clean" for x in x_vals]

        entry = {
            "X": x_vals,
            "Y": y_vals,
            "Classification": classification
        }

        return entry

# Create a noisy array from an array of input x-values
def apply_noise(xarr, ARRAY_SIZE):
    return [np.sin(x) + np.random.normal(0, .5, ARRAY_SIZE) for x in xarr]

# Apply the fourier transform to array
def apply_ft(yarr):
    return np.abs(np.fft.rfft(yarr))
