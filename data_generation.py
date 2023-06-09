"""Creates sinusoid class and methods for applying noise and fourier transforms to datasets"""

import numpy as np

class Sinusoids:
    """Create sinusoids
    """

    def __init__(self, data_size, array_size) -> None:
        self.data_size = data_size
        self.array_size = array_size

    def create_data(self):
        """Create Data

        Returns:
            dict: xval, yval, labels

        """
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
    def apply_noise(self, entry):
        """Apply random noise to signla array

        Args:
            xarr (numpy array): input x values to aply noise transform to
            array_size (int): size of array

        Returns:
            list: list of y values with noise applied
        """
        return [np.sin(x) + np.random.normal(0, .5, self.array_size) for x in entry["X"]]

    # Apply the fourier transform to array
    def apply_ft(self, entry):
        """Apply Fourier Transform to signal array

        Args:
            yarr (numpy array): signal array

        Returns:
            numpy array: fourier transform of signal array
        """
        return np.abs(np.fft.rfft(entry["Y"]))
