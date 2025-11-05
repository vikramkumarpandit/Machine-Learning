import numpy as np

class Conv3x3:
    """
    A simple implementation of a Convolutional Layer using 3x3 filters.
    This layer performs a 2D convolution operation (without padding).
    """

    def __init__(self, num_filters):
        """
        Initialize the convolution layer.

        Parameters:
        -----------
        num_filters : int
            The number of 3x3 filters in this layer.
        """
        self.num_filters = num_filters

        # Each filter is a 3x3 matrix; we create num_filters of them.
        # Dividing by 9 reduces variance (helps keep activations small).
        self.filters = np.random.randn(num_filters, 3, 3) / 9


    def iterate_regions(self, image):
        """
        Generates all possible 3x3 regions of the input image using VALID padding.

        Parameters:
        -----------
        image : numpy.ndarray
            2D array (grayscale image)

        Yields:
        -------
        (im_region, i, j)
            im_region : 3x3 sub-region of the input image
            i, j : top-left coordinates of this region
        """
        h, w = image.shape

        # Slide a 3x3 window over every valid region of the image
        for i in range(h - 2):
            for j in range(w - 2):
                # Extract the 3x3 patch (region)
                im_region = image[i:(i + 3), j:(j + 3)]
                # Yield the region along with its coordinates
                yield im_region, i, j


    def forward(self, input):
        """
        Perform a forward pass of the convolution layer.

        Parameters:
        -----------
        input : numpy.ndarray
            2D input image (grayscale)

        Returns:
        --------
        output : numpy.ndarray
            3D array of shape (h-2, w-2, num_filters)
            Each (i, j) contains a vector of activations (one per filter)
        """
        h, w = input.shape

        # Initialize output (reduced size due to valid padding)
        output = np.zeros((h - 2, w - 2, self.num_filters))

        # For each 3x3 region, compute convolution with all filters
        for im_region, i, j in self.iterate_regions(input):
            # Element-wise multiplication and sum over (3,3)
            # axis=(1,2) → sum across rows & columns for each filter
            output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))

        return output


# -------------------------------------------------------------
# ✅ Example Usage
# -------------------------------------------------------------
if __name__ == "__main__":
    # Create a convolution layer with 2 filters
    conv = Conv3x3(num_filters=2)

    # Example 4x4 grayscale image
    image = np.array([
        [1, 2, 3, 0],
        [0, 1, 2, 3],
        [3, 0, 1, 2],
        [2, 3, 0, 1]
    ])

    # Perform the forward pass
    output = conv.forward(image)

    print("Input shape:", image.shape)
    print("Output shape:", output.shape)
    print("Output feature maps:\n", output)
