import numpy as np

class MaxPool2:
    """
    A Max Pooling layer using a pool size of 2x2.
    This layer reduces the height and width of the input by half
    while keeping the same number of filters (depth).
    """

    def iterate_regions(self, image):
        """
        Generates non-overlapping 2x2 image regions to pool over.

        Parameters:
        - image: a 3D numpy array with shape (h, w, num_filters)

        Yields:
        - im_region: 2x2 region of the image for pooling
        - i, j: position indices in the output
        """
        h, w, _ = image.shape
        new_h = h // 2
        new_w = w // 2

        # Loop over every 2x2 region in the input image
        for i in range(new_h):
            for j in range(new_w):
                # Extract a 2x2 region
                im_region = image[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
                yield im_region, i, j

    def forward(self, input):
        """
        Performs a forward pass of the maxpool layer using the given input.

        Parameters:
        - input: a 3D numpy array with shape (h, w, num_filters)

        Returns:
        - output: a 3D numpy array with shape (h/2, w/2, num_filters)
        """
        self.last_input = input

        h, w, num_filters = input.shape
        output = np.zeros((h // 2, w // 2, num_filters))

        # Apply max pooling operation
        for im_region, i, j in self.iterate_regions(input):
            # Take the maximum value within each 2x2 block for every filter
            output[i, j] = np.amax(im_region, axis=(0, 1))

        return output


# -------------------------------
# 🔹 Example Usage and Testing
# -------------------------------

if __name__ == "__main__":
    # Example 4x4 input image with 2 filters (depth = 2)
    input_data = np.array([
        [[1, 5], [2, 6], [3, 7], [4, 8]],
        [[5, 9], [6, 10], [7, 11], [8, 12]],
        [[9, 13], [10, 14], [11, 15], [12, 16]],
        [[13, 17], [14, 18], [15, 19], [16, 20]]
    ])

    print("Input shape:", input_data.shape)
    print("Input:\n", input_data)

    # Create maxpool layer
    pool = MaxPool2()

    # Perform forward pass
    output = pool.forward(input_data)

    print("\nOutput shape:", output.shape)
    print("Output after Max Pooling:\n", output)
