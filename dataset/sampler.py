import random
import math

class ListIteratorSampler:
    """
    A class that provides both iteration and sampling functionalities for a list.

    The iterator part allows for traversing the list from start to end.
    The sampler part, which operates independently of the iterator's state,
    can sample elements from partitioned segments of the list, either
    deterministically or randomly.
    """

    def __init__(self, data_list):
        """
        Initializes the ListIteratorSampler with a list.

        Args:
            data_list (list): The list of elements to iterate and sample from.
        
        Raises:
            TypeError: If the provided input is not a list.
        """
        if not isinstance(data_list, list):
            raise TypeError("Input data must be a list.")
        self.data = data_list
        self.length = len(data_list)
        self._iterator_index = 0

    # --- Standard Iterator Methods ---

    def hasNext(self):
        """
        Checks if there are more elements to iterate through.

        Returns:
            bool: True if there are more elements, False otherwise.
        """
        return self._iterator_index < self.length

    def next(self):
        """
        Returns the next element in the list and advances the iterator.

        Returns:
            The next element in the list.

        Raises:
            StopIteration: If the end of the list is reached.
        """
        if not self.hasNext():
            raise StopIteration("No more elements to iterate over.")
        
        element = self.data[self._iterator_index]
        self._iterator_index += 1
        return element

    # --- Pythonic Dunder Methods for Iteration ---

    def __iter__(self):
        """
        Makes the object iterable, allowing it to be used in for loops.
        This resets the iterator index for each new iteration loop.
        """
        self._iterator_index = 0
        return self

    def __next__(self):
        """
        Implements the `next()` function for the Python iterator protocol.
        """
        # This calls our custom next() method, which includes the StopIteration logic
        return self.next()

    # --- Sampler Method ---

    def sample(self, sampling_num, use_random=False):
        """
        Samples elements from the list by dividing it into segments.

        This method operates independently of the iterator's current state.

        Args:
            sampling_num (int): The number of elements to sample, which also
                                defines the number of segments.
            use_random (bool, optional): If True, a random element is chosen
                                       from each segment. If False (default),
                                       the first element of each segment is chosen.

        Returns:
            list: A new list containing the sampled elements.

        Raises:
            ValueError: If sampling_num is not a positive integer or is greater
                        than the length of the list.
        """
        if not isinstance(sampling_num, int) or sampling_num <= 0:
            raise ValueError("The number of samples must be a positive integer.")

        if sampling_num > self.length:
            raise ValueError("The number of samples cannot be greater than the list length.")

        sampled_elements = []
        # Calculate the floating-point size of each segment
        segment_size = self.length / sampling_num

        for i in range(sampling_num):
            # Determine the start and end indices for the current segment
            start_index = min(self.length, max(0, math.floor(i * segment_size)))
            end_index = max(start_index, min(self.length, math.floor((i + 1) * segment_size)))

            if use_random:
                # Select a random index within the segment's bounds.
                # random.randrange is exclusive of the stop value, which is perfect
                # as end_index is the start of the next segment.
                sample_index = random.randrange(start_index, end_index)
                sampled_elements.append(self.data[sample_index])
            else:
                # Sample the first element in the segment
                sampled_elements.append(self.data[start_index])

        return sampled_elements

# --- Example Usage ---
if __name__ == "__main__":
    # # Create a sample list from 0 to 19
    # my_data = list(range(20))
    # iterator_sampler = ListIteratorSampler(my_data)

    # print("--- Iterator Demonstration ---")

    # # 1. Using the hasNext() and next() methods with a while loop
    # print("\n1. Iterating with hasNext() and next():")
    # while iterator_sampler.hasNext():
    #     print(iterator_sampler.next(), end=" ")
    # print("\n")

    # # 2. Using the class directly in a for loop (more Pythonic)
    # # The __iter__ method automatically resets the index for the new loop
    # print("2. Iterating with a for loop (re-iterating):")
    # for item in iterator_sampler:
    #     print(item, end=" ")
    # print("\n")

    # print("--- Sampler Demonstration ---")
    
    # # 3. Deterministic sampling (first element of each segment)
    # # List length = 20, sampling_num = 5. Segments are of size 4.
    # # Segments: [0-3], [4-7], [8-11], [12-15], [16-19]
    # # Expected output: [0, 4, 8, 12, 16]
    # print("\n3. Sampling 5 elements (deterministic):")
    # deterministic_sample = iterator_sampler.sample(sampling_num=5, use_random=False)
    # print(f"   Result: {deterministic_sample}")

    # # 4. Randomized sampling (random element from each segment)
    # print("\n4. Sampling 5 elements (randomized):")
    # random_sample = iterator_sampler.sample(sampling_num=5, use_random=True)
    # print(f"   Result: {random_sample} (will vary on each run)")
    
    # # 5. Example with a list length that is not perfectly divisible
    # data_10 = list(range(10))
    # sampler_10 = ListIteratorSampler(data_10)
    # # List length = 10, sampling_num = 3. Segments size = 3.33
    # # Segments: [0-2], [3-5], [6-9]
    # # Expected deterministic output: [0, 3, 6]
    # print("\n5. Sampling 3 from a list of 10 (deterministic):")
    # non_divisible_sample = sampler_10.sample(sampling_num=3, use_random=False)
    # print(f"   Result: {non_divisible_sample}")
    
    # print("\n6. Sampling 3 from a list of 10 (randomized):")
    # non_divisible_random = sampler_10.sample(sampling_num=3, use_random=True)
    # print(f"   Result: {non_divisible_random}")

    from windowing import get_windows
    windows = get_windows(72, 5, 2)
    sampler = ListIteratorSampler(windows)
    for item in sampler:
        print(item)
    
    print("==============")
    print(sampler.sample(4, use_random=False))
    print(sampler.sample(4, use_random=True))
    
    list1 = [12, 3, 4]
    list2 = list1
    list2[2] = 8
    print(list1)
    print(list2)