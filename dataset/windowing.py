def get_windows(
    video_length: int,
    window_width: int,
    stride: int,
    start_index: int = None,
    end_index: int = None,
) -> list[list[int]]:
    """
    Generates a list of windows of frame indices for a video sequence.

    This function creates a sliding window over a sequence of a given length.
    It's useful for tasks like temporal feature extraction in video analysis,
    allowing you to process a video in overlapping chunks.

    Args:
        video_length: The total number of frames in the video sequence.
        window_width: The number of frames in each window.
        stride: The step size to move the window forward after each step.
        start_index: The frame index where the windowing process should begin.
                     If set to None, it defaults to the start of the video (index 0).
        end_index: The last possible frame index that can be included in any window.
                   If set to None, it defaults to the end of the video (video_length - 1).

    Returns:
        A list of lists, where each inner list contains the frame indices
        for a single window. Returns an empty list if no valid windows can be formed.

    Raises:
        ValueError: If any of the input parameters are invalid (e.g.,
                    negative values, out-of-bounds indices, or a start_index
                    greater than the end_index).
    """
    # 1. Set default values for start and end indices if they are not provided
    if start_index is None:
        start_index = 0
    if end_index is None:
        end_index = video_length - 1

    # 2. Validate all inputs to ensure they are logical and within bounds
    if video_length <= 0:
        raise ValueError("video_length must be a positive integer.")
    if window_width <= 0:
        raise ValueError("window_width must be a positive integer.")
    if stride <= 0:
        raise ValueError("stride must be a positive integer.")
    if window_width > video_length:
        raise ValueError("window_width cannot be greater than video_length.")
    if start_index < 0 or start_index >= video_length:
        raise ValueError(f"start_index {start_index} is out of bounds for video_length {video_length}.")
    if end_index < 0 or end_index >= video_length:
        raise ValueError(f"end_index {end_index} is out of bounds for video_length {video_length}.")
    if start_index > end_index:
        raise ValueError(f"start_index ({start_index}) cannot be greater than end_index ({end_index}).")

    windows = []
    current_start = start_index

    # 3. Slide the window across the specified range of the video
    # The loop continues as long as a full window can be formed without its
    # last element exceeding the specified end_index.
    while current_start + window_width - 1 <= end_index:
        # Create a list of frame indices for the current window
        window = list(range(current_start, current_start + window_width))
        windows.append(window)

        # Move to the start position for the next window
        current_start += stride

    return windows


# This block demonstrates how to use the function when the script is run directly.
if __name__ == "__main__":
    # --- Example 1: Basic usage with defaults ---
    # A 20-frame video, with a 5-frame window that slides by 2 frames each time.
    print("--- Example 1: Basic Usage (Defaults) ---")
    video_len = 72
    win_width = 5
    win_stride = 3
    print(f"Video Length: {video_len}, Window Width: {win_width}, Stride: {win_stride}\n")
    result1 = get_windows(video_len, win_width, win_stride)
    print("Generated Windows:")
    for w in result1:
        print(f"  {w}")

    print("\n" + "="*50 + "\n")
