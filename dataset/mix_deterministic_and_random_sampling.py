import random

def augment_with_copies_of_random_sampling(clip_ids: list, copies_of_fixed_stride: int, copies_of_randomness: int) -> tuple[list, dict]:
    augmented_list = []
    sampling_dict = {}
    
    # Add the original data with stride sampling
    for _ in range(copies_of_fixed_stride):
        for clip_id in clip_ids:
            augmented_list.append(clip_id)
            sampling_dict[len(augmented_list) - 1] = False

    # Add copies with random sampling
    for _ in range(copies_of_randomness):
        for clip_id in clip_ids:
            augmented_list.append(clip_id)
            sampling_dict[len(augmented_list) - 1] = True

    # Shuffle the entire augmented list to mix original and augmented data
    # along with their corresponding sampling booleans
    combined_data = list(zip(augmented_list, [sampling_dict[i] for i in range(len(augmented_list))]))
    random.shuffle(combined_data)

    shuffled_clips, shuffled_sampling = zip(*combined_data)

    new_sampling_dict = {i: val for i, val in enumerate(shuffled_sampling)}

    return list(shuffled_clips), new_sampling_dict

# Example usage:
# original_ids = [101, 102, 103, 104]
# new_ids, sampling_map = augment_with_copies_of_random_sampling(original_ids)
# print("New clip IDs:", new_ids)
# print("Sampling map:", sampling_map)