# ------------------------------------------------------------------------
# Reference:
# https://github.com/cvlab-epfl/social-scene-understanding/blob/master/volleyball.py
# https://github.com/wjchaoGit/Group-Activity-Recognition/blob/master/volleyball.py
# ------------------------------------------------------------------------
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import numpy as np
import random
from PIL import Image

from .windowing import get_windows
from .sampler import ListIteratorSampler
from .mix_deterministic_and_random_sampling import augment_with_copies_of_random_sampling

ACTIVITIES = ['r_set', 'r_spike', 'r-pass', 'r_winpoint', 'l_set', 'l-spike', 'l-pass', 'l_winpoint']


def volleyball_read_annotations(path, seqs, num_activities):
    labels = {}
    if num_activities == 8:
        group_to_id = {name: i for i, name in enumerate(ACTIVITIES)}
    # merge pass/set label    
    elif num_activities == 6:
        group_to_id = {'r_set': 0, 'r_spike': 1, 'r-pass': 0, 'r_winpoint': 2,
                       'l_set': 3, 'l-spike': 4, 'l-pass': 3, 'l_winpoint': 5}
    
    for sid in seqs:
        annotations = {}
        with open(path + '/%d/annotations.txt' % sid) as f:
            for line in f.readlines():
                values = line[:-1].split(' ')
                file_name = values[0]
                fid = int(file_name.split('.')[0])

                activity = group_to_id[values[1]]

                annotations[fid] = {
                    'file_name': file_name,
                    'group_activity': activity,
                }
            labels[sid] = annotations

    return labels


def volleyball_all_frames(labels):
    frames = []

    for sid, anns in labels.items():
        for fid, ann in anns.items():
            frames.append((sid, fid))

    return frames


class VolleyballDataset(data.Dataset):
    """
    Volleyball Dataset for PyTorch
    """
    def __init__(self, frames, anns, image_path, args, is_training=True):
        super(VolleyballDataset, self).__init__()

        self.windows = get_windows(args.clip_length, args.window_width, args.window_stride)
        self.sampler = ListIteratorSampler(self.windows)
        self.window_sampling_method = args.window_sampling_method
        self.ramdomness_for_sparse = args.ramdomness_for_sparse
        self.num_windows = args.num_windows
        
        self.is_training = is_training
        self.frames = frames

        if self.window_sampling_method == "sparse_with_mixed_deterministic_and_random":
            if self.is_training:
                f, w = augment_with_copies_of_random_sampling(self.frames, args.copies_of_fixed_stride, args.copies_of_random_sampling)
                self.frames = f
                self.whether_to_apply_randomness = w

        self.anns = anns
        self.image_path = image_path
        self.image_size = (args.image_width, args.image_height)
        self.num_total_frame = args.clip_length
        self.transform = transforms.Compose([
            transforms.Resize((args.image_height, args.image_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, idx):
        # frames = self.select_frames(self.frames[idx])
        # samples = self.load_samples(frames)
        # return samples
        windows = self.sample_windows_for_clip(idx)
        vid, cid = self.frames[idx]
        center_frame_number_str = self.anns[vid][cid]['file_name'].split('.')[0]
        windows = self.translate_frame_indices_into_real_file_names(windows, center_frame_number_str)
        return self.load_windows(windows)

    def __len__(self):
        return len(self.frames)
    
    def sample_windows_for_clip(self, idx):
        vid, sid = self.frames[idx]
        sampled_windows = []

        if self.window_sampling_method == 'sparse':
            sampled_windows = self.sampler.sample(self.num_windows, self.ramdomness_for_sparse if self.is_training else False)
        elif self.window_sampling_method == 'sparse_with_mixed_deterministic_and_random':
            sampled_windows = self.sampler.sample(self.num_windows, self.whether_to_apply_randomness[idx] if self.is_training else False)
        elif self.window_sampling_method == 'dense':
            raise NotImplementedError
        else:
            assert False, 'Unrecognized window sampling method %s. Choose between sparse or dense.' % self.window_sampling_method
        
        output = []
        for i, window in enumerate(sampled_windows):
            w = []
            for j, frame in enumerate(window):
                w.append((vid, sid, frame))
            output.append(w)
        
        return output

    def translate_frame_indices_into_real_file_names(self, windows, frame_num_str): 
        center = int(frame_num_str) 
        radius = (self.num_total_frame - 1) // 2
        assert 2 * radius + 1 == self.num_total_frame
        real_file_name_list = [i for i in range(center - radius, center + radius + 1)]
        assert len(real_file_name_list) == self.num_total_frame
        output = []
        for i, window in enumerate(windows):
            out = []
            for j, (v, c, f) in enumerate(window): 
                out.append((v, c, real_file_name_list[f]))
            output.append(out)
        return output

    def load_windows(self, windows):
        # input: a list of windows
        # output: images and activity tensors
        images = []
        for i, window in enumerate(windows):
            images_in_this_window = []
            for j, (vid, sid, fid) in enumerate(window):
                img = Image.open(self.image_path + '/%d/%d/%s.jpg' % (vid, sid, fid))
                img = self.transform(img)
                images_in_this_window.append(img) 
            images.append(images_in_this_window)
        images = torch.from_numpy(np.array(images)).float()
        activity = torch.tensor(self.anns[vid][sid]['group_activity'], dtype=torch.long)
        return images, activity

