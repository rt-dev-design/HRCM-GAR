import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import numpy as np
import random
from PIL import Image

from .windowing import get_windows
from .sampler import ListIteratorSampler
from .mix_deterministic_and_random_sampling import augment_with_copies_of_random_sampling

ACTIVITIES = ['2p-succ.', '2p-fail.-off.', '2p-fail.-def.',
              '2p-layup-succ.', '2p-layup-fail.-off.', '2p-layup-fail.-def.',
              '3p-succ.', '3p-fail.-off.', '3p-fail.-def.']


def read_ids(path):
    file = open(path)
    values = file.readline()
    values = values.split(',')[:-1]
    values = list(map(int, values))

    return values


def nba_read_annotations(path, seqs):
    # input: path to dataset, video ids
    # output: a effective dict/map: (vid, clip) -> label
    labels = {}
    group_to_id = {name: i for i, name in enumerate(ACTIVITIES)}

    for sid in seqs:
        annotations = {}
        with open(path + '/%d/annotations.txt' % sid) as f:
            for line in f.readlines():
                values = line[:-1].split('\t')
                file_name = values[0]
                fid = int(file_name.split('.')[0])

                activity = group_to_id[values[1]]

                annotations[fid] = {
                    'file_name': file_name,
                    'group_activity': activity,
                }
            labels[sid] = annotations

    return labels


def nba_all_frames(labels):
    # input : a effective dict/map: (vid, clip) -> label
    # output: a list of (sid, fid), or (vid, clip)
    frames = []

    for sid, anns in labels.items():
        for fid, ann in anns.items():
            frames.append((sid, fid))

    return frames


class NBADataset(data.Dataset):
    """
    Volleyball Dataset for PyTorch
    """
    def __init__(self, frames, anns, image_path, args, is_training):
        super(NBADataset, self).__init__()
        # frames = clips
        # anns = clip -> label
        # image_path = /path_to_dataset/videos
        # image_size = (1280, 720), configured in args
        # random_sampling is False for NBA
        # num_frame = number of frames sampled = 18 for NBA
        # preprocessing: resize and normalize
        
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
        return self.load_windows(self.sample_windows_for_clip(idx))

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
            import copy
            sampled_windows = copy.deepcopy(self.windows)
        else:
            assert False, 'Unrecognized window sampling method %s. Choose between sparse or dense.' % self.window_sampling_method
        
        output = []
        for i, window in enumerate(sampled_windows):
            w = []
            for j, frame in enumerate(window):
                w.append((vid, sid, frame))
            output.append(w)
        
        return output

    def load_windows(self, windows):
        # input: a list of windows
        # output: images and activity tensors
        images = []
        for i, window in enumerate(windows):
            images_in_this_window = []
            for j, (vid, sid, fid) in enumerate(window):
                fid = '{:06d}'.format(fid)
                img = Image.open(self.image_path + '/%d/%d/%s.jpg' % (vid, sid, fid))
                img = self.transform(img)
                images_in_this_window.append(img) 
            images.append(images_in_this_window)
        images = torch.from_numpy(np.array(images)).float()
        activity = torch.tensor(self.anns[vid][sid]['group_activity'], dtype=torch.long)
        return images, activity
 