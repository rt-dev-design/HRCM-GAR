from .volleyball import *
from .nba import *


TRAIN_SEQS_VOLLEY = [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54]
VAL_SEQS_VOLLEY = [0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51]
TEST_SEQS_VOLLEY = [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]


def build_zim_dataset(args):
    # In this method, arguably in the whole repository,
    # a data/annotations variable is effectively a map: clip -> label,
    # a frames variable is a list of clip names in the form of tuples (video, clip),
    # in this repository, especially in the realm of datasets and data loaders,
    # file = frame = clip, seq = video, roughly
    if args.dataset == 'volleyball':
        data_path = args.data_path + "Volleyball_dataset"
        image_path = data_path + "/videos"

        train_data = volleyball_read_annotations(image_path, TRAIN_SEQS_VOLLEY + VAL_SEQS_VOLLEY, args.num_classes)
        train_frames = volleyball_all_frames(train_data)

        test_data = volleyball_read_annotations(image_path, TEST_SEQS_VOLLEY, args.num_classes)
        test_frames = volleyball_all_frames(test_data)

        train_set = VolleyballDataset(train_frames, train_data, image_path, args, is_training=True)
        test_set = VolleyballDataset(test_frames, test_data, image_path, args, is_training=False)

    elif args.dataset == 'nba':
        data_path = args.data_path + 'NBA_dataset'
        image_path = data_path + "/videos"

        train_ids = read_ids(data_path + "/train_video_ids")
        test_ids = read_ids(data_path + "/test_video_ids")

        train_data = nba_read_annotations(image_path, train_ids)
        train_frames = nba_all_frames(train_data)

        test_data = nba_read_annotations(image_path, test_ids)
        test_frames = nba_all_frames(test_data)

        train_set = NBADataset(train_frames, train_data, image_path, args, is_training=True)
        test_set = NBADataset(test_frames, test_data, image_path, args, is_training=False)
    else:
        assert False, "Unsupported dataset: {}. Choose between volleyball or nba.".format(args.dataset)

    print("%d train samples and %d test samples" % (len(train_frames), len(test_frames)))

    return train_set, test_set
