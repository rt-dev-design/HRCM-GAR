import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
from sklearn.metrics import confusion_matrix

import os
import time
import argparse

from dataset.dataset import build_zim_dataset
from models.zim import build_zim
from util.utils import *
from util.logger import build_logger
from torch.utils.tensorboard import SummaryWriter
from util.experiment_setting_hash import combined_hash
from util.effective_step_size_statistics import *

parser = argparse.ArgumentParser(description='Zim Group Activity Recognition')

# dataset and data loading
parser.add_argument('--dataset', default='volleyball', type=str, help='volleyball or nba')
parser.add_argument('--data_path', default='./Dataset/', type=str, help='path to the directory containing the dataset directory')
parser.add_argument('--image_width', default=1280, type=int, help='image width to resize to')
parser.add_argument('--image_height', default=720, type=int, help='image height to resize to')
parser.add_argument('--clip_length', default=21, type=int, help='number of frames to consider around the annotated frame in Volleyball video clips')
parser.add_argument('--num_classes', default=6, type=int, help='number of activity classes')
parser.add_argument('--num_workers', default=6, type=int, help='number of workers for the data loader')
# windowing
parser.add_argument('--window_width', default=3, type=int, help='number of frames per window')
parser.add_argument('--num_windows', default=7, type=int, help='number of windows per clip to be fed into the network when using sparse sampling')
parser.add_argument('--window_stride', default=1, type=int, help='stride for the sliding window')
parser.add_argument('--window_sampling_method', default='sparse_with_mixed_deterministic_and_random', type=str, help='window sampling method, sparse, sparse_with_mixed_deterministic_and_random, or dense')
parser.add_argument('--ramdomness_for_sparse', default='False', type=str2bool, help='whether to use uniform randomness within each segment for sparse sampling during training, but not during testing')
parser.add_argument('--copies_of_fixed_stride', default=1, type=int, help='copies of fixed stride sampling during training for sparse_with_mixed_deterministic_and_random')
parser.add_argument('--copies_of_random_sampling', default=1, type=int, help='copies of random sampling during training for sparse_with_mixed_deterministic_and_random')

# model building parameters
# model initialization parameters are in the training parameters section
parser.add_argument('--zim_type', default="basic", type=str, help='ZimBasic or ZimFull')

# CNN backbone, ResNet18
parser.add_argument('--backbone', default='resnet18', type=str, help='choose from various cnn backbones from torchvision')
parser.add_argument('--dilation', default='False', type=str2bool, help='use dilation or not; see code for specific usage')
parser.add_argument('--use_pretrained_cnn', default='True', type=str2bool, help='use pretrained cnn weights to initialize or not')
parser.add_argument('--scale_selection_from_cnn', default="[2, 4]", type=str, help='a string representation of a list of integers for scale selection')
# position embedding for CNN feature maps
parser.add_argument('--position_embedding', default='sine', type=str, help='sine or learned position embeddings')
parser.add_argument('--normalize_position_embedding', default='True', type=str2bool, help='whether to normalize position embedding')

# Zim head
default_hidden_dim = 256
default_nhead = default_hidden_dim // 64
default_dim_feedforward = default_hidden_dim * 4
default_activation = 'gelu'
parser.add_argument('--activation_for_all_of_zim_head', default=default_activation, type=str, help="activation throughout the network except the cnn part")
# adaptors
parser.add_argument('--use_channel_attention_in_adaptors', default="False", type=str2bool, help="whether to use channel attention before channel align in adaptors")
parser.add_argument('--residual_connection_in_channel_attentions', default="False", type=str2bool, help="whether to use residual connection in channel attentions")
parser.add_argument('--use_bn_for_adaptors', default='True', type=str2bool, help='whether to use batch normalization in adaptors')
# the dimension of all kinds of tokens throughout the rest of the network, which is Transforemer-based
parser.add_argument('--hidden_dim', default=default_hidden_dim, type=int, help='transformer channel dimension')
# token encoders for all scales
parser.add_argument('--num_tokens', default=12, type=int, help='window size')
parser.add_argument('--token_encoder_nhead', default=default_nhead, type=int, help='window encoder nhead')
parser.add_argument('--token_encoder_norm_first', default='True', type=str2bool, help='whether to use pre norm or post norm in Transformers')
parser.add_argument('--token_encoder_nlayers', default=1, type=int, help='window encoder nlayers')
parser.add_argument('--token_encoder_dim_feedforward', default=default_dim_feedforward, type=int, help='window encoder dim_feedforward')
parser.add_argument('--token_encoder_dropout', default=0.1, type=float, help='window encoder dropout')
parser.add_argument('--token_encoder_return_intermediate', default='False', type=str2bool, help='window encoder return_intermediate')

# spatial temporal enhancers' d_model, dim_feedforward, nhead, norm fisrt, dropout, and activation follow those of token encoders
# west - window encoder spatial temporal
# csst - clip scale spatial temporal
parser.add_argument('--west_num_time_enc_layers', default=1, type=int, help='')
parser.add_argument('--west_num_space_enc_layers', default=1, type=int, help='')
parser.add_argument('--west_num_time_dec_layers', default=1, type=int, help='')
parser.add_argument('--west_num_space_dec_layers', default=1, type=int, help='')
parser.add_argument('--use_clip_scale_st', default='True', type=str2bool, help='whether to use clip scale spatial temporal enhancement')
parser.add_argument('--csst_num_time_enc_layers', default=1, type=int, help='')
parser.add_argument('--csst_num_space_enc_layers', default=1, type=int, help='')
parser.add_argument('--csst_num_time_dec_layers', default=1, type=int, help='')
parser.add_argument('--csst_num_space_dec_layers', default=1, type=int, help='')
parser.add_argument('--use_time_positional', default='False', type=str2bool, help='whether to use time positional encoding in the st enhancers')
parser.add_argument('--use_space_positional', default='False', type=str2bool, help='whether to use space positional encoding in the st enhancers')

# the pooling method for both of the two spots of grid feature pooling in the network
parser.add_argument('--pooling_method', default='moe', type=str, help='mean, max, attn, mean_max, moe')
parser.add_argument('--moe_num_experts', default=4, type=int, help='number of experts in moe pooling')
parser.add_argument('--mean_residual_connection_for_pooling', default='True', type=str2bool, help='whether to use mean residual connection for pooling')
parser.add_argument('--use_noise_gating_in_moe', default='True', type=str2bool, help='whether to apply a random noise to the logits in moe gates')
parser.add_argument('--use_ffn_in_aggregation', default='True', type=str2bool, help='whether to use ffn after attentions in aggregation modules')
parser.add_argument('--scale_down_std_for_gate_and_noise_weights', default='True', type=str2bool, help="whether to scale down moe gate weights and noise at initialization as an attempt for better performance and stability")

# initialization or checkpoint loading
parser.add_argument('--load_checkpoint', default='False', type=str2bool, help='whether to load a checkpoint for training')
parser.add_argument('--checkpoint_path', default='', type=str, help='checkpoint path')
parser.add_argument('--std_for_init', default=0.04, type=float, help='std for initializing weights')
parser.add_argument('--pooling_query_init_std', default=0.04, type=float, help='std for initializing pooling query')

# training parameters
parser.add_argument('--random_seed', default=1, type=int, help='random seed set in all sorts of library code for reproduction')
parser.add_argument('--max_epochs', default=15, type=int, help='max number of epochs to train for')
parser.add_argument('--train_batch', default=2, type=int, help='train batch size')

parser.add_argument('--max_lr', default=1e-4, type=float, help='target learning rate for the one cycle scheduler')
parser.add_argument('--max_lr_backbone', default=1e-4, type=float, help='target learning rate for finetuning the backbone')
# parser.add_argument('--starting_lr', default=1e-6, type=float, help='starting learning rate for the one cycle scheduler')
# parser.add_argument('--ending_lr', default=1e-8, type=float, help='ending learning rate for the one cycle scheduler')
parser.add_argument('--pct_start', default=0.1, type=float, help='pct_start for the one cycle scheduler')

parser.add_argument('--beta_1', default=0.90, type=float, help='beta_1 for Adam')
parser.add_argument('--beta_2', default=0.99, type=float, help='beta_2 for Adam')
parser.add_argument('--eps', default=1e-8, type=float, help='eps for Adam')

parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay')
parser.add_argument('--gradient_clipping', default='False', type=str2bool, help='whether to use gradient clipping')
parser.add_argument('--max_norm', default=1.0, type=float, help='max norm used in gradient clipping')
parser.add_argument('--dropout', default=0.1, type=float, help='general dropout rate if no module specific dropout is used')

# testing parameters
parser.add_argument('--test_freq', default=1, type=int, help='how often in terms of epochs to test during training')
parser.add_argument('--test_batch', default=2, type=int, help='test batch size')
parser.add_argument('--test_before_train', default='False', type=str2bool, help='whether to test first right before any training')
parser.add_argument('--run_test', default='False', type=str2bool, help="whether to run this test")

# GPU and hardware related parameters
parser.add_argument('--device', default='0', type=str, help='CUDA_VISIBLE_DEVICES, as well as the actual GPU indices the program will see')
parser.add_argument('--developing_using_very_little_gpu', default='False', type=str2bool, help='whether to launch a dummy development experiment using only a little GPU')

# monitoring and bookkeeping
parser.add_argument('--why_what_how_of_this_experiment', default="", type=str, help='add description of this experiment')

# lr finder
parser.add_argument('--find_lr', default='False', type=str2bool, help='whether to run to find the optimal learning rate')
parser.add_argument('--lr_finder_type', default='fastai', type=str, help='fastai or leslie_smith')

args = parser.parse_args()
try:
    args.scale_selection_from_cnn = [int(item) for item in args.scale_selection_from_cnn.strip('[]').split(',')]
    args.num_scales = len(args.scale_selection_from_cnn)
except ValueError:
    print("Error: The provided CNN scale selection string is not a valid list of integers.")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.device

best_mca = 0.0
best_mpca = 0.0
best_mca_epoch = 0
best_mpca_epoch = 0
time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
exp_name = '%s_Zim_%s' % (args.dataset, time_str)
save_path = './result/%s' % exp_name
args.experiment_name = exp_name
args.save_path = save_path
logger = build_logger("logger", save_path, "log.txt", use_this_logger_for_global_exceptions=True)
batch_logger = build_logger("batch_logger", save_path, "batch_log.txt")
effective_step_size_logger = build_logger("effective_step_size_logger", save_path, "effective_size_log.txt")
writer = SummaryWriter(save_path + "/tensorboard")
append_text_to_file(save_path, "description.txt", args.why_what_how_of_this_experiment)


def main():
    set_random_seeds(args.random_seed)

    train_set, test_set = build_zim_dataset(args)
    if args.developing_using_very_little_gpu:
        steal_only_a_little_gpu(train_set)
        steal_only_a_little_gpu(test_set)
    train_loader = data.DataLoader(train_set, batch_size=args.train_batch, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    test_loader = data.DataLoader(test_set, batch_size=args.test_batch, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = torch.nn.DataParallel(build_zim(args)).cuda()

    hook_handler = None
    
    criterion = nn.CrossEntropyLoss().cuda()
    
    cnn_backbone_parameters = [p for name, p in model.named_parameters() if 'module.zim_backbone.cnn' in name]
    zim_head_parameters = [p for name, p in model.named_parameters() if 'module.zim_backbone.cnn' not in name]
    # all lr hyperparameters here are only nominal and formal but invalid because learning rate is scheduled by the scheduler below
    optimizer = torch.optim.AdamW([
        {'params': cnn_backbone_parameters, 'lr': args.max_lr_backbone},
        {'params': zim_head_parameters, 'lr': args.max_lr},
    ], args.max_lr, betas=(args.beta_1, args.beta_2), eps=args.eps, weight_decay=args.weight_decay)
    div_factor = 100
    final_div_factor = 10000
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer, max_lr=[args.max_lr_backbone, args.max_lr], epochs=args.max_epochs, steps_per_epoch=steps_per_epoch,
        pct_start=args.pct_start, cycle_momentum=False, div_factor=div_factor, final_div_factor=final_div_factor
    )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epochs * len(train_loader))

    logger.info("arguments:")
    logger.info(args)
    logger.info("model:")
    logger.info('Number of parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    logger.info(model)

    # initialization
    if args.load_checkpoint:
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        # scheduler must be loaded before optimizer
        scheduler.load_state_dict(checkpoint['scheduler'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info("Model loaded from %s at epoch %d" % (args.checkpoint_path, start_epoch))
    else:
        start_epoch = 1
        initialize_model_parameters(model, args)
        logger.info('Initialized model parameters using the scheme defined in the method initialize_model_parameters and the hyperparameters for pretrained weights')

    logger.info("experiment setting hash: " + combined_hash(args, model, ignore_in_ns=IGNORED_ARGS_FOR_HASHING))

    if args.find_lr:
        from torch_lr_finder import LRFinder
        import matplotlib.pyplot as plt
        if args.lr_finder_type == "fastai":
            start_lr = 1e-7
            end_lr = 10
        elif args.lr_finder_type == "leslie_smith":
            end_lr = 1e-2
            start_lr = end_lr / 10
        else:
            raise NotImplementedError

        optimizer = torch.optim.Adam(
            model.parameters(), lr=start_lr, 
            betas=(args.beta_1, args.beta_2), eps=args.eps, weight_decay=args.weight_decay
        )
        lr_finder = LRFinder(model, optimizer, criterion)
        lr_finder.range_test(train_loader, start_lr=start_lr, end_lr=end_lr, 
            val_loader=None if args.lr_finder_type == "fastai" else test_loader, 
            step_mode="exp" if args.lr_finder_type == "fastai" else "linear",
        )
        lr_finder.plot(log_lr=True if args.lr_finder_type == "fastai" else False)
        lr_finder.reset()
        plt.savefig(args.save_path + '/lr_loss.png')
        exit()

    if args.test_before_train or args.run_test:
        logger.info("Testing the just-initialized model before any training")
        test_log = validate(test_loader, model, criterion, 0)
        logger.info('accuracy: %.2f%%, mean-acc: %.2f%%, loss: %.4f, time: %.1fs' % (test_log['group_acc'], test_log['mean_acc'], test_log['loss'], test_log['time']))
        writer.add_scalars("loss", {
            'test': test_log['loss'],
        }, 0)
        writer.add_scalars("accuracy", {
            'test': test_log['group_acc'],
        }, 0)
        if args.run_test: exit()

    for epoch in range(start_epoch, args.max_epochs + 1):
        # if epoch == 11: break
        logger.info('Training at epoch %d/%d' % (epoch, args.max_epochs))
        train_log = train(train_loader, model, criterion, optimizer, scheduler, hook_handler, epoch)
        logger.info('accuracy: %.2f%%, loss: %.4f, time: %.1fs, current lr: %s' % (train_log['group_acc'], train_log['loss'], train_log['time'], scheduler.get_last_lr()))

        if epoch % args.test_freq == 0:
            logger.info('Testing at epoch %d' % (epoch))
            test_log = validate(test_loader, model, criterion, epoch)
            logger.info('accuracy: %.2f%%, mean-acc: %.2f%%, loss: %.4f, time: %.1fs' % (test_log['group_acc'], test_log['mean_acc'], test_log['loss'], test_log['time']))
            logger.info('So far, best MCA %.2f%% occurred at epoch %d.' % (test_log['best_mca'], test_log['best_mca_epoch']))
            logger.info('So far, best MPCA %.2f%% occurred at epoch %d.' % (test_log['best_mpca'], test_log['best_mpca_epoch']))

            writer.add_scalars("loss", {
                'train': train_log['loss'],
                'test': test_log['loss'],
            }, epoch)
            writer.add_scalars("accuracy", {
                'train': train_log['group_acc'],
                'test': test_log['group_acc'],
            }, epoch)

            if epoch == test_log['best_mca_epoch'] or epoch == test_log['best_mpca_epoch']:
                state = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }
                result_path = save_path + '/epoch%d_%.2f%%.pth' % (epoch, test_log['group_acc'])
                torch.save(state, result_path)
                logger.info("Saved checkpoint to %s at epoch %d." % (result_path, epoch))
    
    writer.close()


def train(train_loader, model, criterion, optimizer, scheduler, hook_handler, epoch):
    """Train for one epoch on the training set"""
    epoch_timer = Timer()
    batch_timer = Timer()
    losses = AverageMeter()
    accuracies = AverageMeter()
    num_batches = len(train_loader)

    model.train()
    for i, (images, activity) in enumerate(train_loader):
        batch_size = images.shape[0]
        images = images.cuda()                                      
        activity = activity.cuda()            
        # [B, N, W, 3, H, W], [B,] -> [B, C]
        score = model(images)
        loss = criterion(score, activity)
        group_acc = accuracy(score, activity)
        losses.update(loss, batch_size)
        accuracies.update(group_acc, batch_size)
        optimizer.zero_grad()
        loss.backward()
        if args.gradient_clipping:
            nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        optimizer.step()
        scheduler.step()
        batch_logger.info("batch %d/%d in epoch %d, loss: %.8f, accuracy: %.8f, lr: %s, time: %.2fs" % (i + 1, num_batches, epoch, loss, group_acc, scheduler.get_last_lr(), batch_timer.timeit()))

    train_log = {
        'epoch': epoch,
        'time': epoch_timer.timeit(),
        'loss': losses.avg,
        'group_acc': accuracies.avg * 100.0
    }
    return train_log


@torch.no_grad()
def validate(test_loader, model, criterion, epoch):
    global best_mca, best_mpca, best_mca_epoch, best_mpca_epoch
    epoch_timer = Timer()
    batch_timer = Timer()
    num_batches = len(test_loader)
    losses = AverageMeter()
    accuracies = AverageMeter()
    true = []
    pred = []
    
    model.eval()
    for i, (images, activity) in enumerate(test_loader):
        batch_size = images.shape[0]
        images = images.cuda()
        activity = activity.cuda()
        score = model(images)
        true = true + activity.tolist()
        pred = pred + torch.argmax(score, dim=1).tolist()
        # calculate loss
        loss = criterion(score, activity)
        # measure accuracy and record loss
        group_acc = accuracy(score, activity)
        losses.update(loss, batch_size)
        accuracies.update(group_acc, batch_size)
        batch_logger.info("batch %d/%d in epoch %d, loss: %.8f, accuracy: %.8f, time: %.2fs" % (i + 1, num_batches, epoch, loss, group_acc, batch_timer.timeit()))
    
    acc = accuracies.avg * 100.0
    confusion = confusion_matrix(true, pred)
    mean_acc = np.mean([confusion[i, i] / confusion[i, :].sum() for i in range(confusion.shape[0])]) * 100.0

    if acc > best_mca:
        best_mca = acc
        best_mca_epoch = epoch
    if mean_acc > best_mpca:
        best_mpca = mean_acc
        best_mpca_epoch = epoch

    test_log = {
        'time': epoch_timer.timeit(),
        'loss': losses.avg,
        'group_acc': acc,
        'mean_acc': mean_acc,
        'best_mca': best_mca,
        'best_mpca': best_mpca,
        'best_mca_epoch': best_mca_epoch,
        'best_mpca_epoch': best_mpca_epoch,
    }

    return test_log


def initialize_model_parameters(model, args):
    initialization_scheme_save_path = args.save_path + '/model_initialization'

    module_dict = {}
    for module_name, module in model.named_modules():
        module_dict[module_name] = module
    
    for pname, param in model.named_parameters():
        append_text_to_file(initialization_scheme_save_path, "all_parameters.txt", pname + ": " + str(param.shape))

    # default general initialization if no module specific initialization is provided
    general_initialization_method = lambda w: nn.init.trunc_normal_(w, mean=0.0, std=args.std_for_init)
    # general_initialization_method = nn.init.kaiming_normal_

    # loaded pretrained weights for ResNet18
    # so no ops here for the CNN

    # module.zim_backbone.adaptors
    # Why module iterator here? Because the adaptors are essentially simple linear layers, 
    # and BatchNorms in adaptors will take care of themselves.
    adaptor_initialization_method = general_initialization_method
    for module_name, module in module_dict.items():
        if 'module.zim_backbone.adaptors' in module_name:
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                adaptor_initialization_method(module.weight)
                append_text_to_file(initialization_scheme_save_path, "adaptors.txt", module_name + ".weight: " + adaptor_initialization_method.__name__)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    append_text_to_file(initialization_scheme_save_path, "adaptors.txt", module_name + ".bias: " + nn.init.zeros_.__name__)

    # module.zim_backbone.window_encoders
    # The first for loop is for matrix initialization,
    # and the second for loop is for vector initialization.
    # CRITICAL WARNING: If we do not use the parameters() iterator to iterate through all matrix parameters in the module,
    # there will be some that are not accessible and hence not initialized with the modules() iterator, 
    # for example, input projects of all attention modules, which are important to be initialized to behave well.
    # There are still some biases that are not visited even with the second loop, but they will take care of themselves by default zeroing.
    # There are switches in the hyperparameters for toggling on and off ffns in the attentions for aggregation,
    # noise gating in moe modules, and spatial temporal positional encoding in st enhancers.
    # The initialization for these switched modules is implicitly taken care of in the code below and can be
    # checked out in the logs.
    window_encoder_initialization_method = general_initialization_method
    for pname, param in module_dict['module.zim_backbone.window_encoders'].named_parameters():
        append_text_to_file(initialization_scheme_save_path, "all_parameters_in_window_encoders.txt", pname)
        if param.dim() > 1:
            window_encoder_initialization_method(param)
            append_text_to_file(initialization_scheme_save_path, "window_encoders.txt", pname + ": " + window_encoder_initialization_method.__name__)

    for module_name, module in module_dict.items():
        if 'module.zim_backbone.window_encoders' in module_name:
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    append_text_to_file(initialization_scheme_save_path, "window_encoders.txt", module_name + ".bias: " + nn.init.zeros_.__name__)
            elif isinstance(module, nn.Embedding):
                general_initialization_method(module.weight)
                append_text_to_file(initialization_scheme_save_path, "tokens.txt", module_name + ".weight: " + general_initialization_method.__name__)
        
    # before the pooling and the classifier, and after the Zim backbone
    # module.window_scale_enhancer
    # There are switches in the hyperparameters for toggling on and off ffns in the attentions for aggregation,
    # noise gating in moe modules, and spatial temporal positional encoding in st enhancers.
    # The initialization for these switched modules is implicitly taken care of in the code below and can be
    # checked out in the logs.
    if args.use_clip_scale_st:
        window_scale_enhancer_initialization_method = general_initialization_method
        for pname, param in module_dict['module.window_scale_enhancer'].named_parameters():
            append_text_to_file(initialization_scheme_save_path, "all_parameters_in_window_scale_enhancer.txt", pname)
            if param.dim() > 1:
                window_scale_enhancer_initialization_method(param)
                append_text_to_file(initialization_scheme_save_path, "window_scale_enhancer.txt", pname + ": " + window_scale_enhancer_initialization_method.__name__)

        for module_name, module in module_dict.items():
            if 'module.window_scale_enhancer' in module_name:
                if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                        append_text_to_file(initialization_scheme_save_path, "window_scale_enhancer.txt", module_name + ".bias: " + nn.init.zeros_.__name__)

    # pooling modules
    if args.pooling_method == "attn" or args.pooling_method == "moe":
        pooling_initialization_method = general_initialization_method
        for pname, param in module_dict['module.pooling'].named_parameters():
            append_text_to_file(initialization_scheme_save_path, "all_parameters_in_pooling.txt", pname)
            if param.dim() > 1:
                pooling_initialization_method(param)
                append_text_to_file(initialization_scheme_save_path, "pooling.txt", pname + ": " + pooling_initialization_method.__name__)

        for module_name, module in module_dict.items():
            if 'module.pooling' in module_name:
                if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                        append_text_to_file(initialization_scheme_save_path, "pooling.txt", module_name + ".bias: " + nn.init.zeros_.__name__)
                elif isinstance(module, nn.Embedding):
                    general_initialization_method(module.weight)
                    append_text_to_file(initialization_scheme_save_path, "tokens.txt", module_name + ".weight: " + general_initialization_method.__name__)

        for module_name, module in module_dict.items():
            if 'pooling' in module_name:
                if isinstance(module, nn.Embedding):
                    nn.init.trunc_normal_(module.weight, mean=0.0, std=args.pooling_query_init_std)
                    append_text_to_file(initialization_scheme_save_path, "pooling_embed.txt", module_name + f".weight: {args.pooling_query_init_std}")

    # classifier
    classifier_initialization_method = general_initialization_method
    for module_name, module in module_dict.items():
        if 'module.classifier' in module_name:        
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                classifier_initialization_method(module.weight)
                append_text_to_file(initialization_scheme_save_path, "classifier.txt", module_name + ".weight: " + classifier_initialization_method.__name__)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    append_text_to_file(initialization_scheme_save_path, "classifier.txt", module_name + ".bias: " + nn.init.zeros_.__name__)

    # scaling down for gating weights and noise weights in MoE modules
    if args.pooling_method != "moe" or not args.scale_down_std_for_gate_and_noise_weights: 
        return
    gate_weights_std_scaled = args.std_for_init / 10.0
    nosie_weights_std_scaled = 1e-7
    for module_name, module in module_dict.items():
        # if 'w_gate' in module_name:
        #     for pname, param in module.named_parameters():
        #         if param.dim() > 1:
        #             nn.init.zeros_(param)
        #             append_text_to_file(initialization_scheme_save_path, "moe_gate_and_noise.txt", module_name + ": zeros_")
        #             # nn.init.trunc_normal_(tensor=param, mean=0, std=gate_weights_std_scaled)
        #             # append_text_to_file(initialization_scheme_save_path, "moe_gate_and_noise.txt", module_name + f": {gate_weights_std_scaled}")
        if 'w_noise' in module_name:
            for pname, param in module.named_parameters():
                if param.dim() > 1:
                    # nn.init.zeros_(param)
                    # append_text_to_file(initialization_scheme_save_path, "moe_gate_and_noise.txt", module_name + ": zeros_")
                    nn.init.trunc_normal_(tensor=param, mean=0, std=nosie_weights_std_scaled)       
                    append_text_to_file(initialization_scheme_save_path, "moe_gate_and_noise.txt", module_name + f": {nosie_weights_std_scaled}")   


def accuracy(output, target):
    output = torch.argmax(output, dim=1)
    correct = torch.sum(torch.eq(target.int(), output.int())).float()
    return correct.item() / output.shape[0]

if __name__ == '__main__':
    main()
