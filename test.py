# test.py (FIXED VERSION)

# CONFIG
import argparse
import os
import functools
import emoji
import time
import random
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader

# Assuming these are available in your project structure
# Ensure these paths are correct relative to where you run the script
from configs import get as get_cfg
from dataloaders.modified_kitti_loader import KittiDepth
from model import get as get_model
from summary import get as get_summary
from metric import get as get_metric
from utility import * # Assuming count_parameters, remove_moudle, count_validpoint are here

parser = argparse.ArgumentParser(description='depth completion testing')
parser.add_argument('-p', '--project_name', type=str, default='test_run')
parser.add_argument('-c', '--configuration', type=str, default='test_config.yml',
                   help='Path to the YAML configuration file for testing.')
parser.add_argument('-m', '--model_path', type=str, required=True,
                   help='Path to the trained model checkpoint (.pth file).')
parser.add_argument('-o', '--output_dir', type=str, default='test_results',
                   help='Directory to save test metrics and output images.')
arg = parser.parse_args() # Corrected: Call parse_args() on the parser instance

# Load configuration from the specified YAML file
config = get_cfg(arg)

# Override config arguments with command line arguments
# Ensure these attributes exist or are handled in your config logic
config.project_name = arg.project_name
config.test_model = arg.model_path
config.test_dir = arg.output_dir

# ENVIRONMENT SETTINGS
rootPath = os.path.abspath(os.path.dirname(__file__))

# VARIANCES
sample_, output_ = None, None
metric_txt_dir = None

# MINIMIZE RANDOMNESS
torch.manual_seed(config.seed)
np.random.seed(config.seed)
random.seed(config.seed)


def test(args):
    # DATASET
    print(emoji.emojize('Prepare data for testing... :writing_hand:', variant="emoji_type"), end=' ')
    global sample_, output_, metric_txt_dir

    # Ensure args.test_option is properly defined in your config.yml
    data_test = KittiDepth(args.test_option, args)
    loader_test = DataLoader(dataset=data_test,
                             batch_size=1,
                             shuffle=False,
                             num_workers=getattr(args, 'num_workers', 0))  # Set to 0 for debugging

    print('Done!')

    # NETWORK
    print(emoji.emojize('Prepare model... :writing_hand:', variant="emoji_type"), end=' ')
    model_builder = get_model(args)
    net = model_builder(args)
    net.cpu()
    print('Done!')

    # METRIC
    print(emoji.emojize('Prepare metric... :writing_hand:', variant="emoji_type"), end=' ')
    metric_builder = get_metric(args)
    metric = metric_builder(args)
    print('Done!')

    # SUMMARY & Output Directory Setup
    print(emoji.emojize('Prepare summary and output directory... :writing_hand:', variant="emoji_type"), end=' ')

    writer_test = None

    try:
        # Create base test directory
        os.makedirs(args.test_dir, exist_ok=True)
        # Create a subdirectory for this specific test run
        output_subdir = os.path.join(args.test_dir, 'test')
        os.makedirs(output_subdir, exist_ok=True)

        metric_txt_dir = os.path.join(output_subdir, 'result_metric.txt')
        with open(metric_txt_dir, 'w') as f:
            f.write('test_model: {} \ntest_option: {} \n'
                    'test_name: {} \ntest_not_random_crop: {} \n'
                    'tta: {}\n \n'.format(args.test_model, args.test_option,
                                           args.project_name,
                                           getattr(args, 'test_not_random_crop', False),
                                           getattr(args, 'tta', False)))

        # Correctly instantiate the Summary class
        SummaryClass = get_summary(args)
        writer_test = SummaryClass(log_dir=output_subdir,
                                   mode='test',
                                   args=args,
                                   loss_name=['loss'],
                                   metric_name=metric.metric_name)
        print('Done!')
    except Exception as e:
        print(f"Error setting up summary or output directory: {e}")
        writer_test = None

    # Load trained model
    if os.path.isfile(args.test_model):
        print(emoji.emojize(f'Found checkpoint at {args.test_model} :white_check_mark:', variant="emoji_type"))
        checkpoint = torch.load(args.test_model, map_location=torch.device('cpu'))
        print("Checkpoint loaded successfully")

        # Handle different checkpoint formats
        model_state_dict = None
        if 'model' in checkpoint:
            model_state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            model_state_dict = checkpoint['state_dict']
        elif isinstance(checkpoint, dict) and any(key.startswith('module.') or '.' in key for key in checkpoint.keys()):
            model_state_dict = checkpoint
        else:
            print(emoji.emojize(f'Error: Could not find model state dictionary in checkpoint. :x:', variant="emoji_type"))
            return

        net.load_state_dict(remove_moudle(model_state_dict))
        print('Model loaded successfully.')
    else:
        print(emoji.emojize(f'Error: No checkpoint found at {args.test_model}. :x:', variant="emoji_type"))
        return

    # Set model to evaluation mode
    net.eval()

    # Testing loop
    print(emoji.emojize('Starting testing... :rocket:', variant="emoji_type"))
    t_total = 0
    num_sample = 0

    pbar_ = tqdm(loader_test, leave=False, dynamic_ncols=True)
    print(emoji.emojize('Testing progress: :hourglass_flowing_sand:', variant="emoji_type"))
    pbar_.set_description("Testing progress")
    print("Length of data_test:", len(data_test))

    for i, sample_batched in enumerate(pbar_):

        # Extract 'paths' separately as it's a list and doesn't need to be moved to CPU
        paths_data = sample_batched.pop('paths', None)

        # Move other data (tensors) to CPU
        sample_gpu = {key: val.cpu() for key, val in sample_batched.items() if val is not None}

        # Forward pass
        t_start = time.time()
        with torch.no_grad():
            output_ = net(sample_gpu)
        t_end = time.time()
        t_total += (t_end - t_start)

        num_sample += 1

        if writer_test is not None:
            _ = writer_test.update(getattr(args, 'epochs', 0), sample_gpu, output_,
                                   online_loss=False, online_metric=False, online_rmse_only=False, online_img=True)
            # Pass paths_data explicitly to write_img if it needs it for naming or saving
            writer_test.save(getattr(args, 'epochs', 0), i, sample_gpu, output_)

        pbar_.set_description("Test Sample %i" % i)
        sample_ = sample_gpu
        output_ = output_

    pbar_.close()

    # Final summary update
    if writer_test is not None and sample_ is not None and output_ is not None:
        _ = writer_test.update(getattr(args, 'epochs', 0), sample_, output_,
                                online_loss=False, online_metric=False, online_rmse_only=False, online_img=False)
    else:
        print("Warning: writer_test or sample_gpu/output_ not initialized. Skipping final summary update.")

    t_avg = t_total / num_sample if num_sample > 0 else 0
    if metric_txt_dir:
        with open(metric_txt_dir, 'a') as f:
            f.write('\nElapsed time : {:.2f} sec, '
                    'Average processing time : {:.4f} sec/sample'.format(t_total, t_avg))

    print('Elapsed time : {:.2f} sec, '
          'Average processing time : {:.4f} sec/sample'.format(t_total, t_avg))


if __name__ == '__main__':
    # Ensure config object has a 'seed' attribute
    if not hasattr(config, 'seed'):
        config.seed = 42
        print("Warning: 'seed' not found in config, defaulting to 42.")

    test(config)