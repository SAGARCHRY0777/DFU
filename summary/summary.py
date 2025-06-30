from . import BaseSummary
import matplotlib.pyplot as plt
import wandb
from PIL import Image
from utility import *
import numpy as np # Ensure numpy is imported

cm = plt.get_cmap('plasma')
log_metric_val = None
log_metric_val_rmse, log_metric_val_mae = None, None


class Summary(BaseSummary):
    def __init__(self, log_dir, mode, args, loss_name, metric_name):
        super(Summary, self).__init__(log_dir, mode, args)

        self.log_dir = log_dir
        self.mode = mode
        self.args = args

        self.loss = []
        self.metric = []

        self.loss_name = loss_name
        self.metric_name = metric_name

        self.path_output = None
        self.img_mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        self.img_std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)

    def add(self, loss=None, metric=None, log_itr=None):
        # loss and metric should be numpy arrays
        if loss is not None:
            # Ensure loss is a numpy array before appending
            if torch.is_tensor(loss):
                self.loss.append(loss.detach().data.cpu().numpy())
            else:
                self.loss.append(loss)
        if metric is not None:
            # Ensure metric is a numpy array before appending
            if torch.is_tensor(metric):
                self.metric.append(metric.detach().data.cpu().numpy())
            else:
                self.metric.append(metric)

        if 'train' in self.mode and log_itr % self.args.vis_step == 0:
            log_dict = {}
            for idx, loss_type in enumerate(self.loss_name):
                val = loss.data.cpu().numpy()[0, idx]
                log_dict[self.mode + '_all_' + loss_type] = val
                self.add_scalar('All/' + loss_type, val, log_itr)
            log_dict['custom_step_loss'] = log_itr
            wandb.log(log_dict)

    def update(self, global_step, sample, output,
               online_loss=True, online_metric=True, online_rmse_only=True, online_img=True):
        """
        update results
        """
        global log_metric_val, log_metric_val_rmse, log_metric_val_mae
        log_dict = {}

        # The error occurs here if self.loss is empty. Add a check.
        if self.loss_name is not None and len(self.loss) > 0:
            self.loss = np.concatenate(self.loss, axis=0)
            self.loss = np.mean(self.loss, axis=0, keepdims=True)

            msg = [" {:<9s}|  ".format(self.mode + '_Loss')]
            for idx, loss_type in enumerate(self.loss_name):
                val = self.loss[0, idx]
                if online_loss:
                    log_dict[self.mode + '_' + loss_type] = val
                self.add_scalar('Loss/' + loss_type, val, global_step)

                msg += ["{:<s}: {:.4f}  ".format(loss_type, val)]

                if (idx + 1) % 10 == 0:
                    msg += ["\n             "]

            msg = "".join(msg)
            print(msg)

            # Ensure f_loss is defined or handled by BaseSummary correctly
            # This assumes BaseSummary handles file opening/closing or self.f_loss exists.
            if hasattr(self, 'f_loss') and self.f_loss is not None:
                f_loss = open(self.f_loss, 'a')
                f_loss.write('{:04d} | {}\n'.format(global_step, msg))
                f_loss.close()
            else:
                print("Warning: f_loss file not initialized for logging loss.")


        # The error occurs here if self.metric is empty. Add a check.
        if self.metric_name is not None and len(self.metric) > 0:
            self.metric = np.concatenate(self.metric, axis=0)
            self.metric = np.mean(self.metric, axis=0, keepdims=True)

            msg = [" {:<9s}|  ".format(self.mode + '_Metric')]
            for idx, name in enumerate(self.metric_name):
                val = self.metric[0, idx]
                if online_metric:
                    if online_rmse_only:
                        if name == 'RMSE':
                            log_metric_val = val
                            log_dict[self.mode + '_' + name] = val
                        else:
                            pass
                    else:
                        if name == 'RMSE':
                            log_metric_val_rmse = val
                        elif name == "MAE":
                            log_metric_val_mae = val
                        log_dict[self.mode + '_' + name] = val
                self.add_scalar('Metric/' + name, val, global_step)

                msg += ["{:<s}: {:.4f}  ".format(name, val)]

                if (idx + 1) % 10 == 0:
                    msg += ["\n             "]

            msg = "".join(msg)
            print(msg)

            if hasattr(self, 'f_metric') and self.f_metric is not None:
                f_metric = open(self.f_metric, 'a')
                f_metric.write('{:04d} | {}\n'.format(global_step, msg))
                f_metric.close()
            else:
                print("Warning: f_metric file not initialized for logging metric.")


            if self.args.test:
                # Ensure args.test_dir is accessible and valid
                if hasattr(self.args, 'test_dir') and self.args.test_dir:
                    test_metric_path = os.path.join(self.args.test_dir, 'test', 'result_metric.txt')
                    # Make sure the directory exists before trying to open the file
                    os.makedirs(os.path.dirname(test_metric_path), exist_ok=True)
                    f_metric = open(test_metric_path, 'a')
                    f_metric.write('\n{:04d} | {}\n'.format(global_step, msg))
                    f_metric.close()
                else:
                    print("Warning: args.test_dir not set for test mode metric logging.")

        if 'train' in self.mode:
            log_dict['custom_step_train'] = global_step
        elif 'val' in self.mode:
            log_dict['custom_step_val'] = global_step
        elif 'test' in self.mode:
            log_dict['custom_step_test'] = global_step

        # Log by wandb
        if len(log_dict) != 0 and 'test' not in self.mode:
                if 'wandb' in globals() and wandb.run is not None: # Check if wandb is initialized
                    wandb.log(log_dict)
                else:
                    print("Warning: wandb is not initialized for logging.")


        # Reset
        self.loss = []
        self.metric = []

        return log_metric_val_rmse, log_metric_val_mae

    def save(self, epoch, idx, sample, output):
        with torch.no_grad():
            if self.args.save_result_only:
                if not self.args.test:
                    self.path_output = '{}/{}/epoch{:04d}'.format(self.log_dir,
                                                              'result_pred', epoch)
                else:
                    self.path_output = '{}/{}/{}'.format(self.log_dir,
                                                                  'test', 'depth_gray')

                os.makedirs(self.path_output, exist_ok=True)

                path_save_pred = '{}/{:010d}.png'.format(self.path_output, idx)

                # Assuming 'output' directly contains the prediction tensor for this case
                # This part needs to align with how your model's output is structured
                # For example, if output is just the tensor: pred = output.detach()
                # If output is a dictionary, like output['results'][-1]:
                # Assuming output is the direct depth prediction tensor
                pred = output.detach() # Changed from output[self.args.output][-1] to just output


                pred = torch.clamp(pred, min=0)

                pred = pred[0, 0, :, :].data.cpu().numpy()

                # This part related to pad_rep might need adjustment based on your utility.py
                # and whether test_not_random_crop is intended to be True/False during testing
                if not self.args.test_not_random_crop:
                    org_size = (352, 1216) # This should match your dataset's original size
                    # Ensure pad_rep is correctly imported from utility
                    if 'pad_rep' in globals():
                        pred = pad_rep(pred, org_size)
                    else:
                        print("Warning: pad_rep function not found. Skipping padding.")


                pred = (pred*256.0).astype(np.uint16)
                pred = Image.fromarray(pred)
                pred.save(path_save_pred)

            else:
                rgb = sample['rgb'].detach()
                rgb.mul_(self.img_std.type_as(rgb)).add_(self.img_mean.type_as(rgb))
                rgb = sample['rgb'].data.cpu().numpy()
                dep = sample['dep'].detach().data.cpu().numpy()
                gt = sample['gt'].detach().data.cpu().numpy() # Changed from sample['dep'] to sample['gt']
                # Assuming 'output' directly contains the prediction tensor for this case
                pred = output['results'][-1].detach().data.cpu().numpy()


                num_summary = gt.shape[0]
                if num_summary > self.args.num_summary:
                    num_summary = self.args.num_summary

                    rgb = rgb[0:num_summary, :, :, :]
                    dep = dep[0:num_summary, :, :, :]
                    gt = gt[0:num_summary, :, :, :]
                    pred = pred[0:num_summary, :, :, :]

                rgb = np.clip(rgb, a_min=0, a_max=1.0)
                dep = np.clip(dep, a_min=0, a_max=self.args.max_depth)
                gt = np.clip(gt, a_min=0, a_max=self.args.max_depth)
                pred = np.clip(pred, a_min=0, a_max=self.args.max_depth)

                list_imgv, list_imgh = [], []
                for b in range(0, num_summary):
                    rgb_tmp = rgb[b, :, :, :]
                    dep_tmp = dep[b, 0, :, :]
                    gt_tmp = gt[b, 0, :, :]
                    pred_tmp = pred[b, 0, :, :]

                    rgb_tmp = rgb_tmp
                    dep_tmp = dep_tmp / self.args.max_depth
                    gt_tmp = gt_tmp / self.args.max_depth
                    pred_tmp = pred_tmp / self.args.max_depth

                    # Check if cm is a callable colormap before using
                    if callable(cm):
                        dep_tmp = (255.0 * cm(dep_tmp)).astype('uint8')
                        gt_tmp = (255.0 * cm(gt_tmp)).astype('uint8')
                        pred_tmp = (255.0 * cm(pred_tmp)).astype('uint8')
                    else:
                        print("Warning: Colormap 'cm' is not callable. Skipping colormap conversion for images.")
                        # Fallback to grayscale if colormap fails
                        dep_tmp = (255.0 * dep_tmp).astype('uint8')
                        gt_tmp = (255.0 * gt_tmp).astype('uint8')
                        pred_tmp = (255.0 * pred_tmp).astype('uint8')


                    rgb_tmp = 255.0 * np.transpose(rgb_tmp, (1, 2, 0))
                    rgb_tmp = np.clip(rgb_tmp, 0, 256).astype('uint8')
                    rgb_tmp = Image.fromarray(rgb_tmp, 'RGB')
                    # Ensure images are RGB, if they are originally grayscale from colormap,
                    # convert them to 3 channels by slicing.
                    dep_tmp = Image.fromarray(dep_tmp[:, :, :3], 'RGB')
                    gt_tmp = Image.fromarray(gt_tmp[:, :, :3], 'RGB')
                    pred_tmp = Image.fromarray(pred_tmp[:, :, :3], 'RGB')

                    # FIXME
                    list_imgv = [rgb_tmp,
                                 dep_tmp,
                                 gt_tmp,
                                 pred_tmp]

                    widths, heights = zip(*(i.size for i in list_imgv))
                    max_width = max(widths)
                    total_height = sum(heights)
                    new_im = Image.new('RGB', (max_width, total_height))
                    y_offset = 0
                    for im in list_imgv:
                        new_im.paste(im, (0, y_offset))
                        y_offset += im.size[1]

                    list_imgh.append(new_im)

                widths, heights = zip(*(i.size for i in list_imgh))
                total_width = sum(widths)
                max_height = max(heights)
                img_total = Image.new('RGB', (total_width, max_height))
                x_offset = 0
                for im in list_imgh:
                    img_total.paste(im, (x_offset, 0))
                    x_offset += im.size[0]

                if not self.args.test:
                    self.path_output = '{}/{}'.format(self.log_dir, 'result_analy')
                else:
                    self.path_output = '{}/{}'.format(self.log_dir, 'test')
                os.makedirs(self.path_output, exist_ok=True)
                if not self.args.test:
                    path_save = '{}/epoch{:04d}_{:08d}_result.png'.format(self.path_output, epoch, idx)
                else:
                    os.makedirs('{}/depth_analy'.format(self.path_output), exist_ok=True)
                    os.makedirs('{}/depth_rgb'.format(self.path_output), exist_ok=True)
                    path_save = '{}/depth_analy/{}'.format(self.path_output, '{}.jpg'.format(idx))
                    pred_tmp.save('{}/depth_rgb/{}'.format(self.path_output, '{}.jpg'.format(idx)))
                img_total.save(path_save)