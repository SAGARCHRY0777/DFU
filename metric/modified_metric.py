import torch
from . import BaseMetric


class Metric(BaseMetric):
    def __init__(self, args):
        super(Metric, self).__init__(args)

        self.args = args
        self.t_valid = 0.0001

        self.metric_name = [
            'RMSE', 'MAE', 'iRMSE', 'iMAE', 'REL', 'D^1', 'D^2', 'D^3'
        ]

    def evaluate(self, output, sample, mode):
        # Ground truth evaluation is disabled as per user's request.
        # This method previously calculated various metrics (RMSE, MAE, etc.)
        # by comparing 'output' (predictions) with 'sample' (ground truth).
        # Since ground truth is not to be used, this method will do nothing.
        pass

        # with torch.no_grad():
        #
        #     pred, gt = output.detach(), sample.detach()
        #
        #     pred_inv = 1.0 / (pred + 1e-8)
        #     gt_inv = 1.0 / (gt + 1e-8)
        #
        #     # For numerical stability
        #     mask = gt > self.t_valid
        #     num_valid = mask.sum()
        #
        #     pred = pred[mask]
        #     gt = gt[mask]
        #
        #     pred_inv = pred_inv[mask]
        #     gt_inv = gt_inv[mask]
        #
        #     pred_inv[pred <= self.t_valid] = 0.0
        #     gt_inv[gt <= self.t_valid] = 0.0
        #
        #     # RMSE / MAE
        #     diff = pred - gt
        #     diff_abs = torch.abs(diff)
        #     diff_sqr = torch.pow(diff, 2)
        #
        #     # tmp = diff_sqr.sum()
        #     rmse = diff_sqr.sum() / (num_valid + 1e-8)
        #     rmse = torch.sqrt(rmse)
        #
        #     mae = diff_abs.sum() / (num_valid + 1e-8)
        #
        #     # iRMSE / iMAE
        #     diff_inv = pred_inv - gt_inv
        #     diff_inv_abs = torch.abs(diff_inv)
        #     diff_inv_sqr = torch.pow(diff_inv, 2)
        #
        #     irmse = diff_inv_sqr.sum() / (num_valid + 1e-8)
        #     irmse = torch.sqrt(irmse)
        #
        #     imae = diff_inv_abs.sum() / (num_valid + 1e-8)
        #
        #     # Rel
        #     rel = diff_abs / (gt + 1e-8)
        #     rel = rel.sum() / (num_valid + 1e-8)
        #
        #     # delta
        #     r1 = gt / (pred + 1e-8)
        #     r2 = pred / (gt + 1e-8)
        #     ratio = torch.max(r1, r2)
        #     d1 = (ratio < 1.25).float().sum() / (num_valid + 1e-8)
        #     d2 = (ratio < 1.25 ** 2).float().sum() / (num_valid + 1e-8)
        #     d3 = (ratio < 1.25 ** 3).float().sum() / (num_valid + 1e-8)
        #
        #     loss_sum = torch.cat((rmse.reshape(1), mae.reshape(1),
        #                           irmse.reshape(1), imae.reshape(1),
        #                           rel.reshape(1), d1.reshape(1),
        #                           d2.reshape(1), d3.reshape(1))).reshape(1, -1)
        #     return loss_sum