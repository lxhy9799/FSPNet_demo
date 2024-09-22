import logging
import os
import time

import numpy as np
import torch as t
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from loader import ngsimDataset, highdDataset
from model import Net


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import argparse

device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
## Network Arguments


parser = argparse.ArgumentParser(description='Evaluating:')
# General setting------------------------------------------
parser.add_argument('--use_cuda', action='store_false', help='if use cuda (default: True)', default=True)
parser.add_argument('--use_fusion', action="store_false", help='if use targets fusion module (default: True)',
                    default=True)
parser.add_argument('--use_true_man', action="store_false", help='(default: True)',
                    default=True)
parser.add_argument('--batch_size', type=int, help='batch size to use (default: 64)', default=128)
parser.add_argument('--tensorboard', action="store_true", help='if use tensorboard (default: True)', default=True)
# IO setting------------------------------------------
parser.add_argument('--grid_size', type=int, help='default: (13,3)', nargs=2, default=[13, 3])
parser.add_argument('--in_length', type=int, help='History sequence (default: 16)',
                    default=16)  # 3s history traj at 5Hz
parser.add_argument('--out_length', type=int, help='Predict sequence (default: 25)',
                    default=25)  # 5s future traj at 5Hz
parser.add_argument('--num_lat_classes', type=int, help='Classes of lateral behaviors', default=3)
parser.add_argument('--num_lon_classes', type=int, help='Classes of longitude behaviors', default=3)
# Network hyperparameters------------------------------------------
parser.add_argument('--num_features', type=int, help='The last dimension of input', default=5)
parser.add_argument('--num_blocks', type=int, default=4, help='')
parser.add_argument('--input_embed_size', type=int, default=32, help='embed size')
parser.add_argument('--lstm_encoder_size', type=int, default=64, help='dimension of input')
parser.add_argument('--att_out_size', type=int, default=32, help='dimension of input')
parser.add_argument('--ff_hidden_size', type=int, default=256, help='')
parser.add_argument('--decoder_size', type=int, default=128, help='LSTM size')
parser.add_argument('--num_heads', type=int, default=8, help='number of attention head')
parser.add_argument('--train_flag', type=bool, default=False, help='train flag')
parser.add_argument('--use_maneuvers', type=bool, default=True, help='')
# Training setting------------------------------------------
parser.add_argument('--name', type=str, help='log name (default: "1")', default="ngsim")
parser.add_argument('--test_set', type=str, default='data/ngsim/TestSet.mat', help='Path to validation datasets')
parser.add_argument("--num_workers", type=int, default=8, help="number of workers used for dataloader")
parser.add_argument('--pretrain_epochs', type=int, help='epochs of pre-training using MSE', default=3)
parser.add_argument('--train_epochs', type=int, help='epochs of training using NLL', default=1)
parser.add_argument('--dataset_name', type=str, help='', default='ngsim')
parser.add_argument('--val_use_mse', type=bool, default=True, help='')
parser.add_argument('--refine_flag', type=bool,  default=True)
parser.add_argument('--spatial_flag', type=bool,  default=True)
parser.add_argument('--stochastic_flag', type=bool,  default=True)
parser.add_argument('--temporal_flag', type=bool,  default=True)

net_args = parser.parse_args()



class Evaluate():
    def __init__(self):
        self.op = 0
        self.drawImg = False
        self.scale = 0.3048
        self.prop = 1
        self.delta = 2.0
        self.v_w = 4
        self.v_l = 8
        self.op_sparse=0

    def main(self, val):
            log_path = r"./trained_models/{}/".format(net_args.name)
            os.makedirs(log_path, exist_ok=True)
            utils.initLogging(log_file=log_path + 'evaluate.log')
            model_step = 1
            net = Net(net_args)
            check_point = t.load('./trained_models/%s/%s-pre%d-nll%d.pt' % (
                net_args.dataset_name,net_args.name, net_args.pretrain_epochs, net_args.train_epochs))
            net.load_state_dict(check_point['model_state_dict'])
            net = net.to(device)
            net.eval()
            if val:
                if net_args.name=="ngsim":
                    t2 = ngsimDataset(net_args.test_set)
                elif net_args.name=="highd":
                    t2 = highdDataset(net_args.test_set)
                valDataloader = DataLoader(t2, batch_size=net_args.batch_size, shuffle=False, num_workers=0,
                                           collate_fn=t2.collate_fn)
            else:
                # ------------------------------------------------------------
                if net_args.dataset_name == "ngsim":
                    t2 = ngsimDataset(net_args.test_set)
                if net_args.dataset_name=='highd':
                    t2 = highdDataset(net_args.test_set)
                valDataloader = DataLoader(t2, batch_size=net_args.batch_size, shuffle=False, num_workers=net_args.num_workers,
                                           collate_fn=t2.collate_fn)
            logging.info("DataSet Prepared : {} test data\n".format(len(t2)))
            lossVals = t.zeros(net_args.out_length).to(device)
            counts = t.zeros(net_args.out_length).to(device)
            avg_val_loss = 0
            all_time = 0
            nbrsss = 0
            val_batch_count = len(valDataloader)
            print("begin.................................\n")
            with(t.no_grad()):
                for idx, data in enumerate(tqdm(valDataloader)):
                    hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask, va, nbrsva, lane, nbrslane, dis, nbrsdis, cls, nbrscls, map_positions= data
                    hist = hist.to(device)
                    nbrs = nbrs.to(device)
                    mask = mask.to(device)
                    lat_enc = lat_enc.to(device)
                    lon_enc = lon_enc.to(device)
                    fut = fut[:net_args.out_length, :, :]
                    fut = fut.to(device)
                    op_mask = op_mask[:net_args.out_length, :, :]
                    op_mask = op_mask.to(device)
                    va = va.to(device)
                    nbrsva = nbrsva.to(device)
                    cls = cls.to(device)
                    nbrscls = nbrscls.to(device)
                    te=time.time()
                    fut_pred, lat_pred, lon_pred= net(hist, nbrs, mask, va, nbrsva, cls, nbrscls, lat_enc, lon_enc)
                    all_time += time.time() - te
                    nbrsss += 1
                    if not net_args.train_flag:
                        indices = []
                        if net_args.val_use_mse:
                            fut_pred_max = t.zeros_like(fut_pred[0])
                            for k in range(lat_pred.shape[0]):  # 128
                                lat_man = t.argmax(lat_enc[k, :]).detach()
                                lon_man = t.argmax(lon_enc[k, :]).detach()
                                index = lon_man * 3 + lat_man
                                indices.append(index)
                                fut_pred_max[:, k, :] = fut_pred[index][:, k, :]
                            l, c, loss = self.maskedMSETest(fut_pred_max, fut, op_mask)
                        else:
                            l, c, loss = self.maskedNLLTest(fut_pred, lat_pred, lon_pred, fut, op_mask,
                                                            use_maneuvers=net_args.use_maneuvers)

                            # self.draw_sparse(spatial_attn_weight,sparse_spatial_attn_weight,temporal_attn_weight,sparse_tempotal_attn_weight,lon_man, lat_man)
                    else:
                        if net_args.val_use_mse:
                            l, c, loss = self.maskedMSETest(fut_pred, fut, op_mask)
                        else:
                            l, c, loss = self.maskedNLLTest(fut_pred, lat_pred, lon_pred, fut, op_mask,
                                                            use_maneuvers=net_args.use_maneuvers)

                    lossVals += l.detach()
                    counts += c.detach()
                    avg_val_loss += loss.item()
                    if idx == int(val_batch_count / 4) * model_step:
                        print('process:', model_step / 4)
                        model_step += 1
                    # if idx==99:
                    #    break
            # tqdm.write('valmse:', avg_val_loss / val_batch_count)
                if net_args.val_use_mse:
                    print('valmse:', avg_val_loss / val_batch_count* 0.3048)
                    print(t.pow(lossVals / counts, 0.5) * 0.3048)  # Calculate RMSE and convert from feet to meters
                    print(all_time / nbrsss, "ref time")
                    rmseOverall = (t.pow(lossVals / counts, 0.5) * 0.3048).cpu()
                    pred_rmse_horiz = horiz_eval(rmseOverall, 5)
                    print(pred_rmse_horiz)
                    print("RMSE (m)\t=> {}, Mean={:.3f}".format(rmseOverall[4::5], rmseOverall[4::5].mean()))#Final error
                    print("ADE (m)\t=> {}".format(rmseOverall.mean(),
                                                               ))  # Final error
                    # Saving Results to a csv file
                    # rmse_eval = np.around(pred_rmse_horiz, decimals=2)
                    # df = pd.DataFrame(rmse_eval)
                    # test_cases = ['overall']
                    # df.to_csv('results/rmse.csv', header=test_cases, index=False)
                else:
                    print('valnll:', avg_val_loss / val_batch_count)
                    print(lossVals / counts)
                    print(lossVals/counts*0.3048)

    def add_car(self, plt, x, y, alp, color0):
        plt.gca().add_patch(plt.Rectangle(
            (x, y - 1),
            4.0,
            2.0,
            color=color0,
            alpha=alp,
        ))

    def maskedMSETest(self, y_pred, y_gt, mask):
        acc = t.zeros_like(mask)
        muX = y_pred[:, :, 0]
        muY = y_pred[:, :, 1]
        x = y_gt[:, :, 0]
        y = y_gt[:, :, 1]
        out = t.pow(x - muX, 2) + t.pow(y - muY, 2)
        acc[:, :, 0] = out
        acc[:, :, 1] = out
        acc = acc * mask
        lossVal = t.sum(acc[:, :, 0], dim=1)
        counts = t.sum(mask[:, :, 0], dim=1)
        loss = t.sum(acc) / t.sum(mask)
        return lossVal, counts, loss
    def FDETest(self,y_pred,y_gt,mask):
        acc = t.zeros_like(mask)
        muX = y_pred[:, :, 0]
        muY = y_pred[:, :, 1]
        x = y_gt[:, :, 0]
        y = y_gt[:, :, 1]
        out = x - muX + y - muY
        acc[:, :, 0] = out
        acc[:, :, 1] = out
        acc = acc * mask
        lossVal = t.sum(acc[:, :, 0], dim=1)
        counts = t.sum(mask[:, :, 0], dim=1)
        loss = t.sum(acc) / t.sum(mask)
        return lossVal, counts, loss

    def ADETest(self,y_pred,y_gt,mask):
        acc = t.zeros_like(mask)
        muX = y_pred[:, :, 0]
        muY = y_pred[:, :, 1]
        x = y_gt[:, :, 0]
        y = y_gt[:, :, 1]
        out = x - muX + y - muY
        acc[:, :, 0] = out
        acc[:, :, 1] = out
        acc = acc * mask
        lossVal = t.sum(acc[:, :, 0], dim=1)
        counts = t.sum(mask[:, :, 0], dim=1)
        loss = t.sum(acc) / t.sum(mask)
        return lossVal, counts, loss

    def logsumexp(self, inputs, dim=None, keepdim=False):
        if dim is None:
            inputs = inputs.view(-1)
            dim = 0
        s, _ = t.max(inputs, dim=dim, keepdim=True)
        outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
        if not keepdim:
            outputs = outputs.squeeze(dim)
        return outputs

    def maskedNLLTest(self, fut_pred, lat_pred, lon_pred, fut, op_mask, num_lat_classes=3, num_lon_classes=3,
                      use_maneuvers=True):
        if use_maneuvers:
            acc = t.zeros(op_mask.shape[0], op_mask.shape[1], num_lon_classes * num_lat_classes).to(device)
            count = 0
            for k in range(num_lon_classes):
                for l in range(num_lat_classes):
                    wts = lat_pred[:, l] * lon_pred[:, k]
                    wts = wts.repeat(len(fut_pred[0]), 1)
                    y_pred = fut_pred[k * num_lat_classes + l]
                    y_gt = fut
                    muX = y_pred[:, :, 0]
                    muY = y_pred[:, :, 1]
                    sigX = y_pred[:, :, 2]
                    sigY = y_pred[:, :, 3]
                    rho = y_pred[:, :, 4]
                    ohr = t.pow(1 - t.pow(rho, 2), -0.5)
                    x = y_gt[:, :, 0]
                    y = y_gt[:, :, 1]
                    # If we represent likelihood in feet^(-1):
                    out = -(0.5 * t.pow(ohr, 2) * (
                            t.pow(sigX, 2) * t.pow(x - muX, 2) + 0.5 * t.pow(sigY, 2) * t.pow(
                        y - muY, 2) - rho * t.pow(sigX, 1) * t.pow(sigY, 1) * (x - muX) * (
                                    y - muY)) - t.log(sigX * sigY * ohr) + 1.8379)
                    acc[:, :, count] = out + t.log(wts)
                    count += 1
            acc = -self.logsumexp(acc, dim=2)
            acc = acc * op_mask[:, :, 0]
            loss = t.sum(acc) / t.sum(op_mask[:, :, 0])
            lossVal = t.sum(acc, dim=1)
            counts = t.sum(op_mask[:, :, 0], dim=1)
            return lossVal, counts, loss
        else:
            acc = t.zeros(op_mask.shape[0], op_mask.shape[1], 1).to(device)
            y_pred = fut_pred
            y_gt = fut
            muX = y_pred[:, :, 0]
            muY = y_pred[:, :, 1]
            sigX = y_pred[:, :, 2]
            sigY = y_pred[:, :, 3]
            rho = y_pred[:, :, 4]
            ohr = t.pow(1 - t.pow(rho, 2), -0.5)  # p
            x = y_gt[:, :, 0]
            y = y_gt[:, :, 1]
            # If we represent likelihood in feet^(-1):
            out = 0.5 * t.pow(ohr, 2) * (
                    t.pow(sigX, 2) * t.pow(x - muX, 2) + t.pow(sigY, 2) * t.pow(y - muY,
                                                                                2) - 2 * rho * t.pow(
                sigX, 1) * t.pow(sigY, 1) * (x - muX) * (y - muY)) - t.log(sigX * sigY * ohr) + 1.8379
            acc[:, :, 0] = out
            acc = acc * op_mask[:, :, 0:1]
            loss = t.sum(acc[:, :, 0]) / t.sum(op_mask[:, :, 0])
            lossVal = t.sum(acc[:, :, 0], dim=1)
            counts = t.sum(op_mask[:, :, 0], dim=1)
            return lossVal, counts, loss
def horiz_eval(loss_total, n_horiz):
    loss_total = loss_total.cpu().numpy()
    avg_res = np.zeros(n_horiz)
    n_all = loss_total.shape[0]
    n_frames = n_all // n_horiz
    for i in range(n_horiz):
        if i == 0:
            st_id = 0
        else:
            st_id = n_frames * i

        if i == n_horiz - 1:
            en_id = n_all - 1
        else:
            en_id = n_frames * i + n_frames - 1

        avg_res[i] = np.mean(loss_total[st_id:en_id + 1])

    return avg_res

if __name__ == '__main__':
    evaluate = Evaluate()
    evaluate.drawImg=True
    evaluate.main(val=False)
