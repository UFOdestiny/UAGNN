from __future__ import division

import copy
import os
import pickle

import numpy as np
import torch.optim as optim

from config import get_config
from model import *
from utils import load_AX, generate_all

args, log_dir, logger = get_config()
# set_seed(args.seed)
device = torch.device(args.device)
A_wave, A_q, A_h, X, stds, means = load_AX(args.A_path, args.X_path, device)
node_dim = X.shape[0]
features = X.shape[1]

SCN1 = MDGCN(args.num_timesteps_input, args.hidden_dim_s, args.orders).to(device=device)
SCN2 = MDGCN(args.hidden_dim_s, args.rank_s, args.orders, activation='linear').to(device=device)
SCN3 = MDGCN(args.rank_s, args.hidden_dim_s, args.orders).to(device=device)
SGB = GaussNorm_A(args.hidden_dim_s * X.shape[1], X.shape[1]).to(device=device)

TCN1 = TCNN(features, args.hidden_dim_t, kernel_size=3).to(device=device)
TCN2 = TCNN(args.rank_t, args.hidden_dim_t, kernel_size=3, activation='linear').to(device=device)
TCN3 = TCNN(args.rank_t, args.hidden_dim_t, kernel_size=3).to(device=device)
TGB = GaussNorm_A(args.hidden_dim_t, node_dim).to(device=device)

STmodel = UAHGNN(SCN1, SCN2, SCN3, TCN1, TCN2, TCN3,
                 SGB, TGB, node_dim, features, args, A_wave=A_wave).to(device=device)

logger.info(f"parameters:{sum([param.nelement() for param in STmodel.parameters()])}")

training_input, training_target, val_input, val_target, test_input, test_target = generate_all(X,
                                                                                               args.num_timesteps_input,
                                                                                               args.num_timesteps_output,
                                                                                               args.train_ratio,
                                                                                               args.val_ratio)

# Define the training process

optimizer = optim.Adam(STmodel.parameters(), lr=1e-5)
loss_criterion = nn.MSELoss()
# loss_criterion = mg_nll
training_nll = []
validation_nll = []
validation_mae = []


def train_epoch(training_input, training_target, batch_size):
    """
    Trains one epoch with the given data.
    :param training_input: Training inputs of shape (num_samples, num_nodes,
    num_timesteps_train, num_features).
    :param training_target: Training targets of shape (num_samples, num_nodes,
    num_timesteps_predict).
    :param batch_size: Batch size to use during training.
    :return: Average loss for this epoch.
    """
    # print(training_input.shape,training_target.shape)
    permutation = torch.randperm(training_input.shape[0])
    epoch_training_losses = []

    for i in range(0, training_input.shape[0], batch_size):
        STmodel.train()
        optimizer.zero_grad()

        indices = permutation[i:i + batch_size]
        X_batch, y_batch = training_input[indices], training_target[indices]
        X_batch = X_batch.to(device=args.device)
        y_batch = y_batch.to(device=args.device)

        # n_train, p_train, pi_train = STmodel(X_batch, A_q, A_h)
        out = STmodel(X_batch, A_q, A_h)
        # out = STmodel(A_wave, X_batch)
        # loss = nb_zeroinflated_nll_loss(y_batch, n_train, p_train, pi_train)

        # stdsx=torch.from_numpy(stds).to(device=device)
        # meansx=torch.from_numpy(means).to(device=device)
        # out_unnormalized = out * stdsx + meansx
        # target_unnormalized = y_batch * stdsx + meansx

        loss = loss_criterion(out, y_batch)
        # loss = loss_criterion(y_batch,out[0],out[1])

        loss.backward()
        optimizer.step()
        epoch_training_losses.append(loss.detach().cpu().numpy())

    return sum(epoch_training_losses) / len(epoch_training_losses)


def val_epoch(val_input, val_target):
    val_input = val_input.to(device=device)
    val_target = val_target.to(device=device)
    val_pred = STmodel(val_input, A_q, A_h)
    # val_pred = STmodel(A_wave, val_input)

    stdsx = torch.from_numpy(stds).to(device=device)
    meansx = torch.from_numpy(means).to(device=device)
    target_unnormalized = val_target * stdsx + meansx
    out_unnormalized = val_pred * stdsx + meansx
    val_loss = loss_criterion(val_pred, val_target)
    # val_loss = loss_criterion(val_target,val_pred[0],val_pred[1])
    validation_nll.append(val_loss.cpu().detach().numpy().item())

    val_pred = out_unnormalized.cpu()
    val_target = target_unnormalized.cpu()

    mae = np.absolute(val_pred - val_target).mean()

    rmse = np.sqrt(np.square(val_pred - val_target).mean())

    mape = np.abs(val_pred - val_target / val_target).mean()

    # mae = np.mean(np.absolute(out_unnormalized - target_unnormalized))

    return mae, rmse, mape


def model_test(test_input, test_target):
    STmodel = torch.load(args.best_model_path).to(device=device)
    STmodel.load_state_dict(torch.load(args.best_model_path, map_location='cpu').state_dict())
    STmodel.eval()

    with torch.no_grad():
        test_input = test_input.to(device=device)
        test_target = test_target.to(device=device)
        # print(test_input.is_cuda, A_q.is_cuda, A_h.is_cuda)

        test_loss_all = []
        test_pred_all = np.zeros_like(test_target.cpu())

        # print(test_input.shape, test_target.shape)
        for i in range(0, test_input.shape[0], args.batch_size):
            x_batch = test_input[i:i + args.batch_size].to(device=device)
            pred = STmodel(x_batch, A_q, A_h).cpu()

            test_loss = loss_criterion(pred, test_target[i:i + args.batch_size].cpu()).cpu().detach().numpy().item()

            # test_loss = nb_zeroinflated_nll_loss(test_target[i:i + batch_size], n_test, p_test, pi_test).to(device="cpu")
            # test_loss = np.ndarray.item(test_loss.detach().numpy())
            #
            # mean_pred = (1 - pi_test.detach().cpu().numpy()) * (
            #             n_test.detach().cpu().numpy() / p_test.detach().cpu().numpy() - n_test.detach().cpu().numpy())

            test_pred_all[i:i + args.batch_size] = pred
            test_loss_all.append(test_loss)
        # The error of each horizon
        mae_list = []
        rmse_list = []
        mape_list = []

        # test_pred_all= test_pred_all * stds + means
        # test_target = test_target.cpu() * stds + means

        # np.save("pth/pred.npy", test_pred_all)
        # np.save("pth/target.npy", test_target.cpu())

        # for horizon in range(test_pred_all.shape[2]):
        #     # mae = np.mean(np.abs(pred - test_target.detach().cpu().numpy()))
        #     mae = np.mean(np.abs(test_pred_all[:, :, horizon,:]- test_target[:, :, horizon, :].detach().cpu().numpy()))
        #
        #     rmse = np.sqrt(np.square(np.mean(test_pred_all[:, :, horizon,:] - test_target[:, :, horizon, :].detach().cpu().numpy())))
        #
        #     mape = np.mean(np.abs((test_pred_all[:, :, horizon, :] - test_target[:, :, horizon, :].detach().cpu().numpy())) /
        #                    np.abs((test_target[:, :, horizon, :].detach().cpu().numpy()+0.5)))
        #
        #     mae_list.append(mae)
        #     rmse_list.append(rmse)
        #     mape_list.append(mape)
        #     logger.info('Horizon %d MAE:%.4f RMSE:%.4f MAPE:%.4f' % (horizon, mae, rmse, mape))
        #     logger.info('BestModel %s overall score: NLL %.5f; mae %.4f; rmse %.4f; mape %.4f' % (
        #     args.best_model_path, test_loss, np.mean(mae_list), np.mean(rmse_list), np.mean(mape_list)))
        test_target = test_target.detach().cpu().numpy()
        mae = np.mean(np.abs(test_pred_all - test_target))
        rmse = np.sqrt(np.square(np.mean(test_pred_all - test_target)))
        mape = np.mean(np.abs((test_pred_all - test_target) / test_target))

        mae_list.append(mae)
        rmse_list.append(rmse)
        mape_list.append(mape)
        logger.info('Horizon %d MAE:%.4f RMSE:%.4f MAPE:%.4f' % (1, mae, rmse, mape))
        logger.info('BestModel %s overall score: NLL %.5f; mae %.4f; rmse %.4f; mape %.4f' % (
            args.best_model_path, test_loss, np.mean(mae_list), np.mean(rmse_list), np.mean(mape_list)))


if __name__ == '__main__':
    for epoch in range(args.epochs):
        ## Step 1, training
        """
        # Begin training, similar training procedure from STGCN
        Trains one epoch with the given data.
        :param training_input: Training inputs of shape (num_samples, num_nodes,
        num_timesteps_train, num_features).
        :param training_target: Training targets of shape (num_samples, num_nodes,
        num_timesteps_predict).
        :param batch_size: Batch size to use during training.
        """
        loss = train_epoch(training_input, training_target, batch_size=args.batch_size)
        training_nll.append(loss)

        ## Step 2, validation
        with torch.no_grad():
            STmodel.eval()
            mae, rmse, mape = val_epoch(val_input, val_target)
            validation_mae.append(mae)
            val_input = val_input.to(device="cpu")
            val_target = val_target.to(device="cpu")

        # logger.info('Epoch: {}'.format(epoch))
        # logger.info("Training loss: {}".format(training_nll[-1]))
        logger.info('Epoch %d: trainNLL %.4f; valNLL %.4f; mae %.4f; rmse %.4f; mape %.4f' % (
            epoch, training_nll[-1], validation_nll[-1], validation_mae[-1], rmse, mape))

        if training_nll[-1].item() == min(training_nll):
            best_model = copy.deepcopy(STmodel.state_dict())
        if not os.path.exists(args.checkpoint_path):
            os.makedirs(args.checkpoint_path)
        with open(args.losses_p, "wb") as fd:
            pickle.dump((training_nll, validation_nll, validation_mae), fd)
        if np.isnan(training_nll[-1]):
            break

        STmodel.load_state_dict(best_model)
        torch.save(STmodel, args.best_model_path)

    model_test(val_input, val_target)
    # model_test(test_input, test_target)
