from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

        # -------- static configs --------
        self.sparse_lambda = getattr(args, 'sparse_lambda', 0.0)
        self.huber_delta = getattr(args, 'huber_delta', 1.0)

        # -------- dynamic RL configs --------
        self.use_dynamic_rl = getattr(args, 'use_dynamic_rl', 0)
        self.eta_a = getattr(args, 'eta_a', 0.0)
        self.eta_c = getattr(args, 'eta_c', 0.0)
        self.rl_warmup_epochs = getattr(args, 'rl_warmup_epochs', 0)
        self.reward_type = getattr(args, 'reward_type', 'improve_over_static')
        self.reward_scale = getattr(args, 'reward_scale', 1.0)

    def _build_model(self):
        model = self.model_dict[self.args.model](self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        loss_name = str(getattr(self.args, 'loss', 'MSE')).lower()

        if loss_name == 'mse':
            criterion = nn.MSELoss()
        elif loss_name == 'mae':
            criterion = nn.L1Loss()
        elif loss_name == 'huber':
            criterion = nn.HuberLoss(delta=self.huber_delta)
        else:
            raise ValueError(f"Unsupported loss type: {self.args.loss}")
        return criterion

    def _parse_model_output(self, model_output):
        """
        Supports:
            1) outputs
            2) (outputs, aux)
        """
        if isinstance(model_output, tuple):
            outputs, aux = model_output
        else:
            outputs, aux = model_output, {}
        return outputs, aux

    def _get_rl_weights(self, epoch):
        """
        RL warmup:
            before warmup epochs, disable RL losses
        """
        if epoch < self.rl_warmup_epochs:
            return 0.0, 0.0
        return self.eta_a, self.eta_c

    def _compute_reward(self, static_loss_var, dynamic_loss_var):
        """
        static_loss_var:  [B, C]
        dynamic_loss_var: [B, C]

        return:
            reward: [B, C]
        """
        if self.reward_type == 'improve_over_static':
            reward = static_loss_var - dynamic_loss_var
        elif self.reward_type == 'neg_dynamic_loss':
            reward = -dynamic_loss_var
        elif self.reward_type == 'normalized_improve_over_static':
            reward = (static_loss_var - dynamic_loss_var) / (torch.abs(static_loss_var) + 1e-8)
        else:
            raise ValueError(f"Unsupported reward_type: {self.reward_type}")

        reward = self.reward_scale * reward
        return reward

    def _align_aux_with_target_dim(self, aux_tensor, target_tensor):
        """
        Align aux outputs (e.g., static_pred / value / action_strength)
        with target output dimensionality after trainer-side f_dim slicing.

        aux_tensor examples:
            static_pred      [B, pred_len, N]
            value            [B, N, 1] or [B, N]
            action_strength  [B, N]
        target_tensor:
            batch_y_target or dynamic_pred after f_dim slicing

        Returns:
            aligned aux_tensor compatible with target output dimension C_tgt.
        """
        if aux_tensor is None:
            return None

        target_c = target_tensor.shape[-1]

        # Case 1: prediction-like tensor [B, T, C]
        if aux_tensor.dim() == 3 and aux_tensor.shape[1] == target_tensor.shape[1]:
            aux_c = aux_tensor.shape[-1]
            if aux_c == target_c:
                return aux_tensor
            # mimic the same slicing rule used in trainer outputs
            if self.args.features == 'MS':
                return aux_tensor[:, :, -target_c:]
            else:
                return aux_tensor[:, :, :target_c]

        # Case 2: per-variable tensor [B, C]
        if aux_tensor.dim() == 2:
            aux_c = aux_tensor.shape[-1]
            if aux_c == target_c:
                return aux_tensor
            if self.args.features == 'MS':
                return aux_tensor[:, -target_c:]
            else:
                return aux_tensor[:, :target_c]

        # Case 3: per-variable tensor [B, C, 1]
        if aux_tensor.dim() == 3 and aux_tensor.shape[-1] == 1:
            aux_c = aux_tensor.shape[1]
            if aux_c == target_c:
                return aux_tensor
            if self.args.features == 'MS':
                return aux_tensor[:, -target_c:, :]
            else:
                return aux_tensor[:, :target_c, :]

        return aux_tensor

    def _compute_rl_losses(self, aux, dynamic_pred, batch_y_target, criterion):
        """
        Compute RL losses using real forecasting targets.

        Inputs:
            aux should contain:
                - static_pred: [B, pred_len, C_full] or aligned
                - value: [B, N, 1] or compatible
                - action_strength: [B, N]
        Returns:
            rl_items dict
        """
        required_keys = ['static_pred', 'value', 'action_strength']
        if not all(k in aux for k in required_keys):
            zero = dynamic_pred.new_tensor(0.0)
            return {
                'static_sup_loss': zero,
                'dynamic_sup_loss': criterion(dynamic_pred, batch_y_target),
                'reward': zero,
                'advantage': zero,
                'actor_loss': zero,
                'critic_loss': zero,
                'reward_mean': 0.0,
                'advantage_mean': 0.0
            }

        static_pred = aux['static_pred']
        value = aux['value']
        action_strength = aux['action_strength']

        # -------- IMPORTANT FIX: align aux tensors to trainer-side target dimensionality --------
        static_pred = self._align_aux_with_target_dim(static_pred, batch_y_target)
        value = self._align_aux_with_target_dim(value, batch_y_target)
        action_strength = self._align_aux_with_target_dim(action_strength, batch_y_target)

        # Per-variable forecasting loss over prediction horizon
        # [B, pred_len, C] -> mean over time dim => [B, C]
        if isinstance(criterion, nn.MSELoss):
            static_loss_var = ((static_pred - batch_y_target) ** 2).mean(dim=1)
            dynamic_loss_var = ((dynamic_pred - batch_y_target) ** 2).mean(dim=1)
        elif isinstance(criterion, nn.L1Loss):
            static_loss_var = torch.abs(static_pred - batch_y_target).mean(dim=1)
            dynamic_loss_var = torch.abs(dynamic_pred - batch_y_target).mean(dim=1)
        elif isinstance(criterion, nn.HuberLoss):
            delta = self.huber_delta
            static_diff = torch.abs(static_pred - batch_y_target)
            dynamic_diff = torch.abs(dynamic_pred - batch_y_target)

            static_loss_elem = torch.where(
                static_diff < delta,
                0.5 * static_diff ** 2,
                delta * (static_diff - 0.5 * delta)
            )
            dynamic_loss_elem = torch.where(
                dynamic_diff < delta,
                0.5 * dynamic_diff ** 2,
                delta * (dynamic_diff - 0.5 * delta)
            )

            static_loss_var = static_loss_elem.mean(dim=1)   # [B, C]
            dynamic_loss_var = dynamic_loss_elem.mean(dim=1) # [B, C]
        else:
            # fallback
            static_loss_var = ((static_pred - batch_y_target) ** 2).mean(dim=1)
            dynamic_loss_var = ((dynamic_pred - batch_y_target) ** 2).mean(dim=1)

        # Scalar supervised losses for logging
        static_sup_loss = static_loss_var.mean()
        dynamic_sup_loss = dynamic_loss_var.mean()

        # Reward: [B, C]
        reward = self._compute_reward(static_loss_var.detach(), dynamic_loss_var.detach())

        # Critic output: [B, C, 1] or [B, C]
        if value.dim() == 3:
            value_pred = value.squeeze(-1)
        else:
            value_pred = value

        # Lightweight advantage baseline
        advantage = reward - value_pred

        # Surrogate actor objective
        actor_loss = -(advantage.detach() * action_strength).mean()

        # Critic objective
        critic_loss = nn.functional.mse_loss(value_pred, reward.detach())

        return {
            'static_sup_loss': static_sup_loss,
            'dynamic_sup_loss': dynamic_sup_loss,
            'reward': reward,
            'advantage': advantage,
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'reward_mean': reward.detach().mean().item(),
            'advantage_mean': advantage.detach().mean().item()
        }

    def _compute_total_loss(self, outputs, batch_y, criterion, aux=None, epoch=0, return_items=False):
        """
        outputs: [B, pred_len, C]
        batch_y: [B, pred_len, C]
        aux: dict from model output
        """
        if aux is None:
            aux = {}

        sup_loss = criterion(outputs, batch_y)

        sparse_loss = aux.get('sparse_loss', sup_loss.new_tensor(0.0))

        # -------- trainer-side RL loss computation --------
        rl_items = self._compute_rl_losses(aux, outputs, batch_y, criterion)
        actor_loss = rl_items['actor_loss']
        critic_loss = rl_items['critic_loss']

        eta_a_eff, eta_c_eff = self._get_rl_weights(epoch)

        if not self.use_dynamic_rl:
            eta_a_eff = 0.0
            eta_c_eff = 0.0

        total_loss = (
            sup_loss
            + self.sparse_lambda * sparse_loss
            + eta_a_eff * actor_loss
            + eta_c_eff * critic_loss
        )

        if return_items:
            return total_loss, {
                'sup_loss': sup_loss.detach().item(),
                'static_sup_loss': rl_items['static_sup_loss'].detach().item() if torch.is_tensor(rl_items['static_sup_loss']) else 0.0,
                'dynamic_sup_loss': rl_items['dynamic_sup_loss'].detach().item() if torch.is_tensor(rl_items['dynamic_sup_loss']) else sup_loss.detach().item(),
                'sparse_loss': sparse_loss.detach().item() if torch.is_tensor(sparse_loss) else float(sparse_loss),
                'actor_loss': actor_loss.detach().item() if torch.is_tensor(actor_loss) else float(actor_loss),
                'critic_loss': critic_loss.detach().item() if torch.is_tensor(critic_loss) else float(critic_loss),
                'total_loss': total_loss.detach().item(),
                'reward_mean': rl_items['reward_mean'],
                'advantage_mean': rl_items['advantage_mean'],
                'eta_a_eff': eta_a_eff,
                'eta_c_eff': eta_c_eff
            }

        return total_loss

    def vali(self, vali_data, vali_loader, criterion):
        total_sup_loss = []
        total_static_sup_loss = []
        total_dynamic_sup_loss = []
        total_sparse_loss = []
        total_actor_loss = []
        total_critic_loss = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        model_output = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        outputs, aux = self._parse_model_output(model_output)
                else:
                    model_output = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    outputs, aux = self._parse_model_output(model_output)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y_target = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                sup_loss = criterion(outputs, batch_y_target)
                sparse_loss = aux.get('sparse_loss', sup_loss.new_tensor(0.0))
                rl_items = self._compute_rl_losses(aux, outputs, batch_y_target, criterion)

                total_sup_loss.append(sup_loss.item())
                total_static_sup_loss.append(
                    rl_items['static_sup_loss'].item() if torch.is_tensor(rl_items['static_sup_loss']) else 0.0
                )
                total_dynamic_sup_loss.append(
                    rl_items['dynamic_sup_loss'].item() if torch.is_tensor(rl_items['dynamic_sup_loss']) else sup_loss.item()
                )
                total_sparse_loss.append(sparse_loss.item() if torch.is_tensor(sparse_loss) else float(sparse_loss))
                total_actor_loss.append(rl_items['actor_loss'].item() if torch.is_tensor(rl_items['actor_loss']) else 0.0)
                total_critic_loss.append(rl_items['critic_loss'].item() if torch.is_tensor(rl_items['critic_loss']) else 0.0)

        avg_sup_loss = np.average(total_sup_loss)
        avg_static_sup = np.average(total_static_sup_loss)
        avg_dynamic_sup = np.average(total_dynamic_sup_loss)
        avg_sparse_loss = np.average(total_sparse_loss)
        avg_actor_loss = np.average(total_actor_loss)
        avg_critic_loss = np.average(total_critic_loss)

        self.model.train()

        print(
            "Validation | Sup: {:.7f} StaticSup: {:.7f} DynamicSup: {:.7f} Sparse: {:.7f} Actor: {:.7f} Critic: {:.7f}".format(
                avg_sup_loss, avg_static_sup, avg_dynamic_sup, avg_sparse_loss, avg_actor_loss, avg_critic_loss
            )
        )

        # Early stopping still uses supervised forecasting quality
        return avg_sup_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0

            train_total_loss = []
            train_sup_loss = []
            train_static_sup_loss = []
            train_dynamic_sup_loss = []
            train_sparse_loss = []
            train_actor_loss = []
            train_critic_loss = []
            train_reward_mean = []
            train_advantage_mean = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        model_output = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        outputs, aux = self._parse_model_output(model_output)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y_target = batch_y[:, -self.args.pred_len:, f_dim:]

                        loss, loss_items = self._compute_total_loss(
                            outputs, batch_y_target, criterion, aux=aux, epoch=epoch, return_items=True
                        )
                else:
                    model_output = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    outputs, aux = self._parse_model_output(model_output)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y_target = batch_y[:, -self.args.pred_len:, f_dim:]

                    loss, loss_items = self._compute_total_loss(
                        outputs, batch_y_target, criterion, aux=aux, epoch=epoch, return_items=True
                    )

                train_total_loss.append(loss_items['total_loss'])
                train_sup_loss.append(loss_items['sup_loss'])
                train_static_sup_loss.append(loss_items['static_sup_loss'])
                train_dynamic_sup_loss.append(loss_items['dynamic_sup_loss'])
                train_sparse_loss.append(loss_items['sparse_loss'])
                train_actor_loss.append(loss_items['actor_loss'])
                train_critic_loss.append(loss_items['critic_loss'])
                train_reward_mean.append(loss_items['reward_mean'])
                train_advantage_mean.append(loss_items['advantage_mean'])

                if (i + 1) % 100 == 0:
                    print(
                        "\titers: {0}, epoch: {1} | total: {2:.7f} | sup: {3:.7f} | static_sup: {4:.7f} | dynamic_sup: {5:.7f} | sparse: {6:.7f} | actor: {7:.7f} | critic: {8:.7f} | reward: {9:.7f} | adv: {10:.7f}".format(
                            i + 1,
                            epoch + 1,
                            loss_items['total_loss'],
                            loss_items['sup_loss'],
                            loss_items['static_sup_loss'],
                            loss_items['dynamic_sup_loss'],
                            loss_items['sparse_loss'],
                            loss_items['actor_loss'],
                            loss_items['critic_loss'],
                            loss_items['reward_mean'],
                            loss_items['advantage_mean']
                        )
                    )
                    print("\tRL weights -> eta_a_eff: {:.6f}, eta_c_eff: {:.6f}".format(
                        loss_items['eta_a_eff'], loss_items['eta_c_eff']
                    ))

                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))

                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            train_total_avg = np.average(train_total_loss)
            train_sup_avg = np.average(train_sup_loss)
            train_static_sup_avg = np.average(train_static_sup_loss)
            train_dynamic_sup_avg = np.average(train_dynamic_sup_loss)
            train_sparse_avg = np.average(train_sparse_loss)
            train_actor_avg = np.average(train_actor_loss)
            train_critic_avg = np.average(train_critic_loss)
            train_reward_avg = np.average(train_reward_mean)
            train_adv_avg = np.average(train_advantage_mean)

            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Total: {2:.7f} Sup: {3:.7f} StaticSup: {4:.7f} DynamicSup: {5:.7f} "
                "Sparse: {6:.7f} Actor: {7:.7f} Critic: {8:.7f} Reward: {9:.7f} Adv: {10:.7f} | "
                "Vali Sup: {11:.7f} Test Sup: {12:.7f}".format(
                    epoch + 1,
                    train_steps,
                    train_total_avg,
                    train_sup_avg,
                    train_static_sup_avg,
                    train_dynamic_sup_avg,
                    train_sparse_avg,
                    train_actor_avg,
                    train_critic_avg,
                    train_reward_avg,
                    train_adv_avg,
                    vali_loss,
                    test_loss
                )
            )

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        model_output = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        outputs, aux = self._parse_model_output(model_output)
                else:
                    model_output = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    outputs, aux = self._parse_model_output(model_output)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

                if i % 20 == 0:
                    input_arr = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input_arr.shape
                        input_arr = test_data.inverse_transform(
                            input_arr.reshape(shape[0] * shape[1], -1)
                        ).reshape(shape)

                    gt = np.concatenate((input_arr[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input_arr[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw_value = np.array(dtw_list).mean()
        else:
            dtw_value = 'Not calculated'

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw_value))

        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw_value))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
