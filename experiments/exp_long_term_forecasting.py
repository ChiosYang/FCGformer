import logging

from tqdm import tqdm

import utils
from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, load_adj
from utils.metrics import metric
from utils.resource_monitor import ResourceMonitor
from utils.lightweight_monitor import LightweightModelMonitor
import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        # 根据 args.model 从 self.model_dict 中选择模型类，并使用 args 初始化这个模型，然后将其转换为浮点型（float()）
        model = self.model_dict[self.args.model].Model(self.args).float()
        # 是否使用多GPU
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        # 获取数据集与数据加载器
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # 选择模型优化器，使用的是Adam
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=1e-4)
        return model_optim

    def _select_criterion(self):
        # 选择损失函数，使用了MSE
        criterion = nn.MSELoss()
        return criterion

    def setup_logging(self,log_file):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    # 计算模型损失
    def vali(self, vali_data, vali_loader, criterion):

        total_loss = []
        self.model.eval()

        # 禁用梯度计算，在这个过程中不需要更新模型权重，因此可以禁用梯度计算来节省内存并加快计算速度
        with torch.no_grad():
            # 遍历 验证集/测试集 的批量数据
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,batch_location) in tqdm(enumerate(vali_loader),total=len(vali_loader), desc="Loading", disable=True):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                # 对于 PEMS 和 Solar 数据集处理

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                batch_location = batch_location.float().to(self.device)
                epoch = 0
                # dec_inp 是解码器的输入，先初始化为与 batch_y 相同形状的零张量，截取 batch_y 的最后 pred_len 个时间步
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                # 将 batch_y 中的前 label_len 个时间步与零张量拼接，作为解码器的最终输入
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                # 检查是否使用 自动混合精度 use_amp
                if self.args.use_amp:
                    # 如果是，则在 torch.cuda.amp.autocast() 下进行前向传播
                    with torch.cuda.amp.autocast():
                        # 检查是否输出注意力信息
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs,frequency_test = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,batch_location,epoch)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        # outputs,frequency_test1,frequency_test2,frequency_test3,frequency_test4,frequency_test5 = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,batch_location,epoch)
                        outputs= self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,batch_location)


                # 根据 features 参数确定要使用的维度，通常 f_dim 为 0 或 -1
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)

        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        # 设置log
        log_file = os.path.join(self.args.checkpoints, setting, 'training_log.txt')
        # print(log_file)
        self.setup_logging(log_file)
        
        # 创建统一的light_result目录结构 - 使用简短的目录名
        # 从setting中提取关键信息：模型ID_模型类型_预测长度
        setting_parts = setting.split('_')
        if len(setting_parts) >= 3:
            short_name = f"{setting_parts[0]}_{setting_parts[1]}_pl{self.args.pred_len}"
        else:
            short_name = setting[:30]  # 如果解析失败，截取前30个字符
        
        light_result_base = os.path.join('.', 'light_result', short_name)
        
        # 初始化资源监控器
        monitor_path = os.path.join(light_result_base, 'resource_logs')
        resource_monitor = ResourceMonitor(log_interval=2.0, save_path=monitor_path)
        
        # 初始化轻量化分析监控器
        lightweight_path = os.path.join(light_result_base, 'lightweight_analysis')
        lightweight_monitor = LightweightModelMonitor(save_path=lightweight_path)
        
        # 获取模型信息
        model_info = resource_monitor.get_model_info(self.model)
        logging.info(f"Model info: {model_info}")
        
        # 开始资源监控
        resource_monitor.start_monitoring()

        time_now = time.time()

        train_steps = len(train_loader)
       #  print('train_steps : ',train_steps)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_location) in tqdm(enumerate(train_loader),total=len(train_loader), desc="Loading", disable=True):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                # print('batch_x : ',batch_x.size())
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                batch_location =  batch_location.float().to(self.device)
                # print('batch_location : ',batch_location.size())


                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            print(self.model)
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,batch_location)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        # outputs,frequency_out1,frequency_out2,frequency_out3,frequency_out4,frequency_out5 = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,batch_location,epoch)
                        outputs= self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,batch_location)


                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    log_msg = "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item())
                    logging.info(log_msg)
                    # print(log_msg)

                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)

                    log_msg = '\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time)
                    logging.info(log_msg)
                    # print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))

                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(self.args, max_norm=1.0)
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            epoch_duration = time.time() - epoch_time
            print("Epoch: {} cost time: {}".format(epoch + 1, epoch_duration))
            
            # 记录epoch时间
            resource_monitor.record_epoch_time(epoch + 1, epoch_duration)
            
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            # 原文使用的，计算test_loss耗时太久；且test_loss并没有在后续使用过
            # test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
               epoch + 1, train_steps, train_loss, vali_loss))

            log_msg = "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss)
            logging.info(log_msg)
            # print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
            #     epoch + 1, train_steps, train_loss, vali_loss))

            # # 在每个 epoch 结束时提取权重
            # weights = self.model.get_weights()
            #
            # # 输出 FGNN 的权重
            # print(f"FGNN weights after epoch {epoch + 1}:")
            # for name, param in weights['FGNN_weights'].items():
            #     print(f"{name}: {param.shape}")
            #
            # # 输出 iTransformer 的权重
            # print(f"iTransformer weights after epoch {epoch + 1}:")
            # for name, param in weights['iTransformer_weights'].items():
            #     print(f"{name}: {param.shape}")
            #
            # # 输出融合层的权重
            # print(f"Fusion Layer weights after epoch {epoch + 1}:")
            # for name, param in weights['Fusion_weights'].items():
            #     print(f"{name}: {param.shape}")
            #
            # # 如果需要保存权重到文件
            # torch.save(weights, f'weights_epoch_{epoch + 1}.pth')

            #### 保存频域输出
            #print(frequency_out)
            # output_dir = "frequency_outputs"
            # os.makedirs(output_dir, exist_ok=True)
            # # 将频域输出转换为 numpy 并保存
            # freq_output1 = frequency_out1.detach().cpu().numpy()
            # freq_output2 = frequency_out2.detach().cpu().numpy()
            # freq_output3 = frequency_out3.detach().cpu().numpy()
            # freq_output4 = frequency_out4.detach().cpu().numpy()
            # freq_output5 = frequency_out5.detach().cpu().numpy()
            #
            # file_path = os.path.join(output_dir, f"frequency_output1_epoch_{epoch}.npy")
            # np.save(file_path, freq_output1)
            # file_path = os.path.join(output_dir, f"frequency_output2_epoch_{epoch}.npy")
            # np.save(file_path, freq_output2)
            # file_path = os.path.join(output_dir, f"frequency_output3_epoch_{epoch}.npy")
            # np.save(file_path, freq_output3)
            # file_path = os.path.join(output_dir, f"frequency_output4_epoch_{epoch}.npy")
            # np.save(file_path, freq_output4)
            # file_path = os.path.join(output_dir, f"frequency_output5_epoch_{epoch}.npy")
            # np.save(file_path, freq_output5)

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

            # get_cka(self.args, setting, self.model, train_loader, self.device, epoch)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        # 停止资源监控并保存报告
        resource_monitor.stop_monitoring()
        
        # 保存资源监控报告
        additional_info = {
            'model_setting': setting,
            'train_epochs': self.args.train_epochs,
            'batch_size': self.args.batch_size,
            'learning_rate': self.args.learning_rate,
            'model_info': model_info
        }
        
        report_path = resource_monitor.save_report(
            model_name=self.args.model + "_" + setting, 
            additional_info=additional_info
        )
        
        # 打印资源监控摘要
        resource_monitor.print_summary()
        
        if report_path:
            logging.info(f"Resource monitoring report saved: {report_path}")
        
        # 先保存轻量化监控器以便在test后使用
        self.lightweight_monitor = lightweight_monitor

        return self.model
    # 对测试数据进行评估
    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        # 从指定路径加载模型的权重（checkpoint.pth）
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
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,batch_location) in tqdm(enumerate(test_loader), total=len(test_loader), desc="Loading Test data", disable=True):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)


                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                batch_location = batch_location.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,batch_location)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,batch_location)
                        # outputs,frequency_test1,frequency_test2,frequency_test3,frequency_test4,frequency_test5 = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,batch_location)

                        #print(f' outputs 123 : {outputs.shape}')

                f_dim = -1 if self.args.features == 'MS' else 0

                #print(f' outputs 0 : {outputs.shape}')
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                #print(f' outputs 1 : {outputs.shape}')
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                #print(f' outputs 2 : {outputs.shape}')

                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    #print(f' batch_y : {batch_y.shape}')
                    #print(f' outputs : {outputs.shape}')
                    outputs = outputs * test_data.scaler.scale_[-1] + test_data.scaler.mean_[-1]
                    batch_y = batch_y * test_data.scaler.scale_[-1] + test_data.scaler.mean_[-1]
                    # outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    # batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                # 每20次生成一次预测值与真实值的图像，保存为pdf，很慢
                # if i % 20 == 0:
                #     input = batch_x.detach().cpu().numpy()
                #     if test_data.scale and self.args.inverse:
                #         shape = input.shape
                #         input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                #     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                #     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                #     visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
        # 将预测值和真实值转换为 NumPy 数组，
        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        # 并调整形状为 (batch_size, pred_len, feature_dim)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # 保存预测结果和评估指标
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        
        # 保存性能指标到对象属性，供轻量化分析使用
        self.final_mae = mae
        self.final_mse = mse
        self.final_rmse = rmse
        self.final_mape = mape
        self.final_mspe = mspe
        
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{},rmse:{}, mape:{}, mspe:{}'.format(mse, mae, rmse, mape, mspe))
        f.write('\n')
        f.write('\n')
        f.close()
        # 将评估指标和预测结果分别保存为 .npy 文件
        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        # 在test完成后进行轻量化分析（现在有性能指标了）
        if hasattr(self, 'lightweight_monitor'):
            print("\n🔬 开始模型轻量化分析...")
            try:
                # 构造输入形状 - 根据你的数据调整
                input_shape = (self.args.seq_len, self.args.enc_in)  # (sequence_length, input_features)
                device = 'cuda' if self.args.use_gpu and torch.cuda.is_available() else 'cpu'
                
                # 现在有性能指标了
                performance_metrics = {
                    'mae': self.final_mae,
                    'mse': self.final_mse,
                    'rmse': self.final_rmse
                }
                
                # 进行轻量化分析
                lightweight_results = self.lightweight_monitor.comprehensive_profile(
                    model=self.model,
                    input_shape=input_shape,
                    batch_size=self.args.batch_size,
                    device=device,
                    performance_metrics=performance_metrics
                )
                
                logging.info("Lightweight model analysis completed with performance metrics")
                
            except Exception as e:
                logging.warning(f"Lightweight analysis failed: {e}")
                print(f"⚠️  轻量化分析失败: {e}")

        return


    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs.detach().cpu().numpy()
                if pred_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = pred_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                preds.append(outputs)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
if __name__ == '__main__':
    Exp = Exp_Long_Term_Forecast
    print(Exp.args)