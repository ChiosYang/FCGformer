import argparse
import torch
from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast
import random
import numpy as np

if __name__ == '__main__':
    fix_seed = 3407
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='iTransformer')

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='是否训练')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='模型ID')
    parser.add_argument('--model', type=str, required=True, default='iTransformer',
                        help='')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='custom', help='数据集类型')
    parser.add_argument('--root_path', type=str, default='./data/electricity/', help='数据文件根路径')
    parser.add_argument('--data_path', type=str, default='electricity.csv', help='数据的CSV文件')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M: 多变量预测多变量, S:单变量预测单变量, MS:多变量预测单变量')
    parser.add_argument('--target', type=str, default='OT',
                        help='预测的目标特征，用在 S 或者 MS 任务中（就是上面那个参数）')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='存储的模型checkpoints位置')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='输入序列长度')
    parser.add_argument('--label_len', type=int, default=48,
                        help='开始标记长度')  # no longer needed in inverted Transformers
    parser.add_argument('--pred_len', type=int, default=96, help='预测序列长度')

    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7,
                        help='output size')  # applicable on arbitrary number of variates in inverted Transformers
    parser.add_argument('--d_model', type=int, default=512, help='模型维度')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='Encoder层数')
    parser.add_argument('--d_layers', type=int, default=1, help='Decoder层数')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='是否在编码器中使用蒸馏，使用该参数表示不使用蒸馏',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='激活函数')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='是否预测未见的未来数据/就是真的做预测试试看')
    # 自己加的Wavenet
    parser.add_argument('--num_nodes', type=int, default=64, help='number of nodes')
    parser.add_argument('--residual_channels', type=int, default=15, help='')
    parser.add_argument('--dilation_channels', type=int, default=15, help='')
    parser.add_argument('--skip_channels', type=int, default=256, help='')
    parser.add_argument('--end_channels', type=int, default=512, help='')
    parser.add_argument('--kernel_size', type=int, default=2, help='')
    parser.add_argument('--blocks', type=int, default=1, help='')
    parser.add_argument('--layers', type=int, default=1, help='')
    parser.add_argument('--adjdata', type=str, default='./dataset/AIR-EW/adj_max.pkl', help='adj data path')
    parser.add_argument('--adjtype', type=str, default='doubletransition', help='adj type')
    parser.add_argument('--randomadj', action='store_true', help='whether random initialize adaptive adj')
    parser.add_argument('--aptonly', action='store_true', help='whether only adaptive adj')
    parser.add_argument('--addaptadj', default=True,action='store_true', help='whether add adaptive adj')
    parser.add_argument('--features_len', type=int, default=15, help='特征数量')


    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader使用的工作线程数')
    parser.add_argument('--itr', type=int, default=1, help='实验次数')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='优化器的学习率')
    parser.add_argument('--des', type=str, default='test', help='实验的描述信息')
    parser.add_argument('--loss', type=str, default='MSE', help='损失函数')
    parser.add_argument('--lradj', type=str, default='type1', help='学习率调整策略')
    parser.add_argument('--use_amp', action='store_true',
                        help='是否使用自动混合精度训练（AMP），可以加速训练并减少显存占用', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2', help='device ids of multile gpus')

    # FGNfourier_layers
    parser.add_argument('--dominance_freq', type=int, default=50, help='')
    parser.add_argument('--fourier_layers', type=int, default=3, help='')

    # iTransformer
    parser.add_argument('--exp_name', type=str, required=False, default='MTSF',
                        help='实验的名称, options:[MTSF, partial_train]')
    parser.add_argument('--channel_independence', type=bool, default=False, help='是否使用通道独立机制')
    parser.add_argument('--inverse', action='store_true', help='是否对输出数据进行反向变换', default=True)
    parser.add_argument('--class_strategy', type=str, default='projection',
                        help='类别策略，可以是 projection、average 或 cls_token，用来指定如何对特征进行分类或投影')
    parser.add_argument('--target_root_path', type=str, default='./data/electricity/',
                        help='数据文件的根路径，指定数据文件所在的目录')
    parser.add_argument('--target_data_path', type=str, default='electricity.csv', help='数据文件的文件名')
    parser.add_argument('--efficient_training', type=bool, default=False,
                        help='是否使用高效训练模式。如果开启，需要与 partial_train 实验模式配合使用。')  # See Figure 8 of our paper for the detail
    parser.add_argument('--use_norm', type=int, default=True, help='是否使用数据的标准化和反标准化操作')
    parser.add_argument('--partial_start_index', type=int, default=0, help='用于部分训练的起始索引，控制模型只在一部分变量上进行训练, '
                                                                           'you can select [partial_start_index, min(enc_in + partial_start_index, N)]')
    
    # TimeMixer参数
    parser.add_argument('--task_name', type=str, default='long_term_forecast', help='')
    parser.add_argument('--down_sampling_window', type=int, default='2', help='')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                    help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--use_future_temporal_feature', type=int, default=0,
                    help='whether to use future_temporal_feature; True 1 False 0')
    parser.add_argument('--down_sampling_method', type=str, default='avg',
                    help='down sampling method, only support avg, max, conv')

    # TESTAM specific parameters
    parser.add_argument('--prob_mul', action='store_true', help='whether to multiply by probability in TESTAM gate')
    parser.add_argument('--max_time_index', type=int, default=288, help='max time index for TESTAM temporal embedding')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des,
                args.class_strategy, ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des,
            args.class_strategy, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
