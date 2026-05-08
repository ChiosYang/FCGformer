import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, List, Optional
import json
import os
import logging
import traceback
import inspect
from collections import defaultdict


class LightweightModelMonitor:
    """
    专门用于模型轻量化研究的监控器
    
    监控指标：
    1. 模型复杂度 (FLOPs, MACs, 参数量)
    2. 内存效率 (模型内存, 激活内存)
    3. 推理性能 (延迟, 吞吐量)
    4. 压缩效果 (压缩比, 稀疏度)
    5. 准确性-效率权衡
    """
    
    def __init__(self, save_path: str = "./lightweight_logs", location_dim: int = 182):
        self.save_path = save_path
        self.metrics = defaultdict(list)
        self.model_info = {}
        os.makedirs(save_path, exist_ok=True)
        self.default_location_dim = location_dim
        
        # 错误和警告记录
        self.error_messages = []
        self.warnings = []
        self.debug_info = []
        
        # 设置日志
        self.logger = logging.getLogger(f"LightweightMonitor_{id(self)}")
        self.logger.setLevel(logging.INFO)
        
        # 检测云服务器环境
        self._detect_cloud_environment()
        self._check_gpu_environment()
        self._print_environment_summary()
    
    def _create_failed_speed_metrics(self, error_msg: str) -> Dict:
        """创建推理失败时的速度指标"""
        return {
            'mean_inference_ms': 0,
            'median_inference_ms': 0,
            'std_inference_ms': 0,
            'min_inference_ms': 0,
            'max_inference_ms': 0,
            'p95_inference_ms': 0,
            'p99_inference_ms': 0,
            'throughput_samples_per_sec': 0,
            'batch_size': 0,
            'successful_runs': 0,
            'device_used': 'unknown',
            'failed': True,
            'error_message': error_msg
        }
    
    def _create_model_inputs(self, model: nn.Module, input_shape: Tuple, device: str):
        """根据模型类型创建合适的输入"""
        batch_size = 1
        pred_len = getattr(model, 'pred_len', None)
        label_len = getattr(model, 'label_len', None)
        
        try:
            # 检查模型类型和模块名称
            model_class_name = model.__class__.__name__
            model_module_name = model.__class__.__module__
            
            self.debug_info.append(f"模型类型: {model_class_name}, 模块: {model_module_name}")
            self.logger.info(f"模型类型: {model_class_name}, 模块: {model_module_name}")
            
            import inspect
            forward_signature = inspect.signature(model.forward)
            param_list = [p for p in forward_signature.parameters.values() if p.name != 'self']
            self.debug_info.append(f"forward方法参数数量: {len(param_list)}")
            self.logger.info(f"forward方法参数数量: {len(param_list)}")
            
            tensor_bank = self._build_timeseries_tensor_bank(
                model=model,
                input_shape=input_shape,
                device=device,
                batch_size=batch_size,
                pred_len=pred_len,
                label_len=label_len,
            )
            
            inputs = self._assemble_inputs_from_signature(param_list, tensor_bank)
            return tuple(inputs)
                
        except Exception as e:
            warning_msg = f"检查模型类型失败，使用默认时间序列格式: {str(e)}"
            self.warnings.append(warning_msg)
            self.logger.warning(warning_msg)
            fallback_bank = self._build_timeseries_tensor_bank(
                model=model,
                input_shape=input_shape,
                device=device,
                batch_size=batch_size,
                pred_len=pred_len,
                label_len=label_len,
            )
            return (fallback_bank['x_enc'], fallback_bank['x_mark_enc'])
    
    def _build_timeseries_tensor_bank(
        self,
        model: nn.Module,
        input_shape: Tuple,
        device: str,
        batch_size: int = 1,
        pred_len: Optional[int] = None,
        label_len: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """创建一个包含常用输入张量的字典"""
        if not isinstance(input_shape, (tuple, list)) or len(input_shape) < 2:
            raise ValueError(f"input_shape格式错误: {input_shape}")

        seq_len, features = int(input_shape[0]), int(input_shape[1])
        pred_len = int(pred_len) if isinstance(pred_len, (int, float)) else None
        label_len = int(label_len) if isinstance(label_len, (int, float)) else None
        if label_len is None or label_len <= 0:
            label_len = max(1, seq_len // 2)
        if pred_len is None or pred_len <= 0:
            pred_len = max(1, seq_len // 4)
        dec_len = label_len + pred_len

        x_enc = torch.randn(batch_size, seq_len, features, device=device)
        x_mark_enc = torch.randn(batch_size, seq_len, 4, device=device)
        x_dec = torch.randn(batch_size, dec_len, features, device=device)
        x_mark_dec = torch.randn(batch_size, dec_len, 4, device=device)
        batch_location = torch.zeros(
            batch_size, seq_len, self._infer_location_dim(model, features), device=device
        )
        mask = torch.zeros(batch_size, seq_len, seq_len, device=device)

        tensor_bank = {
            'x_enc': x_enc,
            'x_mark_enc': x_mark_enc,
            'x_dec': x_dec,
            'x_mark_dec': x_mark_dec,
            'batch_location': batch_location,
            'mask': mask,
        }

        self.debug_info.append(
            "创建时间序列输入: "
            f"x_enc={x_enc.shape}, x_mark_enc={x_mark_enc.shape}, "
            f"x_dec={x_dec.shape}, x_mark_dec={x_mark_dec.shape}, "
            f"batch_location={batch_location.shape}"
        )
        return tensor_bank

    def _infer_location_dim(self, model: nn.Module, fallback_features: int) -> int:
        """尝试根据模型属性推断location维度"""
        attr_candidates = [
            'location_dim', 'num_nodes', 'num_node', 'nodes',
            'node_num', 'station_num', 'features_len'
        ]

        for attr in attr_candidates:
            value = getattr(model, attr, None)
            if isinstance(value, int) and value > 0:
                return value

        for child in model.children():
            for attr in attr_candidates:
                value = getattr(child, attr, None)
                if isinstance(value, int) and value > 0:
                    return value

        return max(self.default_location_dim, fallback_features)

    def _assemble_inputs_from_signature(
        self,
        param_list: List[inspect.Parameter],
        tensor_bank: Dict[str, torch.Tensor]
    ) -> List:
        """根据forward签名生成输入列表"""
        inputs = []
        for param in param_list:
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue

            canonical_key = self._canonical_param_key(param.name)
            tensor_value = tensor_bank.get(canonical_key) if canonical_key else None

            if tensor_value is not None:
                inputs.append(tensor_value)
                continue

            if param.default is not inspect._empty:
                # 可选参数，直接使用默认值（不显式传参）
                continue

            raise ValueError(f"无法为参数 {param.name} 构造输入")

        return inputs

    def _canonical_param_key(self, name: str) -> Optional[str]:
        """根据参数名称推断对应的标准输入"""
        if not name:
            return None

        lname = name.lower()
        alias_map = {
            'x_enc': 'x_enc',
            'inputs': 'x_enc',
            'input': 'x_enc',
            'x': 'x_enc',
            'batch_x': 'x_enc',
            'seq_x': 'x_enc',
            'enc_x': 'x_enc',
            'x_mark_enc': 'x_mark_enc',
            'batch_x_mark': 'x_mark_enc',
            'enc_mark': 'x_mark_enc',
            'x_mark': 'x_mark_enc',
            'x_dec': 'x_dec',
            'dec_x': 'x_dec',
            'decoder_input': 'x_dec',
            'decoderinput': 'x_dec',
            'dec_inp': 'x_dec',
            'future_seq': 'x_dec',
            'x_mark_dec': 'x_mark_dec',
            'dec_mark': 'x_mark_dec',
            'decoder_mark': 'x_mark_dec',
            'batch_y_mark': 'x_mark_dec',
            'y_mark': 'x_mark_dec',
            'batch_location': 'batch_location',
            'seq_location': 'batch_location',
            'location': 'batch_location',
            'loc': 'batch_location',
            'batch_loc': 'batch_location',
            'node_features': 'batch_location',
            'node_emb': 'batch_location',
            'mask': 'mask',
            'attn_mask': 'mask',
        }

        if lname in alias_map:
            return alias_map[lname]

        if 'location' in lname or 'station' in lname or lname.endswith('loc') or 'node' in lname:
            return 'batch_location'

        if 'mark' in lname:
            if 'dec' in lname or 'y' in lname:
                return 'x_mark_dec'
            return 'x_mark_enc'

        if 'dec' in lname and 'mark' not in lname:
            return 'x_dec'

        if lname.startswith('batch_y') and 'mark' not in lname:
            return 'x_dec'

        if 'mask' in lname:
            return 'mask'

        if lname in ('batchx', 'batchdata'):
            return 'x_enc'

        return None

    def _adjust_batch_inputs(self, model_inputs: Tuple, batch_size: int) -> Tuple:
        """根据batch_size调整输入的batch维度"""
        if batch_size <= 1:
            return tuple(model_inputs)

        adjusted = []
        for inp in model_inputs:
            if isinstance(inp, torch.Tensor):
                if inp.dim() > 0 and inp.shape[0] == 1:
                    expanded = inp.expand(batch_size, *inp.shape[1:])
                    adjusted.append(expanded)
                else:
                    adjusted.append(inp)
            else:
                adjusted.append(inp)
        return tuple(adjusted)
    
    def profile_model_complexity(self, model: nn.Module, input_shape: Tuple, 
                               device: str = 'cpu') -> Dict:
        """分析模型复杂度使用torchinfo"""
        try:
            model.eval()
            
            # 计算参数量
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # 计算模型大小(MB)
            model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
            
            # 使用torchinfo估算FLOPs
            flops, macs = self._estimate_flops_with_torchinfo(model, input_shape, device)
            
            # 如果torchinfo失败，尝试备用方案
            if flops == 0:
                flops, macs = self._estimate_flops_fallback(model, input_shape, device)
            
            complexity_metrics = {
                'total_parameters': int(total_params),
                'trainable_parameters': int(trainable_params),
                'model_size_mb': float(model_size_mb),
                'estimated_flops': int(flops),
                'estimated_macs': int(macs),
                'params_density': trainable_params / total_params if total_params > 0 else 0,
                'flops_per_param': flops / total_params if total_params > 0 else 0
            }
            
            # 记录成功信息
            if flops > 0:
                self.logger.info(f"模型复杂度分析成功: FLOPs={flops/1e9:.2f}G, MACs={macs/1e9:.2f}G")
            else:
                self.warnings.append("FLOPs估算失败，返回0")
            
            return complexity_metrics
            
        except Exception as e:
            error_msg = f"模型复杂度分析失败: {str(e)}\n{traceback.format_exc()}"
            self.error_messages.append(error_msg)
            self.logger.error(error_msg)
            
            # 返回基本信息
            try:
                total_params = sum(p.numel() for p in model.parameters())
                model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
                return {
                    'total_parameters': int(total_params),
                    'model_size_mb': float(model_size_mb),
                    'estimated_flops': 0,
                    'estimated_macs': 0,
                    'error': str(e)
                }
            except:
                return {'error': str(e)}
    
    def _estimate_flops_with_torchinfo(self, model: nn.Module, input_shape: Tuple, device: str) -> tuple:
        """使用torchinfo估算FLOPs"""
        try:
            # 尝试导入torchinfo
            from torchinfo import summary
            
            self.logger.info("使用torchinfo进行FLOPs分析...")
            
            # 为模型创建输入
            input_data = self._create_torchinfo_inputs(model, input_shape, device)
            
            # 运行torchinfo分析
            model_info = summary(
                model,
                input_data=input_data,
                device=device,
                verbose=0,  # 不打印详细信息
                col_names=["input_size", "output_size", "num_params", "mult_adds"],
                row_settings=["var_names"],
                mode="eval"
            )
            
            # 获取结果
            total_macs = model_info.total_mult_adds
            total_flops = total_macs * 2  # MACs to FLOPs
            
            self.debug_info.append(f"torchinfo分析成功: MACs={total_macs/1e9:.2f}G, FLOPs={total_flops/1e9:.2f}G")
            
            return total_flops, total_macs
            
        except ImportError:
            error_msg = "torchinfo未安装，请运行: pip install torchinfo"
            self.warnings.append(error_msg)
            self.logger.warning(error_msg)
            return 0, 0
            
        except Exception as e:
            error_msg = f"torchinfo分析失败: {str(e)}"
            self.warnings.append(error_msg)
            self.logger.warning(error_msg)
            return 0, 0
    
    def _create_torchinfo_inputs(self, model: nn.Module, input_shape: Tuple, device: str):
        """为torchinfo创建输入数据"""
        try:
            return self._create_model_inputs(model, input_shape, device)
        except Exception as e:
            # 默认使用时间序列格式
            self.warnings.append(f"输入创建失败，使用默认格式: {str(e)}")
            tensor_bank = self._build_timeseries_tensor_bank(
                model=model,
                input_shape=input_shape,
                device=device,
                batch_size=1
            )
            return (tensor_bank['x_enc'], tensor_bank['x_mark_enc'])
    
    def _estimate_flops_fallback(self, model: nn.Module, input_shape: Tuple, device: str) -> tuple:
        """备用FLOPs估算方案"""
        try:
            # 尝试使用thop
            from thop import profile
            
            self.logger.info("使用thop作为备用方案...")
            
            # 创建输入
            sample_inputs = self._create_model_inputs(model, input_shape, device)
            
            # 运行thop分析
            flops, params = profile(model, inputs=sample_inputs, verbose=False)
            
            macs = flops / 2  # FLOPs to MACs
            
            self.debug_info.append(f"thop分析成功: FLOPs={flops/1e9:.2f}G, MACs={macs/1e9:.2f}G")
            
            return flops, macs
            
        except ImportError:
            error_msg = "thop未安装，请运行: pip install thop"
            self.warnings.append(error_msg)
            self.logger.warning(error_msg)
            
            # 最后的备用方案：手动hook
            return self._estimate_flops_manual(model, input_shape, device)
            
        except Exception as e:
            error_msg = f"thop分析失败: {str(e)}"
            self.warnings.append(error_msg)
            self.logger.warning(error_msg)
            
            # 最后的备用方案：手动hook
            return self._estimate_flops_manual(model, input_shape, device)
    
    def _estimate_flops_manual(self, model: nn.Module, input_shape: Tuple, device: str) -> tuple:
        """手动hook方案（最后的备用）"""
        try:
            self.logger.info("使用手动hook方案...")
            
            sample_inputs = self._create_model_inputs(model, input_shape, device)
            flops = self._estimate_flops_with_hooks(model, sample_inputs)
            
            macs = flops / 2
            
            if flops > 0:
                self.debug_info.append(f"手动hook分析成功: FLOPs={flops/1e9:.2f}G")
            else:
                self.warnings.append("手动hook方案也失败了")
            
            return flops, macs
            
        except Exception as e:
            error_msg = f"手动hook方案失败: {str(e)}"
            self.warnings.append(error_msg)
            self.logger.warning(error_msg)
            return 0, 0
    
    def _estimate_flops_with_hooks(self, model: nn.Module, model_inputs: Tuple) -> int:
        """简化的FLOPs估算"""
        model.eval()
        flops = 0
        
        def flop_count_hook(module, input, output):
            nonlocal flops
            
            if isinstance(module, nn.Linear):
                # Linear层: input_size * output_size
                flops += input[0].numel() * module.out_features
                
            elif isinstance(module, nn.Conv2d):
                # Conv2d层: kernel_size * input_channels * output_channels * output_height * output_width
                kernel_flops = module.kernel_size[0] * module.kernel_size[1]
                output_elements = output.numel()
                flops += kernel_flops * module.in_channels * output_elements
                
            elif isinstance(module, (nn.ReLU, nn.GELU, nn.Tanh, nn.Sigmoid, nn.LeakyReLU)):
                # 激活函数
                if isinstance(input[0], torch.Tensor):
                    flops += input[0].numel()
                
            elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                # 归一化层
                if isinstance(input[0], torch.Tensor):
                    flops += input[0].numel() * 2  # mean + std
        
        # 注册钩子
        hooks = []
        for module in model.modules():
            hooks.append(module.register_forward_hook(flop_count_hook))
        
        # 前向传播
        try:
            with torch.no_grad():
                _ = model(*model_inputs)
        except Exception as e:
            warning_msg = f"FLOPs估算失败: {str(e)}"
            self.warnings.append(warning_msg)
            self.logger.warning(warning_msg)
            # 清理钩子
            for hook in hooks:
                hook.remove()
            return 0
        
        # 清理钩子
        for hook in hooks:
            hook.remove()
        
        return flops
    
    def profile_memory_usage(self, model: nn.Module, input_shape: Tuple,
                           batch_size: int = 1, device: str = 'cpu') -> Dict:
        """分析内存使用"""
        memory_metrics = {}
        
        try:
            model.eval()
            
            # 模型参数内存
            model_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
            
            # 模型缓冲区内存
            buffer_memory = sum(b.numel() * b.element_size() for b in model.buffers()) / (1024**2)
            
            memory_metrics.update({
                'model_memory_mb': float(model_memory),
                'buffer_memory_mb': float(buffer_memory),
                'total_model_memory_mb': float(model_memory + buffer_memory)
            })
            
            # GPU内存监控
            if device == 'cuda' and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
                # 创建输入
                model_inputs = self._create_model_inputs(model, input_shape, device)
                model_inputs = self._adjust_batch_inputs(model_inputs, batch_size)
                
                # 前向传播
                try:
                    with torch.no_grad():
                        _ = model(*model_inputs)
                    
                    # 获取GPU内存统计
                    allocated = torch.cuda.memory_allocated() / (1024**2)
                    cached = torch.cuda.memory_reserved() / (1024**2)
                    max_allocated = torch.cuda.max_memory_allocated() / (1024**2)
                    
                    memory_metrics.update({
                        'gpu_allocated_mb': float(allocated),
                        'gpu_cached_mb': float(cached),
                        'gpu_peak_mb': float(max_allocated),
                        'memory_efficiency': allocated / cached if cached > 0 else 0
                    })
                except Exception as e:
                    warning_msg = f"GPU内存分析失败: {str(e)}"
                    self.warnings.append(warning_msg)
                    self.logger.warning(warning_msg)
                
        except Exception as e:
            error_msg = f"内存分析失败: {str(e)}\n{traceback.format_exc()}"
            self.error_messages.append(error_msg)
            self.logger.error(error_msg)
        
        return memory_metrics
    
    def profile_inference_speed(self, model: nn.Module, input_shape: Tuple,
                              batch_size: int = 1, num_runs: int = 100,
                              warmup_runs: int = 10, device: str = 'cpu') -> Dict:
        """分析推理速度"""
        try:
            model.eval()
            
            self.logger.info(f"开始推理速度测试: device={device}, batch_size={batch_size}, input_shape={input_shape}")
            
            # 检查设备可用性
            if device == 'cuda' and not torch.cuda.is_available():
                error_msg = "CUDA不可用，将切换到CPU"
                self.warnings.append(error_msg)
                self.logger.warning(error_msg)
                device = 'cpu'
            elif device == 'cuda':
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                self.debug_info.append(f"GPU可用内存: {gpu_memory:.1f}GB")
                self.logger.info(f"GPU可用内存: {gpu_memory:.1f}GB")
            
            # 创建输入数据  
            try:
                model_inputs = self._create_model_inputs(model, input_shape, device)
                self.debug_info.append(f"模型输入已创建: {[inp.shape for inp in model_inputs]}")
                self.logger.info(f"模型输入已创建: {[inp.shape for inp in model_inputs]}")
            except Exception as e:
                error_msg = f"创建模型输入失败: {str(e)}"
                self.error_messages.append(error_msg)
                self.logger.error(error_msg)
                return self._create_failed_speed_metrics(error_msg)
            
            model_inputs = self._adjust_batch_inputs(model_inputs, batch_size)
            
            # 预热
            with torch.no_grad():
                for _ in range(warmup_runs):
                    try:
                        _ = model(*model_inputs)
                    except Exception as e:
                        error_msg = f"预热失败: {str(e)}\n{traceback.format_exc()}"
                        self.error_messages.append(error_msg)
                        self.logger.error(error_msg)
                        
                        # 尝试降级处理
                        if device == 'cuda':
                            self.warnings.append("尝试切换到CPU进行推理")
                            return self.profile_inference_speed(model, input_shape, batch_size, num_runs, warmup_runs, 'cpu')
                        elif batch_size > 1:
                            self.warnings.append("尝试减小batch_size到1")
                            return self.profile_inference_speed(model, input_shape, 1, num_runs, warmup_runs, device)
                        
                        return self._create_failed_speed_metrics(error_msg)
            
            # 同步GPU（如果使用）
            if device == 'cuda':
                torch.cuda.synchronize()
            
            # 测量推理时间
            inference_times = []
            
            for _ in range(num_runs):
                start_time = time.perf_counter()
                
                with torch.no_grad():
                    try:
                        _ = model(*model_inputs)
                    except Exception as e:
                        error_msg = f"推理失败: {str(e)}\n{traceback.format_exc()}"
                        self.error_messages.append(error_msg)
                        self.logger.error(error_msg)
                        
                        # 如果已经有一些成功的运行，使用现有数据
                        if len(inference_times) > 0:
                            self.warnings.append(f"推理部分失败，使用{len(inference_times)}次成功运行的数据")
                            break
                        
                        # 如果一次都没成功，尝试降级
                        if device == 'cuda':
                            self.warnings.append("尝试切换到CPU进行推理")
                            return self.profile_inference_speed(model, input_shape, batch_size, num_runs, warmup_runs, 'cpu')
                        elif batch_size > 1:
                            self.warnings.append("尝试减小batch_size到1")
                            return self.profile_inference_speed(model, input_shape, 1, num_runs, warmup_runs, device)
                        
                        return self._create_failed_speed_metrics(error_msg)
                
                if device == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                inference_times.append((end_time - start_time) * 1000)  # 转换为毫秒
            
            # 计算统计数据
            if len(inference_times) == 0:
                error_msg = "没有成功的推理运行"
                self.error_messages.append(error_msg)
                return self._create_failed_speed_metrics(error_msg)
            
            inference_times = np.array(inference_times)
            actual_runs = len(inference_times)
            
            if actual_runs < num_runs:
                warning_msg = f"实际运行次数({actual_runs})少于预期({num_runs})"
                self.warnings.append(warning_msg)
                self.logger.warning(warning_msg)
            
            speed_metrics = {
                'mean_inference_ms': float(np.mean(inference_times)),
                'median_inference_ms': float(np.median(inference_times)),
                'std_inference_ms': float(np.std(inference_times)),
                'min_inference_ms': float(np.min(inference_times)),
                'max_inference_ms': float(np.max(inference_times)),
                'p95_inference_ms': float(np.percentile(inference_times, 95)),
                'p99_inference_ms': float(np.percentile(inference_times, 99)),
                'throughput_samples_per_sec': float(batch_size * 1000 / np.mean(inference_times)),
                'batch_size': batch_size,
                'successful_runs': actual_runs,
                'device_used': device
            }
            
            self.logger.info(f"推理速度测试成功: 平均延迟 {speed_metrics['mean_inference_ms']:.2f}ms")
            return speed_metrics
            
        except Exception as e:
            error_msg = f"推理速度测试全部失败: {str(e)}\n{traceback.format_exc()}"
            self.error_messages.append(error_msg)
            self.logger.error(error_msg)
            return self._create_failed_speed_metrics(error_msg)
    
    def calculate_sparsity(self, model: nn.Module) -> Dict:
        """计算模型稀疏度"""
        total_params = 0
        zero_params = 0
        
        sparsity_by_layer = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_count = param.numel()
                zero_count = (param.abs() < 1e-8).sum().item()
                
                total_params += param_count
                zero_params += zero_count
                
                layer_sparsity = zero_count / param_count * 100
                sparsity_by_layer[name] = {
                    'total_params': param_count,
                    'zero_params': zero_count,
                    'sparsity_percent': layer_sparsity
                }
        
        overall_sparsity = zero_params / total_params * 100 if total_params > 0 else 0
        
        return {
            'overall_sparsity_percent': float(overall_sparsity),
            'total_parameters': total_params,
            'zero_parameters': zero_params,
            'layer_wise_sparsity': sparsity_by_layer
        }
    
    def calculate_efficiency_metrics(self, model_metrics: Dict, 
                                   performance_metrics: Dict) -> Dict:
        """计算效率指标"""
        efficiency_metrics = {}
        
        # 检查性能指标是否有效
        if not performance_metrics or 'mae' not in performance_metrics:
            return efficiency_metrics
        
        mae = performance_metrics['mae']
        
        # 检查MAE是否有效（避免除零错误）
        if mae <= 0 or mae is None:
            print("警告: MAE值无效，跳过效率指标计算")
            return efficiency_metrics
        
        try:
            # 参数效率: 1/MAE per million parameters
            if 'total_parameters' in model_metrics and model_metrics['total_parameters'] > 0:
                params_million = model_metrics['total_parameters'] / 1e6
                if params_million > 0:
                    efficiency_metrics['accuracy_per_param'] = (1.0 / mae) / params_million
            
            # FLOPs效率: 1/MAE per GFLOPs
            if 'estimated_flops' in model_metrics and model_metrics['estimated_flops'] > 0:
                gflops = model_metrics['estimated_flops'] / 1e9
                if gflops > 0:
                    efficiency_metrics['accuracy_per_gflops'] = (1.0 / mae) / gflops
            
            # 内存效率: 1/MAE per MB
            if 'model_size_mb' in model_metrics and model_metrics['model_size_mb'] > 0:
                efficiency_metrics['accuracy_per_mb'] = (1.0 / mae) / model_metrics['model_size_mb']
            
            # 速度效率: 1/MAE per ms
            if 'mean_inference_ms' in model_metrics and model_metrics['mean_inference_ms'] > 0:
                efficiency_metrics['accuracy_per_ms'] = (1.0 / mae) / model_metrics['mean_inference_ms']
                
        except (ZeroDivisionError, ValueError) as e:
            error_msg = f"效率指标计算出错: {str(e)}"
            self.error_messages.append(error_msg)
            self.logger.error(error_msg)
        
        return efficiency_metrics
    
    def comprehensive_profile(self, model: nn.Module, input_shape: Tuple,
                            batch_size: int = 1, device: str = 'cpu',
                            performance_metrics: Optional[Dict] = None) -> Dict:
        """综合性能分析"""
        
        print(f"开始模型轻量化分析...")
        print(f"设备: {device}, 批大小: {batch_size}")
        
        results = {}
        
        # 1. 模型复杂度分析
        print("分析模型复杂度...")
        complexity_metrics = self.profile_model_complexity(model, input_shape, device)
        results['complexity'] = complexity_metrics
        
        # 2. 内存使用分析
        print("分析内存使用...")
        memory_metrics = self.profile_memory_usage(model, input_shape, batch_size, device)
        results['memory'] = memory_metrics
        
        # 3. 推理速度分析
        print("分析推理速度...")
        speed_metrics = self.profile_inference_speed(model, input_shape, batch_size, device=device)
        results['speed'] = speed_metrics
        
        # 4. 稀疏度分析
        print("分析模型稀疏度...")
        sparsity_metrics = self.calculate_sparsity(model)
        results['sparsity'] = sparsity_metrics
        
        # 5. 效率指标计算
        if performance_metrics:
            print("计算效率指标...")
            combined_metrics = {**complexity_metrics, **speed_metrics}
            efficiency_metrics = self.calculate_efficiency_metrics(combined_metrics, performance_metrics)
            if efficiency_metrics:  # 只有当有有效的效率指标时才加入结果
                results['efficiency'] = efficiency_metrics
        else:
            print("跳过效率指标计算（未提供性能数据）")
        
        # 6. 添加诊断信息
        results['diagnostic'] = {
            'error_messages': self.error_messages,
            'warnings': self.warnings,
            'debug_info': self.debug_info,
            'cloud_info': getattr(self, 'cloud_info', {}),
            'gpu_info': getattr(self, 'gpu_info', {})
        }
        
        # 7. 打印总结
        self._print_diagnostic_summary()
        
        # 8. 保存结果
        self.save_profile_results(results)
        
        return results
    
    def save_profile_results(self, results: Dict, filename: Optional[str] = None):
        """保存分析结果"""
        if filename is None:
            timestamp = int(time.time())
            filename = f"lightweight_profile_{timestamp}.json"
        
        filepath = os.path.join(self.save_path, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"轻量化分析报告已保存到: {filepath}")
        
        # 打印摘要
        self.print_summary(results)
    
    def print_summary(self, results: Dict):
        """打印分析摘要"""
        print("\n" + "="*60)
        print("模型轻量化分析摘要")
        print("="*60)
        
        if 'complexity' in results:
            comp = results['complexity']
            print(f"📊 模型复杂度:")
            print(f"  参数量: {comp.get('total_parameters', 0):,}")
            print(f"  模型大小: {comp.get('model_size_mb', 0):.2f} MB")
            print(f"  估算FLOPs: {comp.get('estimated_flops', 0)/1e9:.2f} G")
            print(f"  FLOPs/参数: {comp.get('flops_per_param', 0):.2f}")
        
        if 'memory' in results:
            mem = results['memory']
            print(f"\n💾 内存使用:")
            print(f"  模型内存: {mem.get('model_memory_mb', 0):.2f} MB")
            if 'gpu_peak_mb' in mem:
                print(f"  GPU峰值: {mem.get('gpu_peak_mb', 0):.2f} MB")
                print(f"  内存效率: {mem.get('memory_efficiency', 0):.2%}")
        
        if 'speed' in results:
            speed = results['speed']
            print(f"\n⚡ 推理性能:")
            print(f"  平均延迟: {speed.get('mean_inference_ms', 0):.2f} ms")
            print(f"  吞吐量: {speed.get('throughput_samples_per_sec', 0):.1f} samples/s")
            print(f"  P95延迟: {speed.get('p95_inference_ms', 0):.2f} ms")
        
        if 'sparsity' in results:
            sparse = results['sparsity']
            print(f"\n🔍 模型稀疏度:")
            print(f"  整体稀疏度: {sparse.get('overall_sparsity_percent', 0):.2f}%")
        
        if 'efficiency' in results:
            eff = results['efficiency']
            print(f"\n🎯 效率指标:")
            if 'accuracy_per_param' in eff:
                print(f"  准确性/参数: {eff['accuracy_per_param']:.4f}")
            if 'accuracy_per_gflops' in eff:
                print(f"  准确性/GFLOPs: {eff['accuracy_per_gflops']:.4f}")
            if 'accuracy_per_mb' in eff:
                print(f"  准确性/MB: {eff['accuracy_per_mb']:.4f}")
        
        print("="*60)
    
    def _print_diagnostic_summary(self):
        """打印诊断摘要"""
        if self.error_messages or self.warnings:
            print("\n" + "="*60)
            print("🔍 诊断信息")
            print("="*60)
            
            if self.error_messages:
                print(f"❗ 发现 {len(self.error_messages)} 个错误:")
                for i, error in enumerate(self.error_messages[-3:], 1):  # 只显示最后3个
                    error_first_line = error.split('\n')[0] if '\n' in error else error
                    print(f"  {i}. {error_first_line}")
                if len(self.error_messages) > 3:
                    print(f"  ... 及其他 {len(self.error_messages) - 3} 个错误")
            
            if self.warnings:
                print(f"\n⚠️  发现 {len(self.warnings)} 个警告:")
                for i, warning in enumerate(self.warnings[-3:], 1):  # 只显示最后3个
                    print(f"  {i}. {warning}")
                if len(self.warnings) > 3:
                    print(f"  ... 及其他 {len(self.warnings) - 3} 个警告")
            
            print("\n📄 建议:")
            if any('推理失败' in error for error in self.error_messages):
                print("  - 检查模型输入格式是否正确")
                print("  - 尝试减小batch_size或切换到CPU")
            if any('CUDA' in warning for warning in self.warnings):
                print("  - 检查CUDA环境和驱动")
                print("  - 检查GPU内存是否足够")
            
            print("="*60)
    
    def _detect_cloud_environment(self):
        """检测云服务器环境"""
        import platform
        import subprocess
        
        self.cloud_info = {
            'platform': platform.system(),
            'is_container': False,
            'container_type': None,
            'cloud_provider': 'unknown'
        }
        
        try:
            # 检测容器环境
            if os.path.exists('/.dockerenv'):
                self.cloud_info['is_container'] = True
                self.cloud_info['container_type'] = 'Docker'
            elif os.path.exists('/proc/1/cgroup'):
                with open('/proc/1/cgroup', 'r') as f:
                    content = f.read()
                    if 'docker' in content or 'containerd' in content:
                        self.cloud_info['is_container'] = True
                        self.cloud_info['container_type'] = 'Container'
            
            # 检测云服务提供商
            try:
                # AWS
                result = subprocess.run(['curl', '-s', '--max-time', '2', 
                                       'http://169.254.169.254/latest/meta-data/instance-id'], 
                                      capture_output=True, text=True)
                if result.returncode == 0 and result.stdout:
                    self.cloud_info['cloud_provider'] = 'AWS'
            except:
                pass
            
            try:
                # 阿里云
                result = subprocess.run(['curl', '-s', '--max-time', '2',
                                       'http://100.100.100.200/latest/meta-data/instance-id'],
                                      capture_output=True, text=True)
                if result.returncode == 0 and result.stdout:
                    self.cloud_info['cloud_provider'] = 'Aliyun'
            except:
                pass
                
        except Exception as e:
            self.warnings.append(f"检测云环境失败: {str(e)}")
    
    def _check_gpu_environment(self):
        """检查GPU环境"""
        self.gpu_info = {
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': None,
            'gpu_count': 0,
            'gpu_memory_gb': 0,
            'driver_version': None
        }
        
        try:
            if torch.cuda.is_available():
                self.gpu_info['cuda_version'] = torch.version.cuda
                self.gpu_info['gpu_count'] = torch.cuda.device_count()
                
                if self.gpu_info['gpu_count'] > 0:
                    props = torch.cuda.get_device_properties(0)
                    self.gpu_info['gpu_memory_gb'] = props.total_memory / 1024**3
                    self.gpu_info['gpu_name'] = props.name
                
                # 尝试获取驱动版本
                try:
                    import subprocess
                    result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader,nounits'],
                                          capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        self.gpu_info['driver_version'] = result.stdout.strip()
                except:
                    pass
            
        except Exception as e:
            self.warnings.append(f"GPU环境检查失败: {str(e)}")
    
    def _print_environment_summary(self):
        """打印环境摘要"""
        self.logger.info("=" * 60)
        self.logger.info("🌍 环境检测结果")
        self.logger.info("=" * 60)
        
        # 系统信息
        self.logger.info(f"操作系统: {self.cloud_info['platform']}")
        if self.cloud_info['is_container']:
            self.logger.info(f"容器环境: {self.cloud_info['container_type']}")
        self.logger.info(f"云服务提供商: {self.cloud_info['cloud_provider']}")
        
        # GPU信息
        if self.gpu_info['cuda_available']:
            self.logger.info(f"CUDA可用: 是 (v{self.gpu_info['cuda_version']})")
            self.logger.info(f"GPU数量: {self.gpu_info['gpu_count']}")
            if 'gpu_name' in self.gpu_info:
                self.logger.info(f"GPU型号: {self.gpu_info['gpu_name']}")
                self.logger.info(f"GPU内存: {self.gpu_info['gpu_memory_gb']:.1f}GB")
            if self.gpu_info['driver_version']:
                self.logger.info(f"驱动版本: {self.gpu_info['driver_version']}")
        else:
            self.logger.warning("CUDA不可用")
        
        # 环境建议
        if self.cloud_info['is_container'] and not self.gpu_info['cuda_available']:
            self.logger.warning("在容器中检测不到GPU，请检查--gpus all参数")
        
        self.logger.info("=" * 60)
