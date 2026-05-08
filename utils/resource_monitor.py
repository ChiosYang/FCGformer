import time
import json
import csv
import os
import threading
import subprocess
import re
from typing import Dict, List, Optional
import logging

try:
    import psutil
except ImportError:
    psutil = None
    
try:
    import pynvml
except ImportError:
    pynvml = None

import torch


class ResourceMonitor:
    def __init__(self, log_interval: float = 1.0, save_path: str = "./resource_logs"):
        """
        资源监控器
        
        Args:
            log_interval: 监控间隔(秒)
            save_path: 日志保存路径
        """
        self.log_interval = log_interval
        self.save_path = save_path
        self.is_monitoring = False
        self.monitoring_thread = None
        self.error_messages = []
        
        # 初始化数据存储
        self.resource_data = {
            'timestamps': [],
            'gpu_usage': [],
            'gpu_memory': [],
            'gpu_temp': [],
            'gpu_power': [],
            'cpu_usage': [],
            'ram_usage': [],
            'epoch_times': [],
            'batch_times': []
        }
        
        # 创建保存目录
        os.makedirs(save_path, exist_ok=True)
        
        # 初始化GPU监控
        self.gpu_available = self._init_gpu_monitoring()
        
        # 检查psutil可用性
        self.system_monitor_available = psutil is not None
        
        # 监控状态和错误信息
        self.monitoring_status = {
            'pynvml_available': self.gpu_available,
            'nvidia_smi_available': False,
            'psutil_available': self.system_monitor_available,
            'environment_detected': None
        }
        # 检测环境
        self._detect_environment()
        
        # 如果pynvml不可用，尝试nvidia-smi备用方案
        if not self.gpu_available:
            self.monitoring_status['nvidia_smi_available'] = self._test_nvidia_smi()
        
        if not self.system_monitor_available:
            self.error_messages.append("psutil not available. System resource monitoring disabled.")
            logging.warning("psutil not available. System resource monitoring disabled.")
            
        if not self.gpu_available and not self.monitoring_status['nvidia_smi_available']:
            self.error_messages.append("No GPU monitoring method available (neither pynvml nor nvidia-smi).")
            logging.warning("NVIDIA GPU monitoring not available. GPU metrics disabled.")
        elif not self.gpu_available:
            logging.info("pynvml not available, using nvidia-smi fallback for GPU monitoring.")
        
        # 打印初始化摘要
        self._print_initialization_summary()
    
    def _init_gpu_monitoring(self) -> bool:
        """初始化GPU监控"""
        if pynvml is None:
            return False
            
        try:
            pynvml.nvmlInit()
            self.gpu_count = pynvml.nvmlDeviceGetCount()
            self.gpu_handles = []
            
            for i in range(self.gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                self.gpu_handles.append(handle)
                
            return True
        except Exception as e:
            error_msg = f"Failed to initialize GPU monitoring: {e}"
            self.error_messages.append(error_msg)
            logging.warning(error_msg)
            return False
    
    def _detect_environment(self):
        """检测运行环境"""
        try:
            # 检测Docker环境
            if os.path.exists('/.dockerenv'):
                self.monitoring_status['environment_detected'] = 'Docker'
            # 检测容器环境
            elif os.path.exists('/proc/1/cgroup'):
                with open('/proc/1/cgroup', 'r') as f:
                    content = f.read()
                    if 'docker' in content or 'containerd' in content:
                        self.monitoring_status['environment_detected'] = 'Container'
            # 检测Kubernetes
            elif os.path.exists('/var/run/secrets/kubernetes.io'):
                self.monitoring_status['environment_detected'] = 'Kubernetes'
            else:
                self.monitoring_status['environment_detected'] = 'Host'
        except Exception as e:
            logging.warning(f"Failed to detect environment: {e}")
            self.monitoring_status['environment_detected'] = 'Unknown'
    
    def _print_initialization_summary(self):
        """打印初始化摘要"""
        logging.info("=" * 60)
        logging.info("资源监控器初始化摘要")
        logging.info("=" * 60)
        logging.info(f"运行环境: {self.monitoring_status['environment_detected']}")
        logging.info(f"pynvml GPU监控: {'可用' if self.monitoring_status['pynvml_available'] else '不可用'}")
        logging.info(f"nvidia-smi GPU监控: {'可用' if self.monitoring_status['nvidia_smi_available'] else '不可用'}")
        logging.info(f"psutil系统监控: {'可用' if self.monitoring_status['psutil_available'] else '不可用'}")
        
        if self.error_messages:
            logging.warning("初始化警告:")
            for msg in self.error_messages:
                logging.warning(f"  - {msg}")
        
        # 提供修复建议
        if not self.monitoring_status['pynvml_available'] and not self.monitoring_status['nvidia_smi_available']:
            logging.warning("修复建议: 如果在Docker中运行，请使用 --gpus all 参数")
        
        logging.info("=" * 60)
    
    def _test_nvidia_smi(self) -> bool:
        """测试nvidia-smi命令是否可用"""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except Exception:
            return False
    
    def _get_gpu_metrics_nvidia_smi(self) -> Dict:
        """使用nvidia-smi获取GPU指标"""
        try:
            # 获取GPU使用率、显存、温度、功耗
            cmd = ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw', 
                   '--format=csv,noheader,nounits']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                return {'usage': 0, 'memory': 0, 'temp': 0, 'power': 0}
            
            # 解析第一个GPU的数据
            lines = result.stdout.strip().split('\n')
            if not lines or not lines[0]:
                return {'usage': 0, 'memory': 0, 'temp': 0, 'power': 0}
            
            parts = lines[0].split(', ')
            if len(parts) >= 5:
                gpu_usage = float(parts[0]) if parts[0] != '[Not Supported]' else 0
                mem_used = float(parts[1]) if parts[1] != '[Not Supported]' else 0
                mem_total = float(parts[2]) if parts[2] != '[Not Supported]' else 1  # 避免除零
                gpu_memory = (mem_used / mem_total * 100) if mem_total > 0 else 0
                gpu_temp = float(parts[3]) if parts[3] != '[Not Supported]' else 0
                gpu_power = float(parts[4]) if parts[4] != '[Not Supported]' else 0
                
                return {
                    'usage': gpu_usage,
                    'memory': gpu_memory,
                    'temp': gpu_temp,
                    'power': gpu_power
                }
        except Exception as e:
            logging.warning(f"Failed to get GPU metrics via nvidia-smi: {e}")
        
        return {'usage': 0, 'memory': 0, 'temp': 0, 'power': 0}
    
    def _get_gpu_metrics(self) -> Dict:
        """获取GPU指标"""
        if not self.gpu_available:
            # 如果pynvml不可用，尝试nvidia-smi
            if self.monitoring_status['nvidia_smi_available']:
                return self._get_gpu_metrics_nvidia_smi()
            return {'usage': 0, 'memory': 0, 'temp': 0, 'power': 0}
            
        try:
            # 只监控第一个GPU，如果有多个GPU可以扩展
            handle = self.gpu_handles[0]
            
            # GPU使用率
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_usage = util.gpu
            
            # GPU内存
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_memory = mem_info.used / mem_info.total * 100
            
            # GPU温度
            try:
                gpu_temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except:
                gpu_temp = 0
            
            # GPU功耗
            try:
                gpu_power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # 转换为瓦特
            except:
                gpu_power = 0
                
            return {
                'usage': gpu_usage,
                'memory': gpu_memory,
                'temp': gpu_temp,
                'power': gpu_power
            }
        except Exception as e:
            error_msg = f"Failed to get GPU metrics via pynvml: {e}"
            logging.warning(error_msg)
            # 尝试nvidia-smi备用方案
            if self.monitoring_status['nvidia_smi_available']:
                return self._get_gpu_metrics_nvidia_smi()
            return {'usage': 0, 'memory': 0, 'temp': 0, 'power': 0}
    
    def _get_system_metrics(self) -> Dict:
        """获取系统资源指标"""
        if not self.system_monitor_available:
            return {'cpu_usage': 0, 'ram_usage': 0}
            
        try:
            # CPU使用率
            cpu_usage = psutil.cpu_percent(interval=None)
            
            # 内存使用率
            memory = psutil.virtual_memory()
            ram_usage = memory.percent
            
            return {
                'cpu_usage': cpu_usage,
                'ram_usage': ram_usage
            }
        except Exception as e:
            error_msg = f"Failed to get system metrics: {e}"
            logging.warning(error_msg)
            return {'cpu_usage': 0, 'ram_usage': 0}
    
    def _monitoring_loop(self):
        """监控循环"""
        while self.is_monitoring:
            timestamp = time.time()
            
            # 获取GPU指标
            gpu_metrics = self._get_gpu_metrics()
            
            # 获取系统指标
            system_metrics = self._get_system_metrics()
            
            # 存储数据
            self.resource_data['timestamps'].append(timestamp)
            self.resource_data['gpu_usage'].append(gpu_metrics['usage'])
            self.resource_data['gpu_memory'].append(gpu_metrics['memory'])
            self.resource_data['gpu_temp'].append(gpu_metrics['temp'])
            self.resource_data['gpu_power'].append(gpu_metrics['power'])
            self.resource_data['cpu_usage'].append(system_metrics['cpu_usage'])
            self.resource_data['ram_usage'].append(system_metrics['ram_usage'])
            
            time.sleep(self.log_interval)
    
    def start_monitoring(self):
        """开始监控"""
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logging.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """停止监控"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
        logging.info("Resource monitoring stopped")
    
    def record_epoch_time(self, epoch: int, duration: float):
        """记录epoch时间"""
        self.resource_data['epoch_times'].append({
            'epoch': epoch,
            'duration': duration,
            'timestamp': time.time()
        })
    
    def record_batch_time(self, epoch: int, batch: int, duration: float):
        """记录batch时间"""
        self.resource_data['batch_times'].append({
            'epoch': epoch,
            'batch': batch,
            'duration': duration,
            'timestamp': time.time()
        })
    
    def get_model_info(self, model) -> Dict:
        """获取模型信息"""
        try:
            # 计算参数数量
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # 估算模型大小(MB)
            model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
            
            return {
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'model_size_mb': model_size_mb
            }
        except Exception as e:
            logging.warning(f"Failed to get model info: {e}")
            return {}
    
    def save_report(self, model_name: str = "model", additional_info: Dict = None):
        """保存监控报告"""
        try:
            # 计算统计信息
            stats = self._calculate_stats()
            
            # 创建报告
            report = {
                'model_name': model_name,
                'monitoring_duration': len(self.resource_data['timestamps']) * self.log_interval,
                'statistics': stats,
                'monitoring_status': self.monitoring_status,
                'error_messages': self.error_messages,
                'additional_info': additional_info or {}
            }
            
            # 保存JSON报告
            report_path = os.path.join(self.save_path, f"{model_name}_resource_report.json")
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            # 保存CSV详细数据
            self._save_csv_data(model_name)
            
            logging.info(f"Resource report saved to {report_path}")
            return report_path
            
        except Exception as e:
            logging.error(f"Failed to save report: {e}")
            return None
    
    def _calculate_stats(self) -> Dict:
        """计算统计信息"""
        stats = {}
        
        # GPU统计
        if self.resource_data['gpu_usage']:
            # 检查GPU数据是否有效(不是全零)
            gpu_data_valid = (
                max(self.resource_data['gpu_usage']) > 0 or
                max(self.resource_data['gpu_memory']) > 0 or
                max(self.resource_data['gpu_temp']) > 0 or
                max(self.resource_data['gpu_power']) > 0
            )
            
            stats['gpu'] = {
                'avg_usage': sum(self.resource_data['gpu_usage']) / len(self.resource_data['gpu_usage']),
                'max_usage': max(self.resource_data['gpu_usage']),
                'avg_memory': sum(self.resource_data['gpu_memory']) / len(self.resource_data['gpu_memory']),
                'max_memory': max(self.resource_data['gpu_memory']),
                'avg_temp': sum(self.resource_data['gpu_temp']) / len(self.resource_data['gpu_temp']),
                'max_temp': max(self.resource_data['gpu_temp']),
                'avg_power': sum(self.resource_data['gpu_power']) / len(self.resource_data['gpu_power']),
                'max_power': max(self.resource_data['gpu_power']),
                'data_valid': gpu_data_valid,
                'monitoring_method': 'pynvml' if self.gpu_available else ('nvidia-smi' if self.monitoring_status['nvidia_smi_available'] else 'none')
            }
        
        # 系统统计
        if self.resource_data['cpu_usage']:
            # 检查系统数据是否有效
            system_data_valid = (
                max(self.resource_data['cpu_usage']) > 0 or
                max(self.resource_data['ram_usage']) > 0
            )
            
            stats['system'] = {
                'avg_cpu': sum(self.resource_data['cpu_usage']) / len(self.resource_data['cpu_usage']),
                'max_cpu': max(self.resource_data['cpu_usage']),
                'avg_ram': sum(self.resource_data['ram_usage']) / len(self.resource_data['ram_usage']),
                'max_ram': max(self.resource_data['ram_usage']),
                'data_valid': system_data_valid,
                'monitoring_method': 'psutil' if self.system_monitor_available else 'none'
            }
        
        # Epoch时间统计
        if self.resource_data['epoch_times']:
            epoch_durations = [item['duration'] for item in self.resource_data['epoch_times']]
            stats['training'] = {
                'total_epochs': len(epoch_durations),
                'avg_epoch_time': sum(epoch_durations) / len(epoch_durations),
                'total_training_time': sum(epoch_durations)
            }
        
        return stats
    
    def _save_csv_data(self, model_name: str):
        """保存CSV格式的详细数据"""
        try:
            csv_path = os.path.join(self.save_path, f"{model_name}_resource_data.csv")
            
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # 写入表头
                headers = ['timestamp', 'gpu_usage', 'gpu_memory', 'gpu_temp', 'gpu_power', 'cpu_usage', 'ram_usage']
                writer.writerow(headers)
                
                # 写入数据
                for i in range(len(self.resource_data['timestamps'])):
                    row = [
                        self.resource_data['timestamps'][i],
                        self.resource_data['gpu_usage'][i],
                        self.resource_data['gpu_memory'][i],
                        self.resource_data['gpu_temp'][i],
                        self.resource_data['gpu_power'][i],
                        self.resource_data['cpu_usage'][i],
                        self.resource_data['ram_usage'][i]
                    ]
                    writer.writerow(row)
                    
            logging.info(f"Resource CSV data saved to {csv_path}")
            
        except Exception as e:
            logging.warning(f"Failed to save CSV data: {e}")
    
    def print_summary(self):
        """打印监控摘要"""
        stats = self._calculate_stats()
        
        print("\n" + "="*50)
        print("资源使用监控摘要")
        print("="*50)
        
        # 打印监控状态
        print(f"运行环境: {self.monitoring_status['environment_detected']}")
        print(f"GPU监控: {'pynvml' if self.gpu_available else ('nvidia-smi' if self.monitoring_status['nvidia_smi_available'] else '不可用')}")
        print(f"系统监控: {'psutil' if self.system_monitor_available else '不可用'}")
        
        if self.error_messages:
            print(f"错误信息: {'; '.join(self.error_messages)}")
            print("-"*50)
        
        if 'gpu' in stats:
            gpu_stats = stats['gpu']
            valid_suffix = "" if gpu_stats.get('data_valid', True) else " [数据可能不准确]"
            print(f"GPU平均使用率: {gpu_stats['avg_usage']:.1f}% (最大: {gpu_stats['max_usage']:.1f}%){valid_suffix}")
            print(f"GPU平均显存: {gpu_stats['avg_memory']:.1f}% (最大: {gpu_stats['max_memory']:.1f}%){valid_suffix}")
            print(f"GPU平均温度: {gpu_stats['avg_temp']:.1f}°C (最大: {gpu_stats['max_temp']:.1f}°C){valid_suffix}")
            print(f"GPU平均功耗: {gpu_stats['avg_power']:.1f}W (最大: {gpu_stats['max_power']:.1f}W){valid_suffix}")
        
        if 'system' in stats:
            sys_stats = stats['system']
            valid_suffix = "" if sys_stats.get('data_valid', True) else " [数据可能不准确]"
            print(f"CPU平均使用率: {sys_stats['avg_cpu']:.1f}% (最大: {sys_stats['max_cpu']:.1f}%){valid_suffix}")
            print(f"内存平均使用率: {sys_stats['avg_ram']:.1f}% (最大: {sys_stats['max_ram']:.1f}%){valid_suffix}")
        
        if 'training' in stats:
            train_stats = stats['training']
            print(f"总训练时长: {train_stats['total_training_time']:.1f}秒")
            print(f"平均每epoch时长: {train_stats['avg_epoch_time']:.1f}秒")
            print(f"总epoch数: {train_stats['total_epochs']}")
        
        print("="*50)