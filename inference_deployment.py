import tensorflow as tf
import numpy as np
import cv2
import time
import os
from datetime import datetime
from model_architecture import bce_dice_loss, focal_loss

class ThyroidDetector:
    def __init__(self, model_path, is_tflite=False, input_size=(256, 256)):
        self.input_size = input_size
        self.is_tflite = is_tflite
        
        if is_tflite:
            # 加载TFLite模型
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            
            # 获取输入和输出张量信息
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        else:
            # 加载Keras模型
            self.model = tf.keras.models.load_model(
                model_path,
                custom_objects={
                    'bce_dice_loss': bce_dice_loss,
                    'focal_loss': focal_loss,
                    'dice_coef': self._dice_coef
                }
            )
    
    def _dice_coef(self, y_true, y_pred, smooth=1):
        """计算Dice系数"""
        intersection = K.sum(y_true * y_pred, axis=[1,2,3])
        union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
        return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)
    
    def preprocess(self, image):
        """预处理输入图像"""
        # 调整大小
        image = cv2.resize(image, self.input_size)
        
        # 归一化
        image = image / 255.0
        
        # 扩展维度
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def predict(self, image):
        """对输入图像进行预测"""
        # 预处理
        input_data = self.preprocess(image)
        
        if self.is_tflite:
            # 设置输入张量
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data.astype(np.float32))
            
            # 运行推理
            start_time = time.time()
            self.interpreter.invoke()
            inference_time = (time.time() - start_time) * 1000  # 毫秒
            
            # 获取输出张量
            mask = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            malignancy = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
            tirads = self.interpreter.get_tensor(self.output_details[2]['index'])[0]
        else:
            # 运行推理
            start_time = time.time()
            mask, malignancy, tirads = self.model.predict(input_data)
            inference_time = (time.time() - start_time) * 1000  # 毫秒
            
            # 去除批次维度
            mask = mask[0]
            malignancy = malignancy[0]
            tirads = tirads[0]
        
        # 处理输出
        mask = (mask > 0.5).astype(np.uint8) * 255
        malignancy_class = np.argmax(malignancy)
        malignancy_prob = malignancy[malignancy_class]
        tirads_class = np.argmax(tirads) + 1  # TI-RADS等级从1开始
        
        return {
            'mask': mask,
            'malignancy': {
                'class': malignancy_class,
                'probability': float(malignancy_prob)
            },
            'tirads': {
                'class': int(tirads_class),
                'probabilities': [float(p) for p in tirads]
            },
            'inference_time': inference_time
        }
    
    def visualize(self, image, prediction, output_path=None):
        """可视化预测结果"""
        # 调整掩码大小以匹配原始图像
        original_height, original_width = image.shape[:2]
        mask = cv2.resize(prediction['mask'], (original_width, original_height))
        
        # 创建彩色掩码
        mask_color = np.zeros_like(image)
        mask_color[mask > 127] = [0, 255, 0]  # 绿色
        
        # 叠加掩码
        overlay = cv2.addWeighted(image, 0.7, mask_color, 0.3, 0)
        
        # 添加文本信息
        malignancy_text = f"Malignancy: {'Malignant' if prediction['malignancy']['class'] == 1 else 'Benign'} ({prediction['malignancy']['probability']:.2f})"
        tirads_text = f"TI-RADS: {prediction['tirads']['class']}"
        time_text = f"Inference Time: {prediction['inference_time']:.2f} ms"
        
        cv2.putText(overlay, malignancy_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(overlay, tirads_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(overlay, time_text, (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 保存或返回结果
        if output_path:
            cv2.imwrite(output_path, overlay)
        
        return overlay

def deploy_to_jetson(model_path, is_tflite=True, test_images=None):
    """模拟部署到Jetson Nano"""
    # 创建检测器
    detector = ThyroidDetector(model_path, is_tflite)
    
    # 测试推理性能
    if test_images is not None and len(test_images) > 0:
        inference_times = []
        for image in test_images[:100]:  # 只测试100张图像以节省时间
            result = detector.predict(image)
            inference_times.append(result['inference_time'])
        
        # 计算平均推理时间
        avg_time = sum(inference_times) / len(inference_times)
        max_time = max(inference_times)
        min_time = min(inference_times)
        
        print(f"Jetson Nano 推理性能:")
        print(f"平均推理时间: {avg_time:.2f} ms")
        print(f"最小推理时间: {min_time:.2f} ms")
        print(f"最大推理时间: {max_time:.2f} ms")
        
        return {
            'avg_time': avg_time,
            'min_time': min_time,
            'max_time': max_time
        }
    else:
        print("未提供测试图像，无法评估推理性能")
        return None

def create_trt_engine(model_path, output_path, input_shape=(1, 256, 256, 3)):
    """创建TensorRT引擎"""
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    
    # 创建TensorRT记录器
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    # 创建构建器和网络
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # 从ONNX文件加载模型
    with open(model_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("ONNX解析错误:")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    # 配置构建器
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB
    
    # 构建引擎
    with builder.build_engine(network, config) as engine:
        # 保存引擎
        with open(output_path, 'wb') as f:
            f.write(engine.serialize())
    
    print(f"TensorRT引擎已保存至: {output_path}")
    return output_path

def run_benchmark(model_path, is_tflite=True, num_runs=100):
    """运行基准测试"""
    # 创建一个随机图像用于测试
    test_image = np.random.randint(0, 256, size=(512, 512, 3), dtype=np.uint8)
    
    # 创建检测器
    detector = ThyroidDetector(model_path, is_tflite)
    
    # 预热运行
    for _ in range(10):
        detector.predict(test_image)
    
    # 测量性能
    start_time = time.time()
    for _ in range(num_runs):
        detector.predict(test_image)
    end_time = time.time()
    
    # 计算性能指标
    total_time = end_time - start_time
    avg_time = total_time / num_runs * 1000  # 毫秒
    fps = num_runs / total_time
    
    # 估计功耗 (Jetson Nano 典型值)
    power = 5  # 瓦
    energy_per_inference = (power * avg_time / 1000) / 1000  # 焦耳
    
    return {
        'avg_latency': avg_time,
        'fps': fps,
        'energy_per_inference': energy_per_inference
    }    