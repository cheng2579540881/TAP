import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras.models import load_model
from model_architecture import bce_dice_loss, focal_loss
import numpy as np
import time
import os
from datetime import datetime

def apply_pruning(model, pruning_params, validation_data):
    """应用结构化剪枝"""
    pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=pruning_params['initial_sparsity'],
        final_sparsity=pruning_params['final_sparsity'],
        begin_step=pruning_params['begin_step'],
        end_step=pruning_params['end_step']
    )
    
    model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(
        model, pruning_schedule=pruning_schedule
    )
    
    # 编译模型
    model_for_pruning.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss={
            'segmentation': bce_dice_loss,
            'malignancy': focal_loss(alpha=0.75),
            'tirads': focal_loss(alpha=0.75)
        },
        metrics={
            'segmentation': [tf.keras.metrics.MeanIoU(num_classes=2), 'accuracy'],
            'malignancy': [tf.keras.metrics.AUC(), 'accuracy'],
            'tirads': [tf.keras.metrics.AUC(multi_label=True), 'accuracy']
        }
    )
    
    # 回调函数
    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
        tf.keras.callbacks.ModelCheckpoint(
            'checkpoints/pruned_model.h5',
            monitor='val_malignancy_accuracy',
            save_best_only=True,
            mode='max'
        )
    ]
    
    # 微调模型
    model_for_pruning.fit(
        validation_data[0],
        {
            'segmentation': validation_data[1],
            'malignancy': validation_data[2],
            'tirads': validation_data[3]
        },
        epochs=5,
        batch_size=16,
        callbacks=callbacks,
        validation_split=0.2
    )
    
    # 移除剪枝包装器
    model_pruned = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
    
    return model_pruned

def apply_quantization_aware_training(model, validation_data):
    """应用量化感知训练"""
    quantize_model = tfmot.quantization.keras.quantize_model
    
    # 量化模型
    q_aware_model = quantize_model(model)
    
    # 编译模型
    q_aware_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss={
            'segmentation': bce_dice_loss,
            'malignancy': focal_loss(alpha=0.75),
            'tirads': focal_loss(alpha=0.75)
        },
        metrics={
            'segmentation': [tf.keras.metrics.MeanIoU(num_classes=2), 'accuracy'],
            'malignancy': [tf.keras.metrics.AUC(), 'accuracy'],
            'tirads': [tf.keras.metrics.AUC(multi_label=True), 'accuracy']
        }
    )
    
    # 训练模型
    q_aware_model.fit(
        validation_data[0],
        {
            'segmentation': validation_data[1],
            'malignancy': validation_data[2],
            'tirads': validation_data[3]
        },
        epochs=3,
        batch_size=16,
        validation_split=0.2
    )
    
    return q_aware_model

def knowledge_distillation(teacher_model, student_model, validation_data, epochs=10):
    """实现知识蒸馏"""
    class Distiller(tf.keras.Model):
        def __init__(self, student, teacher):
            super(Distiller, self).__init__()
            self.teacher = teacher
            self.student = student
            
        def compile(
            self,
            optimizer,
            metrics,
            student_loss_fn,
            distillation_loss_fn,
            alpha=0.1,
            temperature=3,
        ):
            """配置蒸馏器。

            Args:
                optimizer: Keras优化器用于学生权重
                metrics: 评估模型使用的指标
                student_loss_fn: 学生输出与真实标签的损失函数
                distillation_loss_fn: 学生输出与教师输出的损失函数
                alpha: 权重因子，用于平衡两种损失
                temperature: 蒸馏温度
            """
            super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
            self.student_loss_fn = student_loss_fn
            self.distillation_loss_fn = distillation_loss_fn
            self.alpha = alpha
            self.temperature = temperature
            
        def train_step(self, data):
            # 解包数据
            x, y = data
            
            # 前向传播教师
            teacher_predictions = self.teacher(x, training=False)
            
            with tf.GradientTape() as tape:
                # 前向传播学生
                student_predictions = self.student(x, training=True)
                
                # 计算损失
                student_loss = self.student_loss_fn(y, student_predictions)
                
                # 计算蒸馏损失
                distillation_loss = self.distillation_loss_fn(
                    tf.nn.softmax(teacher_predictions[1] / self.temperature, axis=1),
                    tf.nn.softmax(student_predictions[1] / self.temperature, axis=1)
                )
                
                # 计算总损失
                loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
                
            # 计算梯度
            trainable_vars = self.student.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)
            
            # 更新权重
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))
            
            # 更新指标
            self.compiled_metrics.update_state(y, student_predictions)
            
            # 返回指标和损失
            results = {m.name: m.result() for m in self.metrics}
            results.update(
                {"student_loss": student_loss, "distillation_loss": distillation_loss}
            )
            return results
            
        def test_step(self, data):
            # 解包数据
            x, y = data
            
            # 计算预测
            y_preds = self.student(x, training=False)
            
            # 更新指标
            self.compiled_metrics.update_state(y, y_preds)
            
            # 返回指标
            return {m.name: m.result() for m in self.metrics}
    
    # 创建并编译蒸馏器
    distiller = Distiller(student=student_model, teacher=teacher_model)
    distiller.compile(
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy'],
        student_loss_fn=tf.keras.losses.CategoricalCrossentropy(),
        distillation_loss_fn=tf.keras.losses.KLDivergence(),
        alpha=0.3,
        temperature=5,
    )
    
    # 解包验证数据
    X_val, y_mask_val, y_malignancy_val, y_tirads_val = validation_data
    
    # 训练学生模型
    distiller.fit(
        X_val, 
        y_malignancy_val,
        epochs=epochs,
        batch_size=16,
        validation_split=0.2
    )
    
    return student_model

def convert_to_tflite(model, quantization='float16', representative_data=None):
    """将模型转换为TFLite格式"""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantization == 'float16':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quantization == 'int8':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        def representative_dataset_gen():
            for i in range(100):
                yield [representative_data[i:i+1]]
                
        converter.representative_dataset = representative_dataset_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    
    # 转换模型
    tflite_model = converter.convert()
    
    # 保存TFLite模型
    tflite_path = f"models/model_{quantization}.tflite"
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    return tflite_path

def evaluate_latency(model, test_images, num_runs=100, is_tflite=False):
    """评估模型延迟"""
    if is_tflite:
        # 加载TFLite模型
        interpreter = tf.lite.Interpreter(model_path=model)
        interpreter.allocate_tensors()
        
        # 获取输入和输出张量
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # 预热运行
        for i in range(10):
            input_data = test_images[i:i+1].astype(np.float32)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
        
        # 测量延迟
        start_time = time.time()
        for i in range(num_runs):
            input_data = test_images[i % len(test_images):(i % len(test_images))+1].astype(np.float32)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
        end_time = time.time()
        
        # 计算平均延迟
        avg_latency = (end_time - start_time) / num_runs * 1000  # 毫秒
    else:
        # 预热运行
        for i in range(10):
            model.predict(test_images[i:i+1])
        
        # 测量延迟
        start_time = time.time()
        for i in range(num_runs):
            model.predict(test_images[i % len(test_images):(i % len(test_images))+1])
        end_time = time.time()
        
        # 计算平均延迟
        avg_latency = (end_time - start_time) / num_runs * 1000  # 毫秒
    
    return avg_latency

def count_parameters(model, is_tflite=False):
    """计算模型参数数量"""
    if is_tflite:
        # 对于TFLite模型，我们可以通过加载并统计张量大小来估算
        interpreter = tf.lite.Interpreter(model_path=model)
        interpreter.allocate_tensors()
        
        tensor_details = interpreter.get_tensor_details()
        total_params = 0
        
        for tensor in tensor_details:
            tensor_shape = tensor['shape']
            params = np.prod(tensor_shape)
            total_params += params
        
        # 将参数数量转换为M
        total_params_m = total_params / 1e6
    else:
        # 对于Keras模型，直接使用model.count_params()
        total_params = model.count_params()
        total_params_m = total_params / 1e6
    
    return total_params_m

def optimize_model(model, validation_data, pruning_params=None, apply_quantization=True, apply_kd=False, teacher_model=None):
    """应用多种优化技术优化模型"""
    results = {}
    
    # 原始模型
    original_model = model
    
    # 应用剪枝
    if pruning_params:
        pruned_model = apply_pruning(original_model, pruning_params, validation_data)
        results['pruned'] = {
            'model': pruned_model,
            'params': count_parameters(pruned_model)
        }
    else:
        pruned_model = original_model
        results['pruned'] = {
            'model': pruned_model,
            'params': count_parameters(pruned_model)
        }
    
    # 应用知识蒸馏
    if apply_kd and teacher_model:
        # 创建学生模型
        student_model = create_unetpp_classifier()
        distilled_model = knowledge_distillation(teacher_model, student_model, validation_data)
        results['distilled'] = {
            'model': distilled_model,
            'params': count_parameters(distilled_model)
        }
    else:
        distilled_model = pruned_model
        results['distilled'] = {
            'model': distilled_model,
            'params': count_parameters(distilled_model)
        }
    
    # 应用量化
    if apply_quantization:
        qat_model = apply_quantization_aware_training(distilled_model, validation_data)
        
        # 转换为TFLite
        representative_data = validation_data[0][:100]
        tflite_float16_path = convert_to_tflite(qat_model, 'float16', representative_data)
        tflite_int8_path = convert_to_tflite(qat_model, 'int8', representative_data)
        
        results['tflite_float16'] = {
            'model_path': tflite_float16_path,
            'params': count_parameters(tflite_float16_path, is_tflite=True)
        }
        
        results['tflite_int8'] = {
            'model_path': tflite_int8_path,
            'params': count_parameters(tflite_int8_path, is_tflite=True)
        }
    else:
        results['final_model'] = {
            'model': distilled_model,
            'params': count_parameters(distilled_model)
        }
    
    return results

def evaluate_all_models(models, test_data):
    """评估所有模型的性能"""
    X_test, y_mask_test, y_malignancy_test, y_tirads_test = test_data
    results = {}
    
    for name, model_info in models.items():
        is_tflite = 'model_path' in model_info
        model = model_info['model_path'] if is_tflite else model_info['model']
        
        # 评估性能
        if is_tflite:
            # 对于TFLite模型，需要使用解释器评估
            interpreter = tf.lite.Interpreter(model_path=model)
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # 预测
            y_pred_mask = []
            y_pred_malignancy = []
            y_pred_tirads = []
            
            for i in range(len(X_test)):
                input_data = X_test[i:i+1].astype(np.float32)
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                
                # 获取输出
                output_mask = interpreter.get_tensor(output_details[0]['index'])
                output_malignancy = interpreter.get_tensor(output_details[1]['index'])
                output_tirads = interpreter.get_tensor(output_details[2]['index'])
                
                y_pred_mask.append(output_mask[0])
                y_pred_malignancy.append(output_malignancy[0])
                y_pred_tirads.append(output_tirads[0])
            
            y_pred_mask = np.array(y_pred_mask)
            y_pred_malignancy = np.array(y_pred_malignancy)
            y_pred_tirads = np.array(y_pred_tirads)
            
            # 计算指标
            dice = dice_coef(y_mask_test, y_pred_mask).numpy()
            mIoU = tf.keras.metrics.MeanIoU(num_classes=2)
            mIoU.update_state(y_mask_test > 0.5, y_pred_mask > 0.5)
            mIoU = mIoU.result().numpy()
            
            auc_malignancy = tf.keras.metrics.AUC()
            auc_malignancy.update_state(y_malignancy_test, y_pred_malignancy)
            auc_malignancy = auc_malignancy.result().numpy()
            
            accuracy_malignancy = np.mean(np.argmax(y_malignancy_test, axis=1) == np.argmax(y_pred_malignancy, axis=1))
            
            auc_tirads = tf.keras.metrics.AUC(multi_label=True)
            auc_tirads.update_state(y_tirads_test, y_pred_tirads)
            auc_tirads = auc_tirads.result().numpy()
            
            accuracy_tirads = np.mean(np.argmax(y_tirads_test, axis=1) == np.argmax(y_pred_tirads, axis=1))
            
            results[name] = {
                'dice': dice,
                'mIoU': mIoU,
                'auc_malignancy': auc_malignancy,
                'accuracy_malignancy': accuracy_malignancy,
                'auc_tirads': auc_tirads,
                'accuracy_tirads': accuracy_tirads,
                'params': model_info['params'],
                'latency': evaluate_latency(model, X_test, is_tflite=True)
            }
        else:
            # 对于Keras模型，直接评估
            eval_results = model.evaluate(
                X_test, 
                {
                    'segmentation': y_mask_test,
                    'malignancy': y_malignancy_test,
                    'tirads': y_tirads_test
                },
                batch_size=16,
                return_dict=True
            )
            
            results[name] = {
                'dice': 1 - eval_results['segmentation_loss'],
                'mIoU': eval_results['segmentation_mean_io_u'],
                'auc_malignancy': eval_results['malignancy_auc'],
                'accuracy_malignancy': eval_results['malignancy_accuracy'],
                'auc_tirads': eval_results['tirads_auc'],
                'accuracy_tirads': eval_results['tirads_accuracy'],
                'params': model_info['params'],
                'latency': evaluate_latency(model, X_test)
            }
    
    return results    