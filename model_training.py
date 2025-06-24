import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC
import numpy as np
import os
from model_architecture import create_unetpp_classifier, create_deeplabv3_classifier, bce_dice_loss, focal_loss
from data_preprocessing import DataProcessor

def train_model(data_dir, model_type='unetpp', epochs=50, batch_size=16):
    # 准备数据
    processor = DataProcessor(data_dir)
    images, masks, labels = processor.load_data()
    train_generator, test_data = processor.create_generators(
        images, masks, labels, batch_size=batch_size
    )
    
    # 解包测试数据
    X_test, y_mask_test, y_malignancy_test, y_tirads_test = test_data
    
    # 创建模型
    if model_type == 'unetpp':
        model = create_unetpp_classifier()
    else:
        model = create_deeplabv3_classifier()
    
    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss={
            'segmentation': bce_dice_loss,
            'malignancy': focal_loss(alpha=0.75),
            'tirads': focal_loss(alpha=0.75)
        },
        metrics={
            'segmentation': [tf.keras.metrics.MeanIoU(num_classes=2), 'accuracy'],
            'malignancy': [AUC(), 'accuracy'],
            'tirads': [AUC(multi_label=True), 'accuracy']
        },
        loss_weights={
            'segmentation': 1.0,
            'malignancy': 1.0,
            'tirads': 1.0
        }
    )
    
    # 设置回调
    os.makedirs('checkpoints', exist_ok=True)
    checkpoint_path = f"checkpoints/best_{model_type}.h5"
    
    callbacks = [
        ModelCheckpoint(
            checkpoint_path, 
            monitor='val_malignancy_accuracy', 
            save_best_only=True, 
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_malignancy_accuracy', 
            patience=10, 
            mode='max', 
            verbose=1,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5, 
            patience=5, 
            verbose=1, 
            min_lr=1e-6
        )
    ]
    
    # 计算训练步数
    steps_per_epoch = len(images) * 0.8 // batch_size
    
    # 训练模型
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=(X_test, {
            'segmentation': y_mask_test,
            'malignancy': y_malignancy_test,
            'tirads': y_tirads_test
        }),
        callbacks=callbacks
    )
    
    # 评估模型
    results = model.evaluate(
        X_test, 
        {
            'segmentation': y_mask_test,
            'malignancy': y_malignancy_test,
            'tirads': y_tirads_test
        },
        batch_size=batch_size,
        return_dict=True
    )
    
    # 保存模型
    model.save(f"models/final_{model_type}.h5")
    
    return history, results, model

def visualize_heatmaps(model, test_images, class_index=1, layer_name='re_lu_29'):
    """生成Grad-CAM热力图"""
    import cv2
    import matplotlib.pyplot as plt
    
    # 创建一个用于获取输出的模型
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(layer_name).output, model.get_layer('malignancy').output]
    )
    
    heatmaps = []
    
    for img in test_images:
        # 扩展维度以匹配模型输入
        img_array = np.expand_dims(img, axis=0)
        
        # 获取梯度
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            class_idx = 1  # 恶性类别
            loss = predictions[:, class_idx]
            
        # 获取梯度
        grads = tape.gradient(loss, conv_outputs)
        
        # 平均梯度
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # 权重激活映射
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        
        # 归一化热力图
        heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
        heatmaps.append(heatmap)
    
    return heatmaps

def compare_models(data_dir):
    """比较不同模型架构的性能"""
    results = {}
    
    # 训练U-Net++模型
    print("训练U-Net++模型...")
    unetpp_history, unetpp_results, unetpp_model = train_model(data_dir, model_type='unetpp')
    results['unetpp'] = {
        'history': unetpp_history.history,
        'results': unetpp_results
    }
    
    # 训练DeepLabV3+模型
    print("训练DeepLabV3+模型...")
    deeplab_history, deeplab_results, deeplab_model = train_model(data_dir, model_type='deeplabv3')
    results['deeplabv3'] = {
        'history': deeplab_history.history,
        'results': deeplab_results
    }
    
    return results    