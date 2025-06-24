import argparse
import os
from data_preprocessing import DataProcessor
from model_training import train_model, compare_models
from model_lightweight import optimize_model, evaluate_all_models
from inference_deployment import deploy_to_jetson, run_benchmark

def main():
    parser = argparse.ArgumentParser(description='基于甲状腺超声图像的结节分割、分类及轻量化部署系统')
    parser.add_argument('--data_dir', type=str, default='data/thyroid', help='数据集目录')
    parser.add_argument('--model_type', type=str, default='unetpp', choices=['unetpp', 'deeplabv3'], help='模型架构')
    parser.add_argument('--train', action='store_true', help='训练模型')
    parser.add_argument('--compare', action='store_true', help='比较不同模型架构')
    parser.add_argument('--optimize', action='store_true', help='优化模型')
    parser.add_argument('--deploy', action='store_true', help='部署模型到Jetson Nano')
    parser.add_argument('--benchmark', action='store_true', help='运行性能基准测试')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    
    args = parser.parse_args()
    
    # 创建必要的目录
    os.makedirs('models', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    if args.train:
        print(f"开始训练{args.model_type}模型...")
        history, results, model = train_model(
            args.data_dir, 
            model_type=args.model_type,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        print("训练结果:")
        for key, value in results.items():
            print(f"{key}: {value}")
        
        print(f"模型已保存至 models/final_{args.model_type}.h5")
    
    if args.compare:
        print("开始比较不同模型架构...")
        results = compare_models(args.data_dir)
        
        print("模型比较结果:")
        for model_type, result in results.items():
            print(f"\n{model_type}模型:")
            for metric, value in result['results'].items():
                print(f"{metric}: {value}")
    
    if args.optimize:
        print("开始优化模型...")
        # 加载数据
        processor = DataProcessor(args.data_dir)
        images, masks, labels = processor.load_data()
        _, test_data = processor.create_generators(images, masks, labels)
        
        # 加载基础模型
        from tensorflow.keras.models import load_model
        from model_architecture import bce_dice_loss, focal_loss
        
        base_model = load_model(
            f'models/final_{args.model_type}.h5',
            custom_objects={
                'bce_dice_loss': bce_dice_loss,
                'focal_loss': focal_loss
            }
        )
        
        # 优化模型
        pruning_params = {
            'initial_sparsity': 0.2,
            'final_sparsity': 0.5,
            'begin_step': 0,
            'end_step': 1000
        }
        
        optimized_models = optimize_model(
            base_model,
            test_data,
            pruning_params=pruning_params,
            apply_quantization=True,
            apply_kd=True,
            teacher_model=base_model
        )
        
        # 评估所有模型
        evaluation_results = evaluate_all_models(optimized_models, test_data)
        
        # 打印结果
        print("\n模型优化结果:")
        for model_name, result in evaluation_results.items():
            print(f"\n{model_name}:")
            print(f"  参数数量: {result['params']:.4f} M")
            print(f"  推理延迟: {result['latency']:.2f} ms")
            print(f"  分割Dice系数: {result['dice']:.4f}")
            print(f"  分割mIoU: {result['mIoU']:.4f}")
            print(f"  恶性分类AUC: {result['auc_malignancy']:.4f}")
            print(f"  恶性分类准确率: {result['accuracy_malignancy']:.4f}")
            print(f"  TI-RADS分级AUC: {result['auc_tirads']:.4f}")
            print(f"  TI-RADS分级准确率: {result['accuracy_tirads']:.4f}")
    
    if args.deploy:
        print("开始部署模型到Jetson Nano...")
        # 加载数据
        processor = DataProcessor(args.data_dir)
        images, _, _ = processor.load_data()
        
        # 部署并测试
        results = deploy_to_jetson(
            'models/model_int8.tflite', 
            is_tflite=True, 
            test_images=images
        )
        
        if results:
            print(f"部署完成! 平均推理时间: {results['avg_time']:.2f} ms")
    
    if args.benchmark:
        print("开始运行性能基准测试...")
        results = run_benchmark(
            'models/model_int8.tflite',
            is_tflite=True,
            num_runs=100
        )
        
        print("性能基准测试结果:")
        print(f"平均延迟: {results['avg_latency']:.2f} ms")
        print(f"FPS: {results['fps']:.2f}")
        print(f"每次推理能耗: {results['energy_per_inference']:.6f} J")

if __name__ == "__main__":
    main()    