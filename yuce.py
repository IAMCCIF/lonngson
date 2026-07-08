import sys
import os

# ============================================================
# 1. 路径配置（请确保这些路径在你的龙芯机器上是正确的）
# ============================================================
# 指向你刚才修改了 exporter.py 的那个 ultralytics 源码文件夹
source_code_path = '/home/sancog/Desktop/longxin/ultralytics'
# 指向你的模型文件
model_path = '/home/sancog/Desktop/longxin/best.pt'
# 指向你要预测的图片
image_path = '/home/sancog/Desktop/longxin/1372422923_jpg.rf.e09c80241064db7ebea034036a3d24a8.jpg'

# ============================================================
# 2. 注入源码路径（必须在 import ultralytics 之前）
# ============================================================
if source_code_path not in sys.path:
    sys.path.insert(0, source_code_path)

# ============================================================
# 3. 运行推理
# ============================================================
try:
    print("正在检查环境并加载模型...")
    from ultralytics import YOLO
    
    # 强制不使用 pandas，如果在其他地方还有残留调用，这里可以做个保底
    os.environ['ULTRALYTICS_PANDAS_DISABLED'] = 'True'

    # 加载模型
    model = YOLO(model_path)

    print(f"模型加载成功！开始预测图片: {os.path.basename(image_path)}")
    
    # 执行预测
    # device='cpu' 确保在龙芯上不调用不存在的 CUDA
    results = model.predict(
        source=image_path,
        device='cpu',
        save=True,
        conf=0.25  # 置信度阈值
    )

    print("\n" + "="*50)
    print("✅ 预测成功！")
    # 打印结果保存路径
    for result in results:
        if hasattr(result, 'save_dir'):
            print(f"结果图片已保存至: {result.save_dir}")
    print("="*50)

except ImportError as e:
    print(f"\n❌ 导入失败: {e}")
    print("提示: 请检查 '/home/sancog/Desktop/longxin/ultralytics' 目录下是否存在 __init__.py")
except Exception as e:
    print(f"\n❌ 运行过程中出错: \n{e}")
    # 打印详细错误堆栈，方便我们排查
    import traceback
    traceback.print_exc()