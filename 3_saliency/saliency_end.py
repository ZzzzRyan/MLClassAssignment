import os
import glob
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, applications
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import metric  # 你提供的 metric.py

# ==========================================
# 1. 高级配置
# ==========================================
CONFIG = {
    "IMG_SIZE": (224, 224),  # MobileNet 标准输入
    "BATCH_SIZE": 16,
    "EPOCHS": 15,
    "LEARNING_RATE": 1e-4,
    "TRAIN_ROOT": "./3-Saliency-TrainSet",
    "TEST_ROOT": "./3-Saliency-TestSet",
    "OUTPUT_DIR": "./Report_Assets"  # 所有生成的图表都存在这里，方便写报告
}

if not os.path.exists(CONFIG["OUTPUT_DIR"]):
    os.makedirs(CONFIG["OUTPUT_DIR"])

# ==========================================
# 2. 核心：CC-KLD 组合损失函数
# ==========================================
def cc_loss(y_true, y_pred):
    x = K.batch_flatten(y_pred)
    y = K.batch_flatten(y_true)
    mx = K.mean(x, axis=1, keepdims=True)
    my = K.mean(y, axis=1, keepdims=True)
    xm = x - mx
    ym = y - my
    r_num = K.sum(xm * ym, axis=1)
    r_den = K.sqrt(K.sum(K.square(xm), axis=1) * K.sum(K.square(ym), axis=1)) + K.epsilon()
    return -K.mean(r_num / r_den)

def kld_loss(y_true, y_pred):
    y_true = K.batch_flatten(y_true)
    y_pred = K.batch_flatten(y_pred)
    y_true = y_true / (K.sum(y_true, axis=1, keepdims=True) + K.epsilon())
    y_pred = y_pred / (K.sum(y_pred, axis=1, keepdims=True) + K.epsilon())
    return K.mean(K.sum(y_true * K.log((y_true / (y_pred + K.epsilon())) + K.epsilon()), axis=1))

def combined_loss(y_true, y_pred):
    # 组合损失：兼顾像素准确(BCE)、趋势一致(CC)、分布相似(KLD)
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    cc = cc_loss(y_true, y_pred)
    kld = kld_loss(y_true, y_pred)
    return 10.0 * K.mean(bce) + 2.0 * cc + 1.0 * kld

def tf_cc_metric(y_true, y_pred):
    return -cc_loss(y_true, y_pred)

# ==========================================
# 3. 数据管道
# ==========================================
def load_and_preprocess(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, CONFIG["IMG_SIZE"])
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img) # [-1, 1]

    gt_path = tf.strings.regex_replace(img_path, "Stimuli", "FIXATIONMAPS")
    mask = tf.io.read_file(gt_path)
    mask = tf.image.decode_jpeg(mask, channels=1)
    mask = tf.image.resize(mask, CONFIG["IMG_SIZE"])
    mask = mask / 255.0
    return img, mask

def get_dataset(root_dir, batch_size, is_train=True):
    img_paths = glob.glob(os.path.join(root_dir, "Stimuli", "*", "*.jpg"))
    if not img_paths: return None, 0
    ds = tf.data.Dataset.from_tensor_slices(img_paths)
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if is_train: ds = ds.shuffle(buffer_size=500)
    ds = ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds, len(img_paths)

# ==========================================
# 4. MobileNetV2 U-Net 模型
# ==========================================
def build_mobilenet_unet(input_shape):
    inputs = layers.Input(input_shape)
    base_model = applications.MobileNetV2(input_tensor=inputs, include_top=False, weights='imagenet')
    base_model.trainable = True # 微调

    # Encoder Feature Maps
    skips = [
        base_model.get_layer('block_1_expand_relu').output,  # 112x112
        base_model.get_layer('block_3_expand_relu').output,  # 56x56
        base_model.get_layer('block_6_expand_relu').output,  # 28x28
        base_model.get_layer('block_13_expand_relu').output  # 14x14
    ]
    bottleneck = base_model.get_layer('out_relu').output     # 7x7

    # Decoder
    x = bottleneck
    for i, filters in enumerate([256, 128, 64, 64]):
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Concatenate()([x, skips[3-i]])
        x = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x) # 添加BN层加速收敛

    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(x)

    return models.Model(inputs, outputs)

# ==========================================
# 5. 辅助功能：计算额外指标 & 绘图
# ==========================================
def calc_sim(p, q):
    """计算相似度 (Similarity)"""
    p = p / (np.sum(p) + 1e-7)
    q = q / (np.sum(q) + 1e-7)
    return np.sum(np.minimum(p, q))

def save_training_curves(history):
    loss = history.history['loss']
    cc = history.history['tf_cc_metric']
    epochs = range(1, len(loss) + 1)

    plt.figure(figsize=(14, 6))
    
    # Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, 'b-', linewidth=2, label='Total Loss')
    plt.title('Training Loss Convergence', fontsize=14)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # CC 曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, cc, 'r-', linewidth=2, label='Correlation Coefficient (CC)')
    plt.title('Performance Improvement (CC)', fontsize=14)
    plt.xlabel('Epochs')
    plt.ylabel('CC Score')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    save_path = os.path.join(CONFIG["OUTPUT_DIR"], '1_training_curves.png')
    plt.savefig(save_path, dpi=300)
    print(f"[图表] 训练曲线已保存: {save_path}")
    plt.close()

def plot_histogram(cc_scores):
    """绘制 CC 分数分布直方图，用于展示整体性能分布"""
    plt.figure(figsize=(10, 6))
    plt.hist(cc_scores, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('Test Set CC Score Distribution', fontsize=15)
    plt.xlabel('CC Score')
    plt.ylabel('Number of Images')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 标出平均值
    avg = np.mean(cc_scores)
    plt.axvline(avg, color='red', linestyle='dashed', linewidth=2, label=f'Average: {avg:.3f}')
    plt.legend()
    
    save_path = os.path.join(CONFIG["OUTPUT_DIR"], '4_score_distribution.png')
    plt.savefig(save_path, dpi=300)
    print(f"[图表] 成绩分布图已保存: {save_path}")
    plt.close()

def visualize_cases(case_list, title, filename):
    n = len(case_list)
    if n == 0: return
    plt.figure(figsize=(15, 4 * n))
    
    for i, item in enumerate(case_list):
        # 原图
        plt.subplot(n, 3, i*3 + 1)
        plt.imshow(item['img'])
        plt.title(f"Original ({item['category']})", fontsize=12)
        plt.axis('off')

        # 真值
        plt.subplot(n, 3, i*3 + 2)
        plt.imshow(item['gt'], cmap='gray')
        plt.title(f"Ground Truth", fontsize=12)
        plt.axis('off')

        # 预测 (带指标)
        plt.subplot(n, 3, i*3 + 3)
        plt.imshow(item['pred'], cmap='gray')
        plt.title(f"Pred\nCC={item['cc']:.3f} | KLD={item['kld']:.2f} | MAE={item['mae']:.3f}", fontsize=11, color='blue')
        plt.axis('off')
        
    plt.suptitle(title, fontsize=16, y=0.96)
    plt.tight_layout()
    save_path = os.path.join(CONFIG["OUTPUT_DIR"], filename)
    plt.savefig(save_path, dpi=300)
    print(f"[图表] {title} 已保存: {save_path}")
    plt.close()

# ==========================================
# 6. 主程序
# ==========================================
def main():
    # ---------------------------
    # 步骤 1: 准备数据
    # ---------------------------
    print(">>> 1. 构建数据管道...")
    train_ds, train_count = get_dataset(CONFIG["TRAIN_ROOT"], CONFIG["BATCH_SIZE"], is_train=True)
    if train_ds is None: 
        print("错误：找不到训练集！")
        return

    # ---------------------------
    # 步骤 2: 模型构建与保存结构
    # ---------------------------
    print(">>> 2. 构建 MobileNetV2 U-Net 模型...")
    model = build_mobilenet_unet((CONFIG["IMG_SIZE"][0], CONFIG["IMG_SIZE"][1], 3))
    
    # 编译
    model.compile(optimizer=optimizers.Adam(learning_rate=CONFIG["LEARNING_RATE"]),
                  loss=combined_loss,
                  metrics=[tf_cc_metric])
    
    # 保存模型结构摘要 (用于报告第三部分：实验模型结构和参数)
    summary_path = os.path.join(CONFIG["OUTPUT_DIR"], 'model_structure.txt')
    with open(summary_path, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    print(f"[报告素材] 模型结构已保存至: {summary_path}")

    # ---------------------------
    # 步骤 3: 训练
    # ---------------------------
    print(f">>> 3. 开始训练 (共 {CONFIG['EPOCHS']} 轮)...")
    callbacks_list = [
        callbacks.ModelCheckpoint('fast_best_model.h5', monitor='tf_cc_metric', mode='max', save_best_only=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, verbose=1)
    ]
    
    history = model.fit(train_ds, epochs=CONFIG["EPOCHS"], callbacks=callbacks_list, verbose=1)
    save_training_curves(history) # 保存图表1

    # ---------------------------
    # 步骤 4: 评估与详细分析
    # ---------------------------
    print("\n>>> 4. 开始测试集全量评估...")
    try:
        model.load_weights('fast_best_model.h5')
        print("已加载最佳权重。")
    except:
        print("警告：使用当前权重进行评估。")

    test_image_paths = glob.glob(os.path.join(CONFIG["TEST_ROOT"], "Stimuli", "*", "*.jpg"))
    results = [] # 存储详细结果
    
    cc_scores = [] # 用于画直方图

    for i, p in enumerate(test_image_paths):
        # 读取与预处理
        img_raw = cv2.imread(p)
        if img_raw is None: continue
        h, w = img_raw.shape[:2]
        img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        
        img_resized = cv2.resize(img, CONFIG["IMG_SIZE"])
        img_input = tf.keras.applications.mobilenet_v2.preprocess_input(img_resized.astype(np.float32))
        img_input = np.expand_dims(img_input, axis=0)
        
        # 预测
        pred = model.predict(img_input, verbose=0)
        pred_map = np.squeeze(pred)
        pred_map = cv2.resize(pred_map, (w, h))
        
        # GT
        gt_path = p.replace("Stimuli", "FIXATIONMAPS")
        gt_map = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt_map is None: continue
        gt_map = gt_map.astype(np.float32) / 255.0
        
        # 计算多维度指标 (报告加分项)
        try:
            cc = metric.calc_cc_score(gt_map, pred_map)
            kld = metric.KLD(gt_map, pred_map)
            mae = np.mean(np.abs(gt_map - pred_map)) # 新增 MAE
            sim = calc_sim(gt_map, pred_map)         # 新增 SIM
            
            if not np.isnan(cc):
                cc_scores.append(cc)
                category = os.path.basename(os.path.dirname(p)) # 获取图片类别 (如 Action, Art)
                results.append({
                    "img": img, "gt": gt_map, "pred": pred_map,
                    "cc": cc, "kld": kld, "mae": mae, "sim": sim,
                    "category": category
                })
        except: pass
        
        if (i+1) % 50 == 0: print(f"已处理 {i+1}/{len(test_image_paths)}...")

    # ---------------------------
    # 步骤 5: 生成报告所需的统计数据和图表
    # ---------------------------
    if not results: return

    # 排序：按 CC 从高到低
    results.sort(key=lambda x: x['cc'], reverse=True)

    # 5.1 打印总体指标表格
    avg_cc = np.mean([r['cc'] for r in results])
    avg_kld = np.mean([r['kld'] for r in results])
    avg_mae = np.mean([r['mae'] for r in results])
    avg_sim = np.mean([r['sim'] for r in results])

    print("\n" + "="*50)
    print("   最终测试集性能报告 (Test Set Summary)")
    print("="*50)
    print(f"Metrics (Avg) |  Score   |  Target  ")
    print("-" * 40)
    print(f"CC (相关系数) |  {avg_cc:.4f}  |  越高越好 (>0.6)")
    print(f"SIM (相似度)  |  {avg_sim:.4f}  |  越高越好")
    print(f"KLD (KL散度)  |  {avg_kld:.4f}  |  越低越好")
    print(f"MAE (绝对误差)|  {avg_mae:.4f}  |  越低越好")
    print("="*50)

    # 5.2 生成成功案例图 (Top 3)
    visualize_cases(results[:3], "Top 3 Successful Cases", "2_success_cases.png")

    # 5.3 生成失败案例图 (Bottom 3) - 报告必须包含的部分
    # 注意：results是降序，所以最后3个是最差的
    worst_cases = results[-3:]
    visualize_cases(worst_cases, "Worst 3 Failure Cases", "3_failure_cases.png")

    # 5.4 生成分数分布图 (加分项)
    plot_histogram(cc_scores)

    print("\n[完成] 所有实验素材已生成至 ./Report_Assets 文件夹：")
    print("1. 1_training_curves.png (训练 Loss/CC 曲线)")
    print("2. 2_success_cases.png (成功案例)")
    print("3. 3_failure_cases.png (失败案例 - 用于报告分析)")
    print("4. 4_score_distribution.png (CC 分数分布直方图)")
    print("5. model_structure.txt (模型结构文本)")

if __name__ == '__main__':
    main()