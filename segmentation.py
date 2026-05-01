"""
#################################
 Fire Segmentation on Fire Class to extract fire pixels from each frame based on the Ground Truth data (masks)
 Train, Validation, Test Data: Items (9) and (10) on https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs
 Keras version: 2.4.0
 Tensorflow Version: 2.3.0
 GPU: Nvidia RTX 2080 Ti
 OS: Ubuntu 18.04
################################
"""

#########################################################
# import libraries
import os
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    concatenate,
    MaxPooling2D,
    Dropout,
    Lambda,
    Conv2D,
    Conv2DTranspose,
    BatchNormalization,
    Activation,
    Add,
    Multiply,
    GlobalAveragePooling2D,
    Reshape,
    Dense,
    Concatenate,
)

import tensorflow.keras.backend as K

# 假设你有 config.py，如果没有，请手动替换下面的配置获取部分
try:
    from config import config_segmentation, segmentation_new_size
except ImportError:
    # 如果没有配置文件，这里提供默认值防止报错，请根据实际情况调整
    config_segmentation = {
        "batch_size": 8,
        "Epochs": 50,
        "CHANNELS": 3,
        "num_class": 1
    }
    segmentation_new_size = {"width": 256, "height": 256}

#########################################################
# Global parameters and definitions

METRICS = [
    tf.keras.metrics.AUC(name="auc"),
    tf.keras.metrics.Recall(name="recall"),
    tf.keras.metrics.Precision(name="precision"),
    tf.keras.metrics.myiou(name='iou'),  
]


def plot_training_curves(history, save_dir="Output/Model_figure"):
    os.makedirs(save_dir, exist_ok=True)

    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])

    if len(loss) > 0:
        epochs = range(1, len(loss) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, loss, 'b-', label='Training Loss')
        plt.plot(epochs, val_loss, 'r--', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        save_path_loss = os.path.join(save_dir, 'loss_curve.png')
        plt.savefig(save_path_loss, dpi=150)
        plt.close()
        print(f"[INFO] Loss curve saved to: {save_path_loss}")
    iou = history.history.get('iou', [])
    val_iou = history.history.get('val_iou', [])

    if len(iou) > 0:
        epochs = range(1, len(iou) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, iou, 'b-', label='Training IoU')
        plt.plot(epochs, val_iou, 'r--', label='Validation IoU')
        plt.title('Training and Validation IoU')
        plt.xlabel('Epochs')
        plt.ylabel('IoU')
        plt.legend()
        plt.grid(True)

        save_path_iou = os.path.join(save_dir, 'iou_curve.png')
        plt.savefig(save_path_iou, dpi=150)
        plt.close()
        print(f"[INFO] IoU curve saved to: {save_path_iou}")


def save_error_analysis_masks(model_predict, x_val, y_val, save_dir,
                              threshold=0.5, mask_names=None):
    os.makedirs(save_dir, exist_ok=True)
    num_samples = len(x_val)
    print(f"[INFO] 开始保存误差分析掩码 (Red=FP, Blue=FN)，总数: {num_samples}")

    for i in range(num_samples):
        img = x_val[i]
        gt = y_val[i]  # shape (H, W, 1)

        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)
        img_input = np.expand_dims(img, axis=0)

        pred = model_predict.predict(img_input, batch_size=1, verbose=0)
        if isinstance(pred, (list, tuple)):
            pred_seg = pred[0]
        else:
            pred_seg = pred

        pred_prob = pred_seg[0]
        if pred_prob.ndim == 3:
            pred_prob = pred_prob[..., 0]

        pred_bin = (pred_prob > threshold).astype(np.uint8)

        gt_mask = gt
        if gt_mask.ndim == 3:
            gt_mask = gt_mask[..., 0]
        gt_bin = (gt_mask > 0).astype(np.uint8) 

        H, W = pred_bin.shape
        error_map = np.zeros((H, W, 3), dtype=np.uint8)

        
        # TP (True Positive): GT=1 & Pred=1 -> white
        tp_mask = (gt_bin == 1) & (pred_bin == 1)
        error_map[tp_mask] = [255, 255, 255]

        # FP (False Positive): GT=0 & Pred=1 -> red (BGR: 0, 0, 255)
        fp_mask = (gt_bin == 0) & (pred_bin == 1)
        error_map[fp_mask] = [0, 0, 255]

        # FN (False Negative): GT=1 & Pred=0 -> blue (BGR: 255, 0, 0)
        fn_mask = (gt_bin == 1) & (pred_bin == 0)
        error_map[fn_mask] = [255, 0, 0]

        # 5. save
        if mask_names is not None:
            base_name = mask_names[i]
            save_path = os.path.join(save_dir, f"{base_name}.png")
        else:
            save_path = os.path.join(save_dir, f"mask_error_{i:04d}.png")

        cv2.imwrite(save_path, error_map)

        if (i + 1) % 10 == 0 or (i + 1) == num_samples:
            print(f"[INFO] 已保存 {i + 1}/{num_samples} : {save_path}")

    print("[INFO] 误差分析图保存完毕。")


#########################################################
# Dice 
#########################################################
def dice_coef(y_true, y_pred, smooth=1.0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    y_true_f = tf.cast(y_true_f, tf.float32)
    y_pred_f = tf.cast(y_pred_f, tf.float32)

    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
            K.sum(y_true_f) + K.sum(y_pred_f) + smooth
    )


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


#########################################################
#  BCE + Dice
#########################################################
def edge_weighted_bce(edge_weight=5.0, kernel_size=5):
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred)
        epsilon = K.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        y_true_smooth = tf.nn.avg_pool2d(
            y_true, ksize=kernel_size, strides=1, padding="SAME"
        )
        edge_map = tf.abs(y_true - y_true_smooth)

        weights = 1.0 + (edge_map * 2.0) * (edge_weight - 1.0)

        bce_loss = -(
                y_true * tf.math.log(y_pred)
                + (1.0 - y_true) * tf.math.log(1.0 - y_pred)
        )

        weighted_loss = bce_loss * weights
        return K.mean(weighted_loss)

    return loss_fn


def combo_edge_dice_loss(y_true, y_pred):
    loss_edge = edge_weighted_bce(edge_weight=5.0)(y_true, y_pred)
    loss_dice = dice_loss(y_true, y_pred)
    return loss_edge + loss_dice


#########################################################
# Image | GT | Prediction
#########################################################
def save_segmentation_results(
        x_val, y_val, y_pred, save_dir="Output/SegmentationResults", num_samples=6
):
    os.makedirs(save_dir, exist_ok=True)

    num_samples = min(num_samples, len(x_val))
    indices = np.arange(num_samples)

    for i, idx in enumerate(indices):
        img = x_val[idx]
        gt = y_val[idx].squeeze()
        pred = y_pred[idx].squeeze()

        if gt.max() > 1:
            gt = gt / 255.0
        if pred.max() > 1:
            pred = pred / 255.0

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(img.astype(np.uint8))
        axes[0].set_title("Image")
        axes[0].axis("off")

        axes[1].imshow(gt, cmap="gray")
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")

        axes[2].imshow(pred, cmap="gray")
        axes[2].set_title("Prediction")
        axes[2].axis("off")

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"val_sample_{i:03d}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


#########################################################
#
#########################################################
def load_dataset(img_dir, mask_dir, img_size, img_channels, desc=""):
    H, W = img_size[1], img_size[0]

    img_paths = sorted(
        [
            os.path.join(img_dir, fname)
            for fname in os.listdir(img_dir)
            if fname.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
    )
    mask_paths = sorted(
        [
            os.path.join(mask_dir, fname)
            for fname in os.listdir(mask_dir)
            if fname.lower().endswith(".png") and not fname.startswith(".")
        ]
    )

    assert len(img_paths) == len(mask_paths), (
        f"[ERROR] {desc} 图像数量 {len(img_paths)} 和 mask 数量 {len(mask_paths)} 不一致，"
        f"请检查目录 {img_dir} / {mask_dir}"
    )

    x = np.zeros((len(img_paths), H, W, img_channels), dtype=np.uint8)
    y = np.zeros((len(mask_paths), H, W, 1), dtype=np.uint8)
    mask_names = []

    print(f"\n[INFO] Loading {desc} images from {img_dir}, num = {len(img_paths)}")
    for n, file_ in enumerate(tqdm(img_paths)):
        img = tf.keras.preprocessing.image.load_img(file_, target_size=img_size)
        x[n] = img

    print(f"\n[INFO] Loading {desc} masks from {mask_dir}, num = {len(mask_paths)}")
    for n, file_ in enumerate(tqdm(mask_paths)):
        img = tf.keras.preprocessing.image.load_img(
            file_, target_size=img_size, color_mode="grayscale"
        )
        mask = tf.keras.preprocessing.image.img_to_array(img)
        mask = (mask > 127).astype(np.uint8)
        y[n] = mask

        base = os.path.splitext(os.path.basename(file_))[0]
        mask_names.append(base)

    return x, y, mask_names


#########################################################
# Test
#########################################################
class TestEvalCallback(tf.keras.callbacks.Callback):
    def __init__(self, x_test, y_test,
                 save_path="FireSegmentation.h5",
                 batch_size=1):
        super().__init__()
        self.x_test = x_test
        self.y_test = y_test
        self.save_path = save_path
        self.batch_size = batch_size
        self.best_iou = -1.0

    def on_epoch_end(self, epoch, logs=None):
        if self.x_test is None or len(self.x_test) == 0:
            return

        print(f"\n[INFO] Epoch {epoch + 1}: 正在对 TEST 集评估 ...")

        results = self.model.evaluate(
            self.x_test,
            self.y_test,
            batch_size=self.batch_size,
            verbose=0
        )
        metrics_names = self.model.metrics_names

        msg_parts = []
        for name, value in zip(metrics_names, results):
            msg_parts.append(f"{name} = {value:.4f}")
        msg = " | ".join(msg_parts)
        print(f"[TEST] Epoch {epoch + 1}: {msg}")

        if logs is not None:
            for name, value in zip(metrics_names, results):
                logs[f"test_{name}"] = value

        if "iou" in metrics_names:
            iou_idx = metrics_names.index("iou")
            test_iou = results[iou_idx]
        else:
            test_iou = 0.0

        if test_iou > self.best_iou:
            self.best_iou = test_iou
            self.model.save(self.save_path)
            print(
                f"[TEST]  best TEST IoU = {test_iou:.4f}，: {self.save_path}"
            )


#########################################################
# 
#########################################################
def conv_block(
        filters,
        kernel_size=(3, 3),
        activation="elu",
        padding="same",
        kernel_initializer="he_normal",
):
    def layer(x):
        x = Conv2D(
            filters, kernel_size, padding=padding, kernel_initializer=kernel_initializer
        )(x)
        x = BatchNormalization()(x)
        if activation:
            x = Activation(activation)(x)
        return x

    return layer


def preprocess_rgb(x):
    R = x[..., 0:1]
    B = x[..., 2:3]
    diff = R - B

    diff_min = tf.reduce_min(diff, axis=[1, 2], keepdims=True)
    diff_max = tf.reduce_max(diff, axis=[1, 2], keepdims=True)
    diff_norm = (diff - diff_min) / (diff_max - diff_min + 1e-6) * 255.0

    binary = tf.where(diff_norm > 125.0, 1.0, 0.0)

    binary_4D = binary
    sobel = tf.image.sobel_edges(binary_4D)
    gx = sobel[..., 0]
    gy = sobel[..., 1]
    magnitude = tf.sqrt(gx * gx + gy * gy)

    mag_min = tf.reduce_min(magnitude, axis=[1, 2, 3], keepdims=True)
    mag_max = tf.reduce_max(magnitude, axis=[1, 2, 3], keepdims=True)
    mag_norm = (magnitude - mag_min) / (mag_max - mag_min + 1e-6)

    edges = tf.where(mag_norm > 0.2, 1.0, 0.0)
    return edges


def HVD_Block(low_feat, high_feat, filters):
    low_feat = Conv2D(filters, (1, 1), padding="same")(low_feat)
    high_feat = Conv2D(filters, (1, 1), padding="same")(high_feat)

    merged = Add()([low_feat, high_feat])

    row_att = Conv2D(filters, (1, 7), padding="same", activation="sigmoid")(merged)
    feat_row = Multiply()([high_feat, row_att])
    feat_row = Add()([feat_row, low_feat])

    col_att = Conv2D(filters, (7, 1), padding="same", activation="sigmoid")(merged)
    feat_col = Multiply()([high_feat, col_att])
    feat_col = Add()([feat_col, low_feat])

    out = Add()([feat_row, feat_col])
    return out


def GatedSpatialConv2d(input_features, gating_features, filters):
    target_shape = tf.shape(input_features)[1:3]
    gating_resized = tf.image.resize(gating_features, target_shape)

    concat_feat = concatenate([input_features, gating_resized], axis=-1)

    x = BatchNormalization()(concat_feat)
    x = Conv2D(filters, (1, 1), padding="same", activation="relu")(x)
    x = Conv2D(1, (1, 1), padding="same")(x)
    x = BatchNormalization()(x)
    alphas = Activation("sigmoid")(x)

    alphas_plus_1 = Lambda(lambda t: t + 1.0)(alphas)
    out = Multiply()([input_features, alphas_plus_1])

    out = Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal")(out)
    return out


def MEA_Block(edge_feat, main_feat, filters):
    u_0 = Conv2D(filters, (1, 1), padding="same")(edge_feat)

    r1 = conv_block(filters)(u_0)
    delta_u_0 = conv_block(filters, activation=None)(r1)
    u_1 = Add()([u_0, delta_u_0])
    u_1 = Activation("relu")(u_1)

    r2 = conv_block(filters)(u_1)
    r2 = conv_block(filters, activation=None)(r2)
    u_2 = Add()([u_1, r2])
    u_2 = Activation("relu")(u_2)

    u_3_pre = GatedSpatialConv2d(u_2, main_feat, filters)

    term1 = Lambda(lambda t: t * 3.0)(delta_u_0)
    sum_all = Add()([term1, u_2, u_3_pre])
    return sum_all


def model_unet_kaggle(img_height, img_width, img_channel, num_classes):
    inputs = Input((img_height, img_width, img_channel))

    s = Lambda(preprocess_rgb, name="sobel_preprocess")(inputs)
    edge_input = Conv2D(32, (3, 3), padding="same")(s)

    norm_inputs = Lambda(lambda x: x / 255.0)(inputs)

    # ===== Encoder =====
    c1 = conv_block(16)(norm_inputs)
    c1 = Dropout(0.1)(c1)
    c1 = conv_block(16)(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = conv_block(32)(p1)
    c2 = Dropout(0.1)(c2)
    c2 = conv_block(32)(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = conv_block(64)(p2)
    c3 = Dropout(0.2)(c3)
    c3 = conv_block(64)(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = conv_block(128)(p3)
    c4 = Dropout(0.2)(c4)
    c4 = conv_block(128)(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    c5 = conv_block(256)(p4)
    c5 = Dropout(0.3)(c5)
    c5 = conv_block(256)(c5)

    # ===== Decoder =====
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(c5)
    u6 = concatenate([u6, c4])
    c6 = conv_block(128)(u6)
    c6 = Dropout(0.2)(c6)
    c6 = conv_block(128)(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c6)
    c7_fused = HVD_Block(c3, u7, filters=64)
    c7 = conv_block(64)(c7_fused)
    c7 = Dropout(0.2)(c7)
    c7 = conv_block(64)(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(c7)
    c8_fused = HVD_Block(c2, u8, filters=32)
    c8 = conv_block(32)(c8_fused)
    c8 = Dropout(0.1)(c8)
    c8 = conv_block(32)(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding="same")(c8)
    u9 = concatenate([u9, c1])
    c9 = conv_block(16)(u9)
    c9 = Dropout(0.1)(c9)
    c9 = conv_block(16)(c9)

    # ===== Edge Branch (MEA_Block) =====
    edge_feat_1 = MEA_Block(edge_input, c3, filters=32)
    edge_feat_2 = MEA_Block(edge_feat_1, c7, filters=32)
    edge_feat_3 = MEA_Block(edge_feat_2, c8, filters=32)

    edge_output = Conv2D(1, (1, 1), activation="sigmoid", name="edge_output")(edge_feat_3)

    # ===== Fusion =====
    attention_map = edge_output
    att_feat = Multiply(name="att_mul")([c9, attention_map])

    fused_feat = Add(name="att_add")([c9, att_feat])

    final_x = conv_block(16)(fused_feat)
    seg_output = Conv2D(1, (1, 1), activation="sigmoid", name="seg_output")(final_x)

    model = Model(inputs=[inputs], outputs=[seg_output])
    return model

# def multi_scale_block(x, filters):
#     """
#    
#     """
#     
#     branch1 = Conv2D(filters, (1, 1), padding='same', kernel_initializer='he_normal')(x)
#     branch1 = BatchNormalization()(branch1)
#     branch1 = Activation('relu')(branch1)

#    
#     branch2 = Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal')(x)
#     branch2 = BatchNormalization()(branch2)
#     branch2 = Activation('relu')(branch2)

#     
#     branch3 = Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal')(x)
#     branch3 = Activation('relu')(branch3)
#     branch3 = Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal')(branch3)
#     branch3 = BatchNormalization()(branch3)
#     branch3 = Activation('relu')(branch3)

#    
#     concat_out = concatenate([branch1, branch2, branch3], axis=-1)

#     
#     out = Conv2D(filters, (1, 1), padding='same', kernel_initializer='he_normal')(concat_out)
#     out = BatchNormalization()(out)
#     out = Activation('relu')(out)

#     return out


# def model_unet_kaggle(img_height, img_width, img_channel, num_classes):
#     """
#     Args:
#         img_height (int):
#         img_width (int): 
#         img_channel (int): 
#         num_classes (int):
#     Returns:
#         model: Keras Model 
#     """

#     # 1
#     inputs = Input((img_height, img_width, img_channel))

#     # ---------------- Encoder  ----------------
#     # 
#     s = Lambda(lambda x: x / 255.0)(inputs)

#     # Layer 1
#     c1 = multi_scale_block(s, 16)
#     p1 = MaxPooling2D((2, 2))(c1)
#     p1 = Dropout(0.1)(p1)

#     # Layer 2
#     c2 = multi_scale_block(p1, 32)
#     p2 = MaxPooling2D((2, 2))(c2)
#     p2 = Dropout(0.1)(p2)

#     # Layer 3
#     c3 = multi_scale_block(p2, 64)
#     p3 = MaxPooling2D((2, 2))(c3)
#     p3 = Dropout(0.2)(p3)

#     # Layer 4
#     c4 = multi_scale_block(p3, 128)
#     p4 = MaxPooling2D((2, 2))(c4)
#     p4 = Dropout(0.2)(p4)

#     # ---------------- Bottleneck  ----------------
#     c5 = multi_scale_block(p4, 256)

#     # ---------------- Decoder  ----------------
#     # Layer 6 (Up 1)
#     u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
#     u6 = concatenate([u6, c4])  # 
#     c6 = multi_scale_block(u6, 128)  # 

#     # Layer 7 (Up 2)
#     u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
#     u7 = concatenate([u7, c3])  # 64 + 64 = 128
#     c7 = multi_scale_block(u7, 64)  #  64

#     # Layer 8 (Up 3)
#     u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
#     u8 = concatenate([u8, c2])
#     c8 = multi_scale_block(u8, 32)  # 32

#     # Layer 9 (Up 4)
#     u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
#     u9 = concatenate([u9, c1])
#     c9 = multi_scale_block(u9, 16)  # 16

#     # ---------------- Output ----------------
#     # 
#     if num_classes == 1:
#         activation = 'sigmoid'
#     else:
#         activation = 'softmax'

#     outputs = Conv2D(num_classes, (1, 1), activation=activation)(c9)

#     model = Model(inputs=[inputs], outputs=[outputs], name="U3UNet_Kaggle")

#     return model

# def res_conv_block(x, filters):
#     """
#    (Residual Block)
#     Input -> [Conv-BN-ReLU] -> [Conv-BN] -> Add(Input, Output) -> ReLU
#     """
#
#     # --- Shortcut Path  ---
#     shortcut = x
#     # 
#     if x.shape[-1] != filters:
#         shortcut = Conv2D(filters, (1, 1), padding='same')(shortcut)
#         shortcut = BatchNormalization()(shortcut)
#
#     # --- Main Path ---
#     # 
#     c = Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal')(x)
#     c = BatchNormalization()(c)
#     c = Activation('relu')(c)
#
#     # 
#     c = Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal')(c)
#     c = BatchNormalization()(c)
#
#     # --- Addition & Activation ---
#     # 
#     x = Add()([c, shortcut])
#     x = Activation('relu')(x)
#
#     return x
#
#
# def model_unet_kaggle(img_height, img_width, img_channel, num_classes):#resnet
#     inputs = Input((img_height, img_width, img_channel))
#
#     # 
#     s = tf.cast(inputs, tf.float32) / 255.0
#
#     # ===== Encoder  =====
#
#     # Level 1
#     c1 = res_conv_block(s, 16)
#     p1 = MaxPooling2D((2, 2))(c1)
#     p1 = Dropout(0.1)(p1)
#
#     # Level 2
#     c2 = res_conv_block(p1, 32)
#     p2 = MaxPooling2D((2, 2))(c2)
#     p2 = Dropout(0.1)(p2)
#
#     # Level 3
#     c3 = res_conv_block(p2, 64)
#     p3 = MaxPooling2D((2, 2))(c3)
#     p3 = Dropout(0.2)(p3)
#
#     # Level 4
#     c4 = res_conv_block(p3, 128)
#     p4 = MaxPooling2D((2, 2))(c4)
#     p4 = Dropout(0.2)(p4)
#
#     # ===== Bridge =====
#     c5 = res_conv_block(p4, 256)
#     c5 = Dropout(0.3)(c5)
#
#     # ===== Decoder  =====
#
#     # Up 1
#     u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
#     u6 = concatenate([u6, c4])
#     c6 = res_conv_block(u6, 128)
#     c6 = Dropout(0.2)(c6)
#
#     # Up 2
#     u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
#     u7 = concatenate([u7, c3])
#     c7 = res_conv_block(u7, 64)
#     c7 = Dropout(0.2)(c7)
#
#     # Up 3
#     u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
#     u8 = concatenate([u8, c2])
#     c8 = res_conv_block(u8, 32)
#     c8 = Dropout(0.1)(c8)
#
#     # Up 4
#     u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
#     u9 = concatenate([u9, c1])
#     c9 = res_conv_block(u9, 16)
#     c9 = Dropout(0.1)(c9)
#
#     # ===== Output =====
#     if num_classes == 1:
#         outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
#     else:
#         outputs = Conv2D(num_classes, (1, 1), activation='softmax')(c9)
#
#     model = Model(inputs=[inputs], outputs=[outputs], name="ResUNet_Kaggle")
#
#     return model
#################msunet####################
# def multi_scale_block(x, filters, kernel_size=3):
#     """
#     ：
#     
#     
#     """
#     # 
#     c1 = Conv2D(filters, kernel_size, padding='same', activation='relu')(x)
#     c1 = BatchNormalization()(c1)
#
#     # 
#     # 
#     c2 = Conv2D(filters, kernel_size, padding='same', dilation_rate=2, activation='relu')(x)
#     c2 = BatchNormalization()(c2)
#
#     #
#     x = Concatenate()([c1, c2])
#
#     # 
#     x = Conv2D(filters, (1, 1), padding='same', activation='relu')(x)
#     x = BatchNormalization()(x)
#
#     return x
#
#
# def model_unet_kaggle(img_height, img_width, img_channel, num_classes):
#     """
#     Args:
#         img_height (int): 
#         img_width (int):
#         img_channel (int): 
#         num_classes (int): 
#     Returns:
#         model: Keras Model 
#     """
#
#     # 
#     inputs = Input((img_height, img_width, img_channel))
#
#     # ---------------- Encoder  ----------------
#     #
#     s = Lambda(lambda x: x / 255.0)(inputs)
#
#
#     # Layer 1
#     c1 = multi_scale_block(s, 16)
#     p1 = MaxPooling2D((2, 2))(c1)
#     p1 = Dropout(0.1)(p1)
#
#     # Layer 2
#     c2 = multi_scale_block(p1, 32)
#     p2 = MaxPooling2D((2, 2))(c2)
#     p2 = Dropout(0.1)(p2)
#
#     # Layer 3
#     c3 = multi_scale_block(p2, 64)
#     p3 = MaxPooling2D((2, 2))(c3)
#     p3 = Dropout(0.2)(p3)
#
#     # Layer 4
#     c4 = multi_scale_block(p3, 128)
#     p4 = MaxPooling2D((2, 2))(c4)
#     p4 = Dropout(0.2)(p4)
#
#     # ---------------- Bottleneck  ----------------
#     c5 = multi_scale_block(p4, 256)
#
#     # ---------------- Decoder  ----------------
#     # Layer 6 (Up 1)
#     u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
#     u6 = concatenate([u6, c4])
#     c6 = multi_scale_block(u6, 128)
#
#     # Layer 7 (Up 2)
#     u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
#     u7 = concatenate([u7, c3])
#     c7 = multi_scale_block(u7, 64)
#
#     # Layer 8 (Up 3)
#     u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
#     u8 = concatenate([u8, c2])
#     c8 = multi_scale_block(u8, 32)
#
#     # Layer 9 (Up 4)
#     u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
#     u9 = concatenate([u9, c1])
#     c9 = multi_scale_block(u9, 16)
#
#     # ---------------- Output  ----------------
#
#     outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
#
#
#     model = Model(inputs=[inputs], outputs=[outputs], name="MSUNet_Kaggle")
#
#     return model

#########################################################
# 
#########################################################
def segmentation_keras_load():
    # -------- inf--------
    batch_size = config_segmentation.get("batch_size")
    img_size = (segmentation_new_size.get("width"), segmentation_new_size.get("height"))
    img_width = img_size[0]
    img_height = img_size[1]
    epochs = config_segmentation.get("Epochs")
    img_channels = config_segmentation.get("CHANNELS")
    num_classes = config_segmentation.get("num_class")

    # -------- menu save--------
    train_img_dir = r"D:\Drone\autumn Fire-Detection-UAV-Aerial-Image-Classification-Segmentation-UnmannedAerialVehicle-main\Fire-Detection-UAV-Aerial-Image-Classification-Segmentation-UnmannedAerialVehicle-main\frames\Segmentation\train\Images"
    train_mask_dir = r"D:\Drone\autumn Fire-Detection-UAV-Aerial-Image-Classification-Segmentation-UnmannedAerialVehicle-main\Fire-Detection-UAV-Aerial-Image-Classification-Segmentation-UnmannedAerialVehicle-main\frames\Segmentation\train\Masks"

    val_img_dir = r"D:\Drone\autumn Fire-Detection-UAV-Aerial-Image-Classification-Segmentation-UnmannedAerialVehicle-main\Fire-Detection-UAV-Aerial-Image-Classification-Segmentation-UnmannedAerialVehicle-main\frames\Segmentation\val\Images"
    val_mask_dir = r"D:\Drone\autumn Fire-Detection-UAV-Aerial-Image-Classification-Segmentation-UnmannedAerialVehicle-main\Fire-Detection-UAV-Aerial-Image-Classification-Segmentation-UnmannedAerialVehicle-main\frames\Segmentation\val\Masks"

    test_img_dir = r"D:\Drone\autumn Fire-Detection-UAV-Aerial-Image-Classification-Segmentation-UnmannedAerialVehicle-main\Fire-Detection-UAV-Aerial-Image-Classification-Segmentation-UnmannedAerialVehicle-main\frames\Segmentation\test\Images"
    test_mask_dir = r"D:\Drone\autumn Fire-Detection-UAV-Aerial-Image-Classification-Segmentation-UnmannedAerialVehicle-main\Fire-Detection-UAV-Aerial-Image-Classification-Segmentation-UnmannedAerialVehicle-main\frames\Segmentation\test\Masks"

    # --------  Train / Val / Test --------
    x_train, y_train, train_mask_names = load_dataset(
        train_img_dir, train_mask_dir, img_size, img_channels, desc="Train"
    )
    x_val, y_val, val_mask_names = load_dataset(
        val_img_dir, val_mask_dir, img_size, img_channels, desc="Val"
    )
    x_test, y_test, test_mask_names = load_dataset(
        test_img_dir, test_mask_dir, img_size, img_channels, desc="Test"
    )

    print("[INFO] Train samples:", len(x_train))
    print("[INFO] Val samples  :", len(x_val))
    print("[INFO] Test samples :", len(x_test))

   
    model = model_unet_kaggle(img_height, img_width, img_channels, num_classes)

    model_fig_file = "Output/Model_figure/segmentation_model_u_net.png"
    os.makedirs(os.path.dirname(model_fig_file), exist_ok=True)
    # tf.keras.utils.plot_model(model, to_file=model_fig_file, show_shapes=True)

   
    steps_per_epoch = max(1, len(x_train) // batch_size)

    initial_learning_rate = 1e-4
    first_decay_epochs = 10
    first_decay_steps = steps_per_epoch * first_decay_epochs

    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=initial_learning_rate,
        first_decay_steps=first_decay_steps,
        t_mul=2.0,
        m_mul=0.9,
        alpha=1e-5,
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, epsilon=1e-07
    )

    model.compile(optimizer=optimizer, loss=combo_edge_dice_loss, metrics=METRICS)

   
    early_stopper = tf.keras.callbacks.EarlyStopping(
        patience=500, monitor="val_loss", mode="min", restore_best_weights=True
    )

    test_callback = TestEvalCallback(
        x_test=x_test,
        y_test=y_test,
        save_path="FireSegmentation.h5",
        batch_size=batch_size,
    )

    results = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopper, test_callback],
    )

    # --------  best-test-IoU  --------
    print("[INFO] 加载 best TEST IoU 模型 FireSegmentation.h5 ...")
    model_predict = tf.keras.models.load_model(
        "FireSegmentation.h5",
        custom_objects={"preprocess_rgb": preprocess_rgb},
        compile=False,
    )

    output_root = r"D:\Drone\autumn Fire-Detection-UAV-Aerial-Image-Classification-Segmentation-UnmannedAerialVehicle-main\Fire-Detection-UAV-Aerial-Image-Classification-Segmentation-UnmannedAerialVehicle-main\Output\SegmentationResults\u3u"
    os.makedirs(output_root, exist_ok=True)
    curve_save_dir = os.path.join(output_root, "training_curves")

    plot_training_curves(results, save_dir=curve_save_dir)

    # 
    mask_save_dir = os.path.join(output_root, "test_masks_error_analysis")

    save_error_analysis_masks(
        model_predict=model_predict,
        x_val=x_test,
        y_val=y_test,  #  y_test compare
        save_dir=mask_save_dir,
        threshold=0.5,
        mask_names=test_mask_names,
    )

    # 2)  [Image | GT | Pred]
    num_vis = min(6, len(x_test))
    if num_vis > 0:
        vis_indices = np.arange(num_vis)
        x_vis = x_test[vis_indices]
        y_vis = y_test[vis_indices]

        pred_vis = model_predict.predict(x_vis, batch_size=1, verbose=1)
        if isinstance(pred_vis, (list, tuple)):
            pred_vis_seg = pred_vis[0]
        else:
            pred_vis_seg = pred_vis

        pred_vis_t = (pred_vis_seg > 0.5).astype(np.uint8)

        vis_save_dir = os.path.join(output_root, "vis_test")
        save_segmentation_results(
            x_val=x_vis,
            y_val=y_vis,
            y_pred=pred_vis_t,
            save_dir=vis_save_dir,
            num_samples=num_vis,
        )

    print(f"[INFO] 误差分析图片已保存至: {mask_save_dir}")


if __name__ == "__main__":
    segmentation_keras_load()

    #########################################################
    # 
    #########################################################
    # def model_unet_kaggle(img_height, img_width, img_channel, num_classes):
    #     """
    #    
    #     """
    #     inputs = Input((img_height, img_width, img_channel))
    #
    #    
    #
    #     norm_inputs = Lambda(lambda x: x / 255.0)(inputs)
    #
    #     # ===== Encoder  =====
    #     c1 = conv_block(16)(norm_inputs)
    #     c1 = Dropout(0.1)(c1)
    #     c1 = conv_block(16)(c1)
    #     p1 = MaxPooling2D((2, 2))(c1)
    #
    #     c2 = conv_block(32)(p1)
    #     c2 = Dropout(0.1)(c2)
    #     c2 = conv_block(32)(c2)
    #     p2 = MaxPooling2D((2, 2))(c2)
    #
    #     c3 = conv_block(64)(p2)
    #     c3 = Dropout(0.2)(c3)
    #     c3 = conv_block(64)(c3)
    #     p3 = MaxPooling2D((2, 2))(c3)
    #
    #     c4 = conv_block(128)(p3)
    #     c4 = Dropout(0.2)(c4)
    #     c4 = conv_block(128)(c4)
    #     p4 = MaxPooling2D((2, 2))(c4)
    #
    #     c5 = conv_block(256)(p4)
    #     c5 = Dropout(0.3)(c5)
    #     c5 = conv_block(256)(c5)
    #
    #     # ===== Decoder =====
    #     u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(c5)
    #     u6 = concatenate([u6, c4])
    #     c6 = conv_block(128)(u6)
    #     c6 = Dropout(0.2)(c6)
    #     c6 = conv_block(128)(c6)
    #
    #     u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c6)
    #     # 
    #     c7_fused = HVD_Block(c3, u7, filters=64)
    #     c7 = conv_block(64)(c7_fused)
    #     c7 = Dropout(0.2)(c7)
    #     c7 = conv_block(64)(c7)
    #
    #     u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(c7)
    #     # 
    #     c8_fused = HVD_Block(c2, u8, filters=32)
    #     c8 = conv_block(32)(c8_fused)
    #     c8 = Dropout(0.1)(c8)
    #     c8 = conv_block(32)(c8)
    #
    #     u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding="same")(c8)
    #     u9 = concatenate([u9, c1])
    #     c9 = conv_block(16)(u9)
    #     c9 = Dropout(0.1)(c9)
    #     c9 = conv_block(16)(c9)
    #
    #     # 
    #     # 
    #
    #     # 
    #     seg_output = Conv2D(1, (1, 1), activation="sigmoid", name="seg_output")(c9)
    #
    #     model = Model(inputs=[inputs], outputs=[seg_output])
    #     return model
    #########################################################
    # 
    #########################################################
    # def model_unet_kaggle(img_height, img_width, img_channel, num_classes):
    #     """
    #     
    #     
    #     
    #     """
    #     inputs = Input((img_height, img_width, img_channel))
    #
    #     
    #     s = Lambda(preprocess_rgb, name="sobel_preprocess")(inputs)
    #     edge_input = Conv2D(32, (3, 3), padding="same")(s)
    #
    #     norm_inputs = Lambda(lambda x: x / 255.0)(inputs)
    #
    #     # ===== Encoder =====
    #     c1 = conv_block(16)(norm_inputs)
    #     c1 = Dropout(0.1)(c1)
    #     c1 = conv_block(16)(c1)
    #     p1 = MaxPooling2D((2, 2))(c1)
    #
    #     c2 = conv_block(32)(p1)
    #     c2 = Dropout(0.1)(c2)
    #     c2 = conv_block(32)(c2)
    #     p2 = MaxPooling2D((2, 2))(c2)
    #
    #     c3 = conv_block(64)(p2)
    #     c3 = Dropout(0.2)(c3)
    #     c3 = conv_block(64)(c3)
    #     p3 = MaxPooling2D((2, 2))(c3)
    #
    #     c4 = conv_block(128)(p3)
    #     c4 = Dropout(0.2)(c4)
    #     c4 = conv_block(128)(c4)
    #     p4 = MaxPooling2D((2, 2))(c4)
    #
    #     c5 = conv_block(256)(p4)
    #     c5 = Dropout(0.3)(c5)
    #     c5 = conv_block(256)(c5)
    #
    #     
    #     u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(c5)
    #     u6 = concatenate([u6, c4]) 
    #     c6 = conv_block(128)(u6)
    #     c6 = Dropout(0.2)(c6)
    #     c6 = conv_block(128)(c6)
    #
    #     u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c6)
    #     
    #     u7 = concatenate([u7, c3])
    #     c7 = conv_block(64)(u7)
    #     c7 = Dropout(0.2)(c7)
    #     c7 = conv_block(64)(c7)
    #
    #     u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(c7)
    #    
    #     u8 = concatenate([u8, c2])
    #     c8 = conv_block(32)(u8)
    #     c8 = Dropout(0.1)(c8)
    #     c8 = conv_block(32)(c8)
    #
    #     u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding="same")(c8)
    #     u9 = concatenate([u9, c1]) # 
    #     c9 = conv_block(16)(u9)
    #     c9 = Dropout(0.1)(c9)
    #     c9 = conv_block(16)(c9)
    #
    #     
    #     # MEA_Block 
    #     edge_feat_1 = MEA_Block(edge_input, c3, filters=32)
    #     edge_feat_2 = MEA_Block(edge_feat_1, c7, filters=32)
    #     edge_feat_3 = MEA_Block(edge_feat_2, c8, filters=32)
    #
    #     edge_output = Conv2D(1, (1, 1), activation="sigmoid", name="edge_output")(edge_feat_3)
    #
    #     # ===== Fusion  =====
    #     attention_map = edge_output
    #     att_feat = Multiply(name="att_mul")([c9, attention_map])
    #
    #     fused_feat = Add(name="att_add")([c9, att_feat])
    #
    #     final_x = conv_block(16)(fused_feat)
    #     seg_output = Conv2D(1, (1, 1), activation="sigmoid", name="seg_output")(final_x)
    #
    #     model = Model(inputs=[inputs], outputs=[seg_output])
    #     return model
    #########################################################
    # 
    #########################################################
    # def model_unet_kaggle(img_height, img_width, img_channel, num_classes):
    #     """
    #     
    #    
    #     
    #     
    #     """
    #     inputs = Input((img_height, img_width, img_channel))
    #     norm_inputs = Lambda(lambda x: x / 255.0)(inputs)
    #
    #     
    #     
    #     
    #     
    #     edge_input = Conv2D(32, (3, 3), padding="same", name="learnable_edge_input")(norm_inputs)
    #
    #     # ===== Encoder =====
    #     c1 = conv_block(16)(norm_inputs)
    #     c1 = Dropout(0.1)(c1)
    #     c1 = conv_block(16)(c1)
    #     p1 = MaxPooling2D((2, 2))(c1)
    #
    #     c2 = conv_block(32)(p1)
    #     c2 = Dropout(0.1)(c2)
    #     c2 = conv_block(32)(c2)
    #     p2 = MaxPooling2D((2, 2))(c2)
    #
    #     c3 = conv_block(64)(p2)
    #     c3 = Dropout(0.2)(c3)
    #     c3 = conv_block(64)(c3)
    #     p3 = MaxPooling2D((2, 2))(c3)
    #
    #     c4 = conv_block(128)(p3)
    #     c4 = Dropout(0.2)(c4)
    #     c4 = conv_block(128)(c4)
    #     p4 = MaxPooling2D((2, 2))(c4)
    #
    #     c5 = conv_block(256)(p4)
    #     c5 = Dropout(0.3)(c5)
    #     c5 = conv_block(256)(c5)
    #
    #     
    #     u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(c5)
    #     u6 = concatenate([u6, c4])
    #     c6 = conv_block(128)(u6)
    #     c6 = Dropout(0.2)(c6)
    #     c6 = conv_block(128)(c6)
    #
    #     u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c6)
    #    
    #     u7 = concatenate([u7, c3])
    #     c7 = conv_block(64)(u7)
    #     c7 = Dropout(0.2)(c7)
    #     c7 = conv_block(64)(c7)
    #
    #     u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(c7)
    #     
    #     u8 = concatenate([u8, c2])
    #     c8 = conv_block(32)(u8)
    #     c8 = Dropout(0.1)(c8)
    #     c8 = conv_block(32)(c8)
    #
    #     u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding="same")(c8)
    #     u9 = concatenate([u9, c1])
    #     c9 = conv_block(16)(u9)
    #     c9 = Dropout(0.1)(c9)
    #     c9 = conv_block(16)(c9)
    #
    #     
    #     
    #     edge_feat_1 = MEA_Block(edge_input, c3, filters=32)
    #     edge_feat_2 = MEA_Block(edge_feat_1, c7, filters=32)
    #     edge_feat_3 = MEA_Block(edge_feat_2, c8, filters=32)
    #
    #     edge_output = Conv2D(1, (1, 1), activation="sigmoid", name="edge_output")(edge_feat_3)
    #
    #     
    #     attention_map = edge_output
    #     att_feat = Multiply(name="att_mul")([c9, attention_map])
    #
    #     fused_feat = Add(name="att_add")([c9, att_feat])
    #
    #     final_x = conv_block(16)(fused_feat)
    #     seg_output = Conv2D(1, (1, 1), activation="sigmoid", name="seg_output")(final_x)
    #
    #     model = Model(inputs=[inputs], outputs=[seg_output])
    #     return model
    #########################################################
    # 主模型：U-Net 
    #########################################################
    # def model_unet_kaggle(img_height, img_width, img_channel, num_classes):
    #     """
    #    
    #     [✓] preprocess_rgb (Sobel)
    #     [✓] TFD Block
    #     [x] TOAA Block (concat)
    #     """
    #     inputs = Input((img_height, img_width, img_channel))
    #
    #     
    #     s = Lambda(preprocess_rgb, name="sobel_preprocess")(inputs)
    #     edge_input = Conv2D(32, (3, 3), padding="same")(s)
    #
    #     norm_inputs = Lambda(lambda x: x / 255.0)(inputs)
    #
    #     
    #     c1 = conv_block(16)(norm_inputs)
    #     c1 = Dropout(0.1)(c1)
    #     c1 = conv_block(16)(c1)
    #     p1 = MaxPooling2D((2, 2))(c1)
    #
    #     c2 = conv_block(32)(p1)
    #     c2 = Dropout(0.1)(c2)
    #     c2 = conv_block(32)(c2)
    #     p2 = MaxPooling2D((2, 2))(c2)
    #
    #     c3 = conv_block(64)(p2)
    #     c3 = Dropout(0.2)(c3)
    #     c3 = conv_block(64)(c3)
    #     p3 = MaxPooling2D((2, 2))(c3)
    #
    #     c4 = conv_block(128)(p3)
    #     c4 = Dropout(0.2)(c4)
    #     c4 = conv_block(128)(c4)
    #     p4 = MaxPooling2D((2, 2))(c4)
    #
    #     c5 = conv_block(256)(p4)
    #     c5 = Dropout(0.3)(c5)
    #     c5 = conv_block(256)(c5)
    #
    #     
    #     u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(c5)
    #     u6 = concatenate([u6, c4]) 
    #     c6 = conv_block(128)(u6)
    #     c6 = Dropout(0.2)(c6)
    #     c6 = conv_block(128)(c6)
    #
    #     u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c6)
    #     # [移除 TOAA] ->  concatenate
    #     u7 = concatenate([u7, c3])
    #     c7 = conv_block(64)(u7)
    #     c7 = Dropout(0.2)(c7)
    #     c7 = conv_block(64)(c7)
    #
    #     u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(c7)
    #     # [移除 TOAA] ->  concatenate
    #     u8 = concatenate([u8, c2])
    #     c8 = conv_block(32)(u8)
    #     c8 = Dropout(0.1)(c8)
    #     c8 = conv_block(32)(c8)
    #
    #     u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding="same")(c8)
    #     u9 = concatenate([u9, c1])
    #     c9 = conv_block(16)(u9)
    #     c9 = Dropout(0.1)(c9)
    #     c9 = conv_block(16)(c9)
    #
    #     
    #     
    #     edge_feat_1 = MEA_Block(edge_input, c3, filters=32)
    #     edge_feat_2 = MEA_Block(edge_feat_1, c7, filters=32)
    #     edge_feat_3 = MEA_Block(edge_feat_2, c8, filters=32)
    #
    #     edge_output = Conv2D(1, (1, 1), activation="sigmoid", name="edge_output")(edge_feat_3)
    #
    #     
    #    
    #     attention_map = edge_output
    #     att_feat = Multiply(name="att_mul")([c9, attention_map])
    #
    #     fused_feat = Add(name="att_add")([c9, att_feat])
    #
    #     final_x = conv_block(16)(fused_feat)
    #     seg_output = Conv2D(1, (1, 1), activation="sigmoid", name="seg_output")(final_x)
    #
    #     model = Model(inputs=[inputs], outputs=[seg_output])
    #     return model
