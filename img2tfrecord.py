import tensorflow as tf
import os


def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def image_to_tfrecord(image_dir, output_path):
    with tf.io.TFRecordWriter(output_path) as writer:
        for filename in os.listdir(image_dir):
            if filename.endswith((".jpg", ".png", ".jpeg")):
                img_path = os.path.join(image_dir, filename)

                # 1. 讀取圖片原始位元組 (Raw Bytes)
                image_raw = open(img_path, 'rb').read()

                # 2. 假設從檔名獲取標籤 (例如: cat_01.jpg -> 0)
                # 這裡你可以根據你的需求自定義 label 邏輯
                label = 0 if 'cat' in filename else 1

                # 3. 構建 Feature 字典
                feature = {
                    'image_raw': _bytes_feature(image_raw),
                    'label': _int64_feature(label),
                }

                # 4. 建立 Example 並寫入
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
    print(f"Save to: {output_path}")

def main():
    image_dir = '../data/PLUSVein-FV3/PLUSVein-FV3-SingleFinger/SingleFinger/PLUS-FV3-Laser/DORSAL/01/001'  # 替換為你的圖片目錄
    output_path = 'output.tfrecord'     # 替換為你想要的輸出路徑
    image_to_tfrecord(image_dir, output_path)

if __name__ == "__main__":
    main()