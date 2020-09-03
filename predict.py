
from xml.dom.minidom import Document
from model import *
from tensorflow.keras import Model
import os
import cv2
import numpy as np
import tqdm
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from multiprocessing import Pool

def get_normal_img(image_file):
    img = cv2.imread(image_file).astype(np.float32)
    img = np.float32(img) / 127.5 - 1
    return img

def write_img_result(tmp):
    
    pr = tmp[0]
    index = tmp[1]
    n_class = tmp[2]
    output_path = tmp[3]
    
    pr = pr.reshape((256, 256, n_class)).argmax(axis=2)
    
    #print(pr)
    '''
    print(index)
    print(n_class)
    print(output_path)
    '''
    seg_img = np.zeros((256, 256), dtype=np.uint16)
    for c in range(n_class):
        seg_img[pr[:, :] == c] = c
    seg_img = cv2.resize(seg_img, (256, 256), interpolation=cv2.INTER_NEAREST)
    save_img = np.zeros((256, 256), dtype=np.uint16)
    for i in range(256):
        for j in range(256):
            save_img[i][j] = matches[int(seg_img[i][j])]
    cv2.imwrite(os.path.join(output_path, index+".png"), save_img)

def predict(image_file, index, model, output_path, n_class, weights_path=None):
    if weights_path is not None:
        model.load_weights(weights_path)
    img = cv2.imread(image_file).astype(np.float32)
    img = np.float32(img) / 127.5 - 1
    pr = model.predict(np.array([img]))[0]
    pr = pr.reshape((256,  256, n_class)).argmax(axis=2)
    seg_img = np.zeros((256, 256), dtype=np.uint16)
    for c in range(n_class):
        seg_img[pr[:, :] == c] = c
    seg_img = cv2.resize(seg_img, (256, 256), interpolation=cv2.INTER_NEAREST)
    save_img = np.zeros((256, 256), dtype=np.uint16)
    for i in range(256):
        for j in range(256):
            save_img[i][j] = matches[int(seg_img[i][j])]
    cv2.imwrite(os.path.join(output_path, index+".png"), save_img)


def predict_all(input_path, output_path, model, n_class, weights_path=None):
    ## 预测一个文件夹内的所有图片
    ## input_path：传入图像文件夹
    ## output_path：保存预测图片的文件夹
    ## model：传入模型
    # n_class：类别数量
    # weights_path：权重保存路径
    if weights_path is not None:
        model.load_weights(weights_path)
    for image in os.listdir(input_path):
        print(image)
        index, _ = os.path.splitext(image)
        predict(os.path.join(input_path, image),
                index, model, output_path, n_class)


def predict_batch(input_path, output_path, model, n_class, weights_path=None):
    if weights_path is not None:
        model.load_weights(weights_path)
    batch_size = 128
    all_sample =100000
    step = all_sample//batch_size
    all_imgs = os.listdir(input_path)
    for i in tqdm.trange(step):
        end_sample = (i+1)*batch_size
        if (i+1)*batch_size>all_sample:
            end_sample = all_sample
        pre_imgs_list = all_imgs[i*batch_size:end_sample]

        pre_img = []
        for img_path in pre_imgs_list:
            pre_img.append(get_normal_img(os.path.join(input_path, img_path)))
        res = model.predict(np.array(pre_img))
        indexs = [pre_imgs_list[now_pr].split('.')[0] for now_pr in range(128)]
        output_paths = [output_path]*batch_size
        n_classes = [n_class]*batch_size
        tmp = zip(res, indexs, n_classes, output_paths)
        pool = Pool(16)
        pool.map(write_img_result, tmp)
        pool.close()
        pool.join()

if __name__ == "__main__":
    weights_path = "best.h5"
    input_path = "../data/image_A/"
    output_path = "test/labels/"
    n_class = 8

    model = unet(n_class)
    model.load_weights(weights_path)  # 读取训练的权重
    #predict_all(input_path, output_path, model, n_class, weights_path)

    predict_batch(input_path, output_path, model, n_class, weights_path=weights_path)
