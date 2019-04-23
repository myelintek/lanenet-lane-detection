#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-18 下午7:31
# @Author  : Luo Yao
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : train_lanenet.py
# @IDE: PyCharm Community Edition
"""
训练lanenet模型
"""
import argparse
import math
import os
import os.path as ops
import time

import cv2
import glob
import glog as log
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


from config import global_config
from lanenet_model import lanenet_merge_model
from data_provider import lanenet_data_processor
from lanenet_model import lanenet_cluster
from lanenet_model import lanenet_postprocess

CFG = global_config.cfg
VGG_MEAN = [103.939, 116.779, 123.68]


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    #train args
    parser.add_argument('--dataset_dir', type=str, help='The training dataset dir path')
    parser.add_argument('--net', type=str, help='Which base net work to use', default='vgg')
    parser.add_argument('--checkpoint_path', type=str, help='The path to save checkpoints', default='/job_dir')
    #test args
    parser.add_argument('--is_batch', type=str, help='If test a batch of images', default='false')
    parser.add_argument('--batch_size', type=int, help='The batch size of the test images', default=32)
    parser.add_argument('--save_dir', type=str, help='Test result image save dir', default=None)
    parser.add_argument('--use_gpu', type=int, help='If use gpu set 1 or 0 instead', default=1)
    parser.add_argument('--image_path', type=str, help='The image path or the src image save dir')
    #both train and test
    parser.add_argument('--weights_path', type=str, help='The pretrained weights path')
    parser.add_argument('--train', type=bool, help='train or test', default=True)

    return parser.parse_args()


def minmax_scale(input_arr):
    """

    :param input_arr:
    :return:
    """
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)

    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

    return output_arr


def train_net(dataset_dir, weights_path=None, net_flag='vgg', job_dir='/job_dir'):
    """

    :param dataset_dir:
    :param net_flag: choose which base network to use
    :param weights_path:
    :return:
    """
    train_dataset_file = ops.join(dataset_dir, 'train.txt')
    val_dataset_file = ops.join(dataset_dir, 'val.txt')

    assert ops.exists(train_dataset_file)

    train_dataset = lanenet_data_processor.DataSet(train_dataset_file)
    val_dataset = lanenet_data_processor.DataSet(val_dataset_file)

    with tf.device('/gpu:1'):
        input_tensor = tf.placeholder(dtype=tf.float32,
                                      shape=[CFG.TRAIN.BATCH_SIZE, CFG.TRAIN.IMG_HEIGHT,
                                             CFG.TRAIN.IMG_WIDTH, 3],
                                      name='input_tensor')
        binary_label_tensor = tf.placeholder(dtype=tf.int64,
                                             shape=[CFG.TRAIN.BATCH_SIZE, CFG.TRAIN.IMG_HEIGHT,
                                                    CFG.TRAIN.IMG_WIDTH, 1],
                                             name='binary_input_label')
        instance_label_tensor = tf.placeholder(dtype=tf.float32,
                                               shape=[CFG.TRAIN.BATCH_SIZE, CFG.TRAIN.IMG_HEIGHT,
                                                      CFG.TRAIN.IMG_WIDTH],
                                               name='instance_input_label')
        phase = tf.placeholder(dtype=tf.string, shape=None, name='net_phase')

        net = lanenet_merge_model.LaneNet(net_flag=net_flag, phase=phase)

        # calculate the loss
        compute_ret = net.compute_loss(input_tensor=input_tensor, binary_label=binary_label_tensor,
                                       instance_label=instance_label_tensor, name='lanenet_model')
        total_loss = compute_ret['total_loss']
        binary_seg_loss = compute_ret['binary_seg_loss']
        disc_loss = compute_ret['discriminative_loss']
        pix_embedding = compute_ret['instance_seg_logits']

        # calculate the accuracy
        out_logits = compute_ret['binary_seg_logits']
        out_logits = tf.nn.softmax(logits=out_logits)
        out_logits_out = tf.argmax(out_logits, axis=-1)
        out = tf.argmax(out_logits, axis=-1)
        out = tf.expand_dims(out, axis=-1)

        idx = tf.where(tf.equal(binary_label_tensor, 1))
        pix_cls_ret = tf.gather_nd(out, idx)
        accuracy = tf.count_nonzero(pix_cls_ret)
        accuracy = tf.divide(accuracy, tf.cast(tf.shape(pix_cls_ret)[0], tf.int64))

        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(CFG.TRAIN.LEARNING_RATE, global_step,
                                                   100000, 0.1, staircase=True)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.MomentumOptimizer(
                learning_rate=learning_rate, momentum=0.9).minimize(loss=total_loss,
                                                                    var_list=tf.trainable_variables(),
                                                                    global_step=global_step)

    # Set tf saver
    saver = tf.train.Saver(max_to_keep=1)
    model_save_dir = job_dir
    if not ops.exists(model_save_dir):
        os.makedirs(model_save_dir)
    train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    model_name = 'tusimple_lanenet_{:s}_{:s}.ckpt'.format(net_flag, str(train_start_time))
    model_save_path = ops.join(model_save_dir, model_name)

    # Set tf summary
    tboard_save_path = 'tboard/tusimple_lanenet/{:s}'.format(net_flag)
    if not ops.exists(tboard_save_path):
        os.makedirs(tboard_save_path)
    train_cost_scalar = tf.summary.scalar(name='train_cost', tensor=total_loss)
    val_cost_scalar = tf.summary.scalar(name='val_cost', tensor=total_loss)
    train_accuracy_scalar = tf.summary.scalar(name='train_accuracy', tensor=accuracy)
    val_accuracy_scalar = tf.summary.scalar(name='val_accuracy', tensor=accuracy)
    train_binary_seg_loss_scalar = tf.summary.scalar(name='train_binary_seg_loss', tensor=binary_seg_loss)
    val_binary_seg_loss_scalar = tf.summary.scalar(name='val_binary_seg_loss', tensor=binary_seg_loss)
    train_instance_seg_loss_scalar = tf.summary.scalar(name='train_instance_seg_loss', tensor=disc_loss)
    val_instance_seg_loss_scalar = tf.summary.scalar(name='val_instance_seg_loss', tensor=disc_loss)
    learning_rate_scalar = tf.summary.scalar(name='learning_rate', tensor=learning_rate)
    train_merge_summary_op = tf.summary.merge([train_accuracy_scalar, train_cost_scalar,
                                               learning_rate_scalar, train_binary_seg_loss_scalar,
                                               train_instance_seg_loss_scalar])
    val_merge_summary_op = tf.summary.merge([val_accuracy_scalar, val_cost_scalar,
                                             val_binary_seg_loss_scalar, val_instance_seg_loss_scalar])

    # Set sess configuration
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    summary_writer = tf.summary.FileWriter(tboard_save_path)
    summary_writer.add_graph(sess.graph)

    # Set the training parameters
    train_epochs = CFG.TRAIN.EPOCHS

    log.info('Global configuration is as follows:')
    log.info(CFG)

    with sess.as_default():

        tf.train.write_graph(graph_or_graph_def=sess.graph, logdir='',
                             name='{:s}/lanenet_model.pb'.format(model_save_dir))

        if weights_path is None:
            log.info('Training from scratch')
            init = tf.global_variables_initializer()
            sess.run(init)
        else:
            log.info('Restore model from last model checkpoint {:s}'.format(weights_path))
            saver.restore(sess=sess, save_path=weights_path)

        # 加载预训练参数
        if net_flag == 'vgg' and weights_path is None:
            pretrained_weights = np.load(
                './data/vgg16.npy',
                encoding='latin1').item()

            for vv in tf.trainable_variables():
                weights_key = vv.name.split('/')[-3]
                try:
                    weights = pretrained_weights[weights_key][0]
                    _op = tf.assign(vv, weights)
                    sess.run(_op)
                except Exception as e:
                    continue

        train_cost_time_mean = []
        val_cost_time_mean = []
        for epoch in range(train_epochs):
            # training part
            t_start = time.time()

            with tf.device('/cpu:0'):
                gt_imgs, binary_gt_labels, instance_gt_labels = train_dataset.next_batch(CFG.TRAIN.BATCH_SIZE)
                gt_imgs = [cv2.resize(tmp,
                                      dsize=(CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT),
                                      dst=tmp,
                                      interpolation=cv2.INTER_LINEAR)
                           for tmp in gt_imgs]

                gt_imgs = [tmp - VGG_MEAN for tmp in gt_imgs]
                binary_gt_labels = [cv2.resize(tmp,
                                               dsize=(CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT),
                                               dst=tmp,
                                               interpolation=cv2.INTER_NEAREST)
                                    for tmp in binary_gt_labels]
                binary_gt_labels = [np.expand_dims(tmp, axis=-1) for tmp in binary_gt_labels]
                instance_gt_labels = [cv2.resize(tmp,
                                                 dsize=(CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT),
                                                 dst=tmp,
                                                 interpolation=cv2.INTER_NEAREST)
                                      for tmp in instance_gt_labels]
            phase_train = 'train'

            _, c, train_accuracy, train_summary, binary_loss, instance_loss, embedding, binary_seg_img = \
                sess.run([optimizer, total_loss,
                          accuracy,
                          train_merge_summary_op,
                          binary_seg_loss,
                          disc_loss,
                          pix_embedding,
                          out_logits_out],
                         feed_dict={input_tensor: gt_imgs,
                                    binary_label_tensor: binary_gt_labels,
                                    instance_label_tensor: instance_gt_labels,
                                    phase: phase_train})

            if math.isnan(c) or math.isnan(binary_loss) or math.isnan(instance_loss):
                log.error('cost is: {:.5f}'.format(c))
                log.error('binary cost is: {:.5f}'.format(binary_loss))
                log.error('instance cost is: {:.5f}'.format(instance_loss))
                cv2.imwrite('nan_image.png', gt_imgs[0] + VGG_MEAN)
                cv2.imwrite('nan_instance_label.png', instance_gt_labels[0])
                cv2.imwrite('nan_binary_label.png', binary_gt_labels[0] * 255)
                return

            if epoch % 100 == 0:
                cv2.imwrite('image.png', gt_imgs[0] + VGG_MEAN)
                cv2.imwrite('binary_label.png', binary_gt_labels[0] * 255)
                cv2.imwrite('instance_label.png', instance_gt_labels[0])
                cv2.imwrite('binary_seg_img.png', binary_seg_img[0] * 255)

                for i in range(4):
                    embedding[0][:, :, i] = minmax_scale(embedding[0][:, :, i])
                embedding_image = np.array(embedding[0], np.uint8)
                cv2.imwrite('embedding.png', embedding_image)

            cost_time = time.time() - t_start
            train_cost_time_mean.append(cost_time)
            summary_writer.add_summary(summary=train_summary, global_step=epoch)

            # validation part
            with tf.device('/cpu:0'):
                gt_imgs_val, binary_gt_labels_val, instance_gt_labels_val \
                    = val_dataset.next_batch(CFG.TRAIN.VAL_BATCH_SIZE)
                gt_imgs_val = [cv2.resize(tmp,
                                          dsize=(CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT),
                                          dst=tmp,
                                          interpolation=cv2.INTER_LINEAR)
                               for tmp in gt_imgs_val]
                gt_imgs_val = [tmp - VGG_MEAN for tmp in gt_imgs_val]
                binary_gt_labels_val = [cv2.resize(tmp,
                                                   dsize=(CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT),
                                                   dst=tmp)
                                        for tmp in binary_gt_labels_val]
                binary_gt_labels_val = [np.expand_dims(tmp, axis=-1) for tmp in binary_gt_labels_val]
                instance_gt_labels_val = [cv2.resize(tmp,
                                                     dsize=(CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT),
                                                     dst=tmp,
                                                     interpolation=cv2.INTER_NEAREST)
                                          for tmp in instance_gt_labels_val]
            phase_val = 'test'

            t_start_val = time.time()
            c_val, val_summary, val_accuracy, val_binary_seg_loss, val_instance_seg_loss = \
                sess.run([total_loss, val_merge_summary_op, accuracy, binary_seg_loss, disc_loss],
                         feed_dict={input_tensor: gt_imgs_val,
                                    binary_label_tensor: binary_gt_labels_val,
                                    instance_label_tensor: instance_gt_labels_val,
                                    phase: phase_val})

            if epoch % 100 == 0:
                cv2.imwrite('test_image.png', gt_imgs_val[0] + VGG_MEAN)

            summary_writer.add_summary(val_summary, global_step=epoch)

            cost_time_val = time.time() - t_start_val
            val_cost_time_mean.append(cost_time_val)

            if epoch % CFG.TRAIN.DISPLAY_STEP == 0:
                log.info('Epoch: {:d} total_loss= {:6f} binary_seg_loss= {:6f} instance_seg_loss= {:6f} accuracy= {:6f}'
                         ' mean_cost_time= {:5f}s '.
                         format(epoch + 1, c, binary_loss, instance_loss, train_accuracy,
                                np.mean(train_cost_time_mean)))
                train_cost_time_mean.clear()

            if epoch % CFG.TRAIN.TEST_DISPLAY_STEP == 0:
                log.info('Epoch_Val: {:d} total_loss= {:6f} binary_seg_loss= {:6f} '
                         'instance_seg_loss= {:6f} accuracy= {:6f} '
                         'mean_cost_time= {:5f}s '.
                         format(epoch + 1, c_val, val_binary_seg_loss, val_instance_seg_loss, val_accuracy,
                                np.mean(val_cost_time_mean)))
                val_cost_time_mean.clear()

            if epoch % 2000 == 0:
                saver.save(sess=sess, save_path=model_save_path, global_step=epoch)
    sess.close()

    return

def test_lanenet(image_path, weights_path, use_gpu):
    """

    :param image_path:
    :param weights_path:
    :param use_gpu:
    :return:
    """
    assert ops.exists(image_path), '{:s} not exist'.format(image_path)

    log.info('开始读取图像数据并进行预处理')
    t_start = time.time()
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image_vis = image
    image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
    image = image - VGG_MEAN
    log.info('图像读取完毕, 耗时: {:.5f}s'.format(time.time() - t_start))

    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')
    phase_tensor = tf.constant('test', tf.string)

    net = lanenet_merge_model.LaneNet(phase=phase_tensor, net_flag='vgg')
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='lanenet_model')

    cluster = lanenet_cluster.LaneNetCluster()
    postprocessor = lanenet_postprocess.LaneNetPoseProcessor()

    saver = tf.train.Saver()

    # Set sess configuration
    if use_gpu:
        sess_config = tf.ConfigProto(device_count={'GPU': 1})
    else:
        sess_config = tf.ConfigProto(device_count={'CPU': 0})
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    with sess.as_default():

        saver.restore(sess=sess, save_path=weights_path)

        t_start = time.time()
        binary_seg_image, instance_seg_image = sess.run([binary_seg_ret, instance_seg_ret],
                                                        feed_dict={input_tensor: [image]})
        t_cost = time.time() - t_start
        log.info('单张图像车道线预测耗时: {:.5f}s'.format(t_cost))

        binary_seg_image[0] = postprocessor.postprocess(binary_seg_image[0])
        mask_image = cluster.get_lane_mask(binary_seg_ret=binary_seg_image[0],
                                           instance_seg_ret=instance_seg_image[0])

        for i in range(4):
            instance_seg_image[0][:, :, i] = minmax_scale(instance_seg_image[0][:, :, i])
        embedding_image = np.array(instance_seg_image[0], np.uint8)

        plt.figure('mask_image')
        plt.imshow(mask_image[:, :, (2, 1, 0)])
        plt.figure('src_image')
        plt.imshow(image_vis[:, :, (2, 1, 0)])
        plt.figure('instance_image')
        plt.imshow(embedding_image[:, :, (2, 1, 0)])
        plt.figure('binary_image')
        plt.imshow(binary_seg_image[0] * 255, cmap='gray')
        plt.show()

    sess.close()

    return


def test_lanenet_batch(image_dir, weights_path, batch_size, use_gpu, save_dir=None):
    """

    :param image_dir:
    :param weights_path:
    :param batch_size:
    :param use_gpu:
    :param save_dir:
    :return:
    """
    assert ops.exists(image_dir), '{:s} not exist'.format(image_dir)

    log.info('开始获取图像文件路径...')
    image_path_list = glob.glob('{:s}/**/*.jpg'.format(image_dir), recursive=True) + \
                      glob.glob('{:s}/**/*.png'.format(image_dir), recursive=True) + \
                      glob.glob('{:s}/**/*.jpeg'.format(image_dir), recursive=True)

    input_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 256, 512, 3], name='input_tensor')
    phase_tensor = tf.constant('test', tf.string)

    net = lanenet_merge_model.LaneNet(phase=phase_tensor, net_flag='vgg')
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='lanenet_model')

    cluster = lanenet_cluster.LaneNetCluster()
    postprocessor = lanenet_postprocess.LaneNetPoseProcessor()

    saver = tf.train.Saver()

    # Set sess configuration
    if use_gpu:
        sess_config = tf.ConfigProto(device_count={'GPU': 1})
    else:
        sess_config = tf.ConfigProto(device_count={'GPU': 0})
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    with sess.as_default():

        saver.restore(sess=sess, save_path=weights_path)

        epoch_nums = int(math.ceil(len(image_path_list) / batch_size))

        for epoch in range(epoch_nums):
            log.info('[Epoch:{:d}] 开始图像读取和预处理...'.format(epoch))
            t_start = time.time()
            image_path_epoch = image_path_list[epoch * batch_size:(epoch + 1) * batch_size]
            image_list_epoch = [cv2.imread(tmp, cv2.IMREAD_COLOR) for tmp in image_path_epoch]
            image_vis_list = image_list_epoch
            image_list_epoch = [cv2.resize(tmp, (512, 256), interpolation=cv2.INTER_LINEAR)
                                for tmp in image_list_epoch]
            image_list_epoch = [tmp - VGG_MEAN for tmp in image_list_epoch]
            t_cost = time.time() - t_start
            log.info('[Epoch:{:d}] 预处理{:d}张图像, 共耗时: {:.5f}s, 平均每张耗时: {:.5f}'.format(
                epoch, len(image_path_epoch), t_cost, t_cost / len(image_path_epoch)))

            t_start = time.time()
            binary_seg_images, instance_seg_images = sess.run(
                [binary_seg_ret, instance_seg_ret], feed_dict={input_tensor: image_list_epoch})
            t_cost = time.time() - t_start
            log.info('[Epoch:{:d}] 预测{:d}张图像车道线, 共耗时: {:.5f}s, 平均每张耗时: {:.5f}s'.format(
                epoch, len(image_path_epoch), t_cost, t_cost / len(image_path_epoch)))

            cluster_time = []
            for index, binary_seg_image in enumerate(binary_seg_images):
                t_start = time.time()
                binary_seg_image = postprocessor.postprocess(binary_seg_image)
                mask_image = cluster.get_lane_mask(binary_seg_ret=binary_seg_image,
                                                   instance_seg_ret=instance_seg_images[index])
                cluster_time.append(time.time() - t_start)
                mask_image = cv2.resize(mask_image, (image_vis_list[index].shape[1],
                                                     image_vis_list[index].shape[0]),
                                        interpolation=cv2.INTER_LINEAR)

                if save_dir is None:
                    plt.ion()
                    plt.figure('mask_image')
                    plt.imshow(mask_image[:, :, (2, 1, 0)])
                    plt.figure('src_image')
                    plt.imshow(image_vis_list[index][:, :, (2, 1, 0)])
                    plt.pause(3.0)
                    plt.show()
                    plt.ioff()

                if save_dir is not None:
                    mask_image = cv2.addWeighted(image_vis_list[index], 1.0, mask_image, 1.0, 0)
                    image_name = ops.split(image_path_epoch[index])[1]
                    image_save_path = ops.join(save_dir, image_name)
                    cv2.imwrite(image_save_path, mask_image)

            log.info('[Epoch:{:d}] 进行{:d}张图像车道线聚类, 共耗时: {:.5f}s, 平均每张耗时: {:.5f}'.format(
                epoch, len(image_path_epoch), np.sum(cluster_time), np.mean(cluster_time)))

    sess.close()

    return


if __name__ == '__main__':
    # init args
    args = init_args()
    if args.train:
        # train lanenet
        train_net(args.dataset_dir, args.weights_path, net_flag=args.net, job_dir=args.checkpoint_path)
    else:
        if args.save_dir is not None and not ops.exists(args.save_dir):
            log.error('{:s} not exist and has been made'.format(args.save_dir))
            os.makedirs(args.save_dir)

        if args.is_batch.lower() == 'false':
            # test hnet model on single image
            test_lanenet(args.image_path, args.weights_path, args.use_gpu)
        else:
            # test hnet model on a batch of image
            test_lanenet_batch(image_dir=args.image_path, weights_path=args.weights_path,
                               save_dir=args.save_dir, use_gpu=args.use_gpu, batch_size=args.batch_size)
