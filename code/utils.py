import os
import numpy as np
import imageio

def create_dirs(dirs):
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)

def quantize(img, min_val=0, max_val=255):
    return np.clip(np.round(img), min_val, max_val)

def cal_psnr(out_img, tar_img, scale):
    out_img = quantize(np.squeeze(out_img))
    tar_img = quantize(np.squeeze(tar_img))
    diff = out_img - tar_img
    shave = 6 + scale
    diff = diff[shave:-shave, shave:-shave]
    mse = np.mean(np.power(diff, 2))
    psnr = 10.0 * np.log10(255.0**2/mse)
    return psnr

def save_img(img, dir):
    img = np.uint8(quantize(np.squeeze(img)))
    imageio.imwrite(dir, img)

def mod_crop(img, scale):
    h, w, c = img.shape
    img = img[0:h-np.mod(h, scale), 0:w-np.mod(w, scale)]
    return img

def chop_forward(x, sess, model, scale, shave=10):
    print("predicting...")
    h, w, c = x.shape
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    input1 = x[0:h_size, 0:w_size]
    input2 = x[0:h_size, (w - w_size):w]
    input3 = x[(h - h_size):h, 0:w_size]
    input4 = x[(h - h_size):h, (w - w_size):w]
    tmp1 = sess.run(model.output, feed_dict={model.input:[input1]})[0]
    tmp2 = sess.run(model.output, feed_dict={model.input:[input2]})[0]
    tmp3 = sess.run(model.output, feed_dict={model.input:[input3]})[0]
    tmp4 = sess.run(model.output, feed_dict={model.input:[input4]})[0]
    tmp_image = np.zeros([x.shape[0]*scale, x.shape[1]*scale, 3])
    h, w = h * scale, w * scale
    h_half, w_half = h_half * scale, w_half * scale
    h_size, w_size = h_size * scale, w_size * scale
    tmp_image[0:h_half, 0:w_half] = tmp1[0:h_half, 0:w_half]
    tmp_image[0:h_half, w_half:] = tmp2[0:h_half, (w_size - w + w_half):w_size]
    tmp_image[h_half:, 0:w_half] = tmp3[(h_size - h + h_half):h_size, 0:w_half]
    tmp_image[h_half:, w_half:] = tmp4[(h_size - h + h_half):h_size, (w_size - w + w_half):w_size]
    return tmp_image