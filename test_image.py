'''
    test camera 

Author: Zhengwei Li
Date  : 2018/12/28
'''
import time
import cv2
import torch 
import argparse
import numpy as np
import os 
import torch.nn.functional as F
import streamlit as st
import webbrowser
import codecs
import PIL
from PIL import Image
# parser = argparse.ArgumentParser(description='human matting')
# parser.add_argument('--model', default='./', help='preTrained model')
# parser.add_argument('--size', type=int, default=256, help='input size')
# parser.add_argument('--without_gpu', action='store_true', default=False, help='no use gpu')

# args = parser.parse_args()

torch.set_grad_enabled(False)

    
#################################
#----------------
# if args.without_gpu:
#     print("use CPU !")
#     device = torch.device('cpu')
# else:
#     if torch.cuda.is_available():
#         n_gpu = torch.cuda.device_count()
#         print("----------------------------------------------------------")
#         print("|       use GPU !      ||   Available GPU number is {} !  |".format(n_gpu))
#         print("----------------------------------------------------------")

device = torch.device('cuda:0')

#################################
#---------------
myModel = torch.load("ckpt/human_matting/model/model_obj.pth", map_location=device)
myModel.eval()
myModel.to(device)


def seg_processfg(image, net):

    # opencv
    origin_h, origin_w, c = image.shape
    image_resize = cv2.resize(image, (args.size,args.size), interpolation=cv2.INTER_CUBIC)
    image_resize = (image_resize - (104., 112., 121.,)) / 255.0



    tensor_4D = torch.FloatTensor(1, 3, args.size, args.size)
    
    tensor_4D[0,:,:,:] = torch.FloatTensor(image_resize.transpose(2,0,1))
    inputs = tensor_4D.to(device)

    t0 = time.time()

    trimap, alpha = net(inputs)

    print((time.time() - t0))  

    alpha_np = alpha[0,0,:,:].cpu().data.numpy()


    alpha_np = cv2.resize(alpha_np, (origin_w, origin_h), interpolation=cv2.INTER_CUBIC)

    fg = np.multiply(alpha_np[..., np.newaxis], image)

    bg = image
    bg_gray = np.multiply(1-alpha_np[..., np.newaxis], image)
    bg_gray = cv2.cvtColor(bg_gray, cv2.COLOR_BGR2GRAY)

    bg[:,:,0] = bg_gray
    bg[:,:,1] = bg_gray
    bg[:,:,2] = bg_gray

    # fg[fg<=0] = 0
    # fg[fg>255] = 255
    # fg = fg.astype(np.uint8)
    # out = cv2.addWeighted(fg, 0.7, bg, 0.3, 0)
    out = fg + bg
    out[out<0] = 0
    out[out>255] = 255
    out = out.astype(np.uint8)

    return fg


def camera_seg(net):

    # img = cv2.imread("/home/ubuntu/Desktop/test.jpg")
    st.title('Segmentation')
    st.subheader('WEb based segmentation' )


    st.set_option('deprecation.showfileUploaderEncoding', False)
  
    img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if img_file_buffer is not None:



        img = Image.open(img_file_buffer).convert('RGB') 
        open_cv_image = numpy.array(pil_image)
        open_cv_image1 = open_cv_image[:, :, ::-1].copy()
    
        # get a frame
    
    #frame = cv2.flip(img,1)
    #frame_seg = seg_process(open_cv_image1, net)
    framefg = seg_processfg(open_cv_image1,net)


    st.image(
        framefg,channels="RGB",
        caption=f"You amazing image has shape {framefg.shape[0:2]} ",
        use_column_width=True,
    )
    
        # show a frame
        #cv2.imshow("capture", frame_seg)
    # cv2.imshow("capture", framefg)
    # cv2.waitKey(0)


def main():
    camera_seg(myModel)


main()

