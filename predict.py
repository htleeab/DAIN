'''
If video enabled, convert video to double fps
If video disabled, run slow motion on frame 10 and 11 in MiddleBury dataset
'''

import time
import os
import shutil
import random
import numpy as np
import cv2
import torch
from torch.autograd import Variable
from scipy.misc import imread, imsave
import tqdm
import networks
from AverageMeter import *
from my_args import get_DAIN_parser

torch.backends.cudnn.benchmark=True # Optimize the network for speed up

MB_Other_DATA = "./MiddleBurySet/other-data/"
MB_Other_RESULT = "./MiddleBurySet/other-result-author/"

def parse_args():
    parser = get_DAIN_parser()
    # Video options
    parser.add_argument('-v', '--video', dest='gen_video', action='store_true', help='Generate video, otherwise process frame 10 and 11 in MiddleBury dataset')
    parser.add_argument('--video_input', dest='video_input_filepath', type=str, help='video_input')
    parser.add_argument('--video_output_dir', dest='video_output_dir', default='video_output', type=str, help='video output folder')
    args = parser.parse_args()
    if args.gen_video:
        assert args.video_input_filepath is not None and os.path.exists(args.video_input_filepath), f'video input {args.video_input_filepath} is not exist'
        os.makedirs(args.video_output_dir, exist_ok=True)
    return args

def load_model(args):
    model = networks.__dict__[args.netName](    channel=args.channels,
                                    filter_size = args.filter_size ,
                                    timestep=args.time_step,
                                    training=False)

    if args.use_cuda:
        model = model.cuda()

    args.SAVED_MODEL = './model_weights/best.pth'
    if os.path.exists(args.SAVED_MODEL):
        print("The testing model weight is: " + args.SAVED_MODEL)
        if not args.use_cuda:
            pretrained_dict = torch.load(args.SAVED_MODEL, map_location=lambda storage, loc: storage)
        else:
            pretrained_dict = torch.load(args.SAVED_MODEL)

        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
        # 4. release the pretrained dict for saving memory
        pretrained_dict = []
    else:
        print("*****************************************************************")
        print("**** We don't load any trained weights **************************")
        print("*****************************************************************")

    model = model.eval() # deploy mode
    return model

def frame_interpolation_2_frames(model, img_first, img_second, use_cuda, save_which, dtype):
    start_time = time.time()

    # initialize tensor of input & output
    X0 =  torch.from_numpy( np.transpose(img_first , (2,0,1)).astype("float32")/ 255.0).type(dtype)
    X1 =  torch.from_numpy( np.transpose(img_second , (2,0,1)).astype("float32")/ 255.0).type(dtype)
    assert (X0.size(1) == X1.size(1)) and (X0.size(2) == X1.size(2)), f'Input size do not match.'
    y_ = torch.FloatTensor()

    intWidth = X0.size(2)
    intHeight = X0.size(1)
    channel = X0.size(0)
    if not channel == 3:
        return None, None

    if intWidth != ((intWidth >> 7) << 7):
        intWidth_pad = (((intWidth >> 7) + 1) << 7)  # more than necessary
        intPaddingLeft =int(( intWidth_pad - intWidth)/2)
        intPaddingRight = intWidth_pad - intWidth - intPaddingLeft
    else:
        intWidth_pad = intWidth
        intPaddingLeft = 32
        intPaddingRight= 32

    if intHeight != ((intHeight >> 7) << 7):
        intHeight_pad = (((intHeight >> 7) + 1) << 7)  # more than necessary
        intPaddingTop = int((intHeight_pad - intHeight) / 2)
        intPaddingBottom = intHeight_pad - intHeight - intPaddingTop
    else:
        intHeight_pad = intHeight
        intPaddingTop = 32
        intPaddingBottom = 32

    pader = torch.nn.ReplicationPad2d([intPaddingLeft, intPaddingRight , intPaddingTop, intPaddingBottom])

    torch.set_grad_enabled(False)
    X0 = Variable(torch.unsqueeze(X0,0))
    X1 = Variable(torch.unsqueeze(X1,0))
    X0 = pader(X0)
    X1 = pader(X1)

    if use_cuda:
        X0 = X0.cuda()
        X1 = X1.cuda()
    proc_end = time.time()

    y_s, offset, filter = model(torch.stack((X0, X1),dim = 0))
    y_ = y_s[save_which]

    print(f'*****************current image process time \t {time.time()-proc_end}s \t {time.time()-start_time}s******************')
    if use_cuda:
        X0 = X0.data.cpu().numpy()
        if not isinstance(y_, list):
            y_ = y_.data.cpu().numpy()
        else:
            y_ = [item.data.cpu().numpy() for item in y_]
        offset = [offset_i.data.cpu().numpy() for offset_i in offset]
        filter = [filter_i.data.cpu().numpy() for filter_i in filter]  if filter[0] is not None else None
        X1 = X1.data.cpu().numpy()
    else:
        X0 = X0.data.numpy()
        if not isinstance(y_, list):
            y_ = y_.data.numpy()
        else:
            y_ = [item.data.numpy() for item in y_]
        offset = [offset_i.data.numpy() for offset_i in offset]
        filter = [filter_i.data.numpy() for filter_i in filter]
        X1 = X1.data.numpy()

    X0 = np.transpose(255.0 * X0.clip(0,1.0)[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0))
    y_ = [np.transpose(255.0 * item.clip(0,1.0)[0, :, intPaddingTop:intPaddingTop+intHeight,
                              intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0)) for item in y_]
    offset = [np.transpose(offset_i[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0)) for offset_i in offset]
    filter = [np.transpose(
        filter_i[0, :, intPaddingTop:intPaddingTop + intHeight, intPaddingLeft: intPaddingLeft + intWidth],
        (1, 2, 0)) for filter_i in filter] if filter is not None else None
    X1 = np.transpose(255.0 * X1.clip(0,1.0)[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0))
    return y_

def process_video(model, video_filepath, args):
    # get video input and media info
    video_in = cv2.VideoCapture(video_filepath)
    fps = video_in.get(cv2.CAP_PROP_FPS)
    width = int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frame_num = int(video_in.get(cv2.CAP_PROP_FRAME_COUNT))

    out_fps = fps/args.time_step
    output_video_filepath = os.path.join(args.video_output_dir, os.path.splitext(os.path.basename(video_filepath))[0]+f'_DAIN_{out_fps}p.mp4')
    video_out = cv2.VideoWriter(output_video_filepath, cv2.VideoWriter_fourcc(*'MP4V'), out_fps, (width, height))
    print(f'process {video_filepath} to {output_video_filepath}, {fps}fps to {out_fps}fps, resolution: {width}x{height}')

    pre_frame = None
    for cur_frame_num in tqdm.tqdm(range(total_frame_num)):
        try:
            res, frame = video_in.read()
            frame = frame[..., (2,1,0)] # BGR to RGB
            if pre_frame is not None:
                y_ = frame_interpolation_2_frames(model, pre_frame, frame, args.use_cuda, args.save_which, args.dtype)
                assert y_ is not None
                for out_frame in y_:
                    video_out.write(np.round(out_frame).astype(np.uint8)[..., (2,1,0)])
            pre_frame = frame
            video_out.write(frame[..., (2,1,0)]) # BGR to RGB back
        except KeyboardInterrupt:
            break
    video_in.release()
    video_out.release()

if __name__ == '__main__':
    args = parse_args()
    model = load_model(args)


    DO_MiddleBurryOther = not args.gen_video
    if DO_MiddleBurryOther:
        if not os.path.exists(MB_Other_RESULT):
            os.mkdir(MB_Other_RESULT)
        unique_id =str(random.randint(0, 100000))
        print("The unique id for current testing is: " + str(unique_id))
        gen_dir = os.path.join(MB_Other_RESULT, unique_id)
        os.mkdir(gen_dir)

        for cur_dir in os.listdir(MB_Other_DATA):
            print(cur_dir)
            os.mkdir(os.path.join(gen_dir, cur_dir))
            arguments_strFirst = os.path.join(MB_Other_DATA, cur_dir, "frame10.png")
            arguments_strSecond = os.path.join(MB_Other_DATA, cur_dir, "frame11.png")
            img_first = imread(arguments_strFirst)
            img_second = imread(arguments_strSecond)
            y_ = frame_interpolation_2_frames(model, img_first, img_second, args.use_cuda, args.save_which, args.dtype)
            if y_ is None:
                continue

            count = 0
            shutil.copy(arguments_strFirst, os.path.join(gen_dir, cur_dir, "{:0>4d}.png".format(count)))
            count  = count+1
            for item in y_:
                arguments_strOut = os.path.join(gen_dir, cur_dir, "{:0>4d}.png".format(count))
                count = count + 1
                imsave(arguments_strOut, np.round(item).astype(np.uint8))
            shutil.copy(arguments_strSecond, os.path.join(gen_dir, cur_dir, "{:0>4d}.png".format(count)))
            count = count + 1
    else:
        process_video(model, args.video_input_filepath, args)
