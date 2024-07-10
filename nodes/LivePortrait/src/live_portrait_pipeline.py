# coding: utf-8

"""
Pipeline of LivePortrait
"""

# TODO:
# 1. 当前假定所有的模板都是已经裁好的，需要修改下
# 2. pick样例图 source + driving

import cv2
import numpy as np
import pickle,os
import os.path as osp
# from rich.progress import track

# from .config.argument_config import ArgumentConfig
from .config.inference_config import InferenceConfig
from .config.crop_config import CropConfig
from .utils.cropper import Cropper
from .utils.camera import get_rotation_matrix
from .utils.video import images2video, concat_frames
from .utils.crop import _transform_img
from .utils.retargeting_utils import calc_lip_close_ratio
from .utils.io import load_image_rgb, load_driving_info
from .utils.helper import mkdir, basename, dct2cuda, is_video, is_template, resize_to_limit
from .utils.rprint import rlog as log
from .live_portrait_wrapper import LivePortraitWrapper

import comfy.utils

def add_index_to_filename(output_path, index):
    directory, filename = osp.split(output_path)
    basename, ext = osp.splitext(filename)
    new_filename = f"{basename}_{index}{ext}"
    new_output_path = osp.join(directory, new_filename)
    return new_output_path


# 创建固定长度的list，不足的填充
def create_drivings(elements, max_count, revert=False):
    if not revert:
        if max_count <= len(elements):
            return elements[:max_count]
        elif len(elements)>0:
            return [elements[i % len(elements)] for i in range(max_count)]
        else:
            return [None for i in range(max_count)]
    else:
        if len(elements)==0:
            return [None for i in range(max_count)]
        extended_frames = elements + elements[-2:0:-1]  # 正向加反向中间部分
        if max_count <= len(extended_frames):
            return extended_frames[:max_count]
        else:
            return [extended_frames[i % len(extended_frames)] for i in range(max_count)]
       


def make_abs_path(fn):
    return osp.join(osp.dirname(osp.realpath(__file__)), fn)


class LivePortraitPipeline(object):

    def __init__(self, inference_cfg: InferenceConfig, crop_cfg: CropConfig,landmark_runner_ckpt,insightface_pretrained_weights):
        self.live_portrait_wrapper: LivePortraitWrapper = LivePortraitWrapper(cfg=inference_cfg)
        self.cropper = Cropper(crop_cfg=crop_cfg,landmark_runner_ckpt=landmark_runner_ckpt,insightface_pretrained_weights=insightface_pretrained_weights)

    def execute(self, args):
        inference_cfg = self.live_portrait_wrapper.cfg # for convenience
        ######## process reference portrait ########
        # img_rgb = load_image_rgb(args.source_image)
        img_rgb=args.source_image
        # 增加人脸好的
        crop_info=args.crop_info

        args.driving_info=args.driving_info[0]

        img_rgb = resize_to_limit(img_rgb, inference_cfg.ref_max_shape, inference_cfg.ref_shape_n)
        # log(f"Load source image from {args.source_image}")
        # todo 人脸检测并裁切 - 独立一个节点
        crop_info = self.cropper.crop_single_image(img_rgb,src_face=crop_info)
            
        source_lmk = crop_info['lmk_crop']
        img_crop, img_crop_256x256 = crop_info['img_crop'], crop_info['img_crop_256x256']
        if inference_cfg.flag_do_crop:
            I_s = self.live_portrait_wrapper.prepare_source(img_crop_256x256)
        else:
            I_s = self.live_portrait_wrapper.prepare_source(img_rgb)
        x_s_info = self.live_portrait_wrapper.get_kp_info(I_s)
        x_c_s = x_s_info['kp']
        R_s = get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])
        f_s = self.live_portrait_wrapper.extract_feature_3d(I_s)
        x_s = self.live_portrait_wrapper.transform_keypoint(x_s_info)

        if inference_cfg.flag_lip_zero:
            # let lip-open scalar to be 0 at first
            c_d_lip_before_animation = [0.]
            combined_lip_ratio_tensor_before_animation = self.live_portrait_wrapper.calc_combined_lip_ratio(c_d_lip_before_animation, source_lmk)
            if combined_lip_ratio_tensor_before_animation[0][0] < inference_cfg.lip_zero_threshold:
                inference_cfg.flag_lip_zero = False
            else:
                lip_delta_before_animation = self.live_portrait_wrapper.retarget_lip(x_s, combined_lip_ratio_tensor_before_animation)
        ############################################

        ######## process driving info ########
        if is_video(args.driving_info):
            log(f"Load from video file (mp4 mov avi etc...): {args.driving_info}")
            # TODO: 这里track一下驱动视频 -> 构建模板
            driving_rgb_lst = load_driving_info(args.driving_info)
            driving_rgb_lst_256 = [cv2.resize(_, (256, 256)) for _ in driving_rgb_lst]
            I_d_lst = self.live_portrait_wrapper.prepare_driving_videos(driving_rgb_lst_256)
            n_frames = I_d_lst.shape[0]
            if inference_cfg.flag_eye_retargeting or inference_cfg.flag_lip_retargeting:
                driving_lmk_lst = self.cropper.get_retargeting_lmk_info(driving_rgb_lst)
                input_eye_ratio_lst, input_lip_ratio_lst = self.live_portrait_wrapper.calc_retargeting_ratio(source_lmk, driving_lmk_lst)
        elif is_template(args.driving_info):
            log(f"Load from video templates {args.driving_info}")
            with open(args.driving_info, 'rb') as f:
                template_lst, driving_lmk_lst = pickle.load(f)
            n_frames = template_lst[0]['n_frames']
            input_eye_ratio_lst, input_lip_ratio_lst = self.live_portrait_wrapper.calc_retargeting_ratio(source_lmk, driving_lmk_lst)
        else:
            raise Exception("Unsupported driving types!")
        #########################################

        ######## prepare for pasteback ########
        if inference_cfg.flag_pasteback:
            if inference_cfg.mask_crop is None:
                inference_cfg.mask_crop = cv2.imread(make_abs_path('./utils/resources/mask_template.png'), cv2.IMREAD_COLOR)
            mask_ori = _transform_img(inference_cfg.mask_crop, crop_info['M_c2o'], dsize=(img_rgb.shape[1], img_rgb.shape[0]))
            mask_ori = mask_ori.astype(np.float32) / 255.
            I_p_paste_lst = []
        #########################################

        I_p_lst = []
        R_d_0, x_d_0_info = None, None

        pbar = comfy.utils.ProgressBar(n_frames)
        print('Animating...',  n_frames)
        for i in range(n_frames):
        # track(range(n_frames), description='Animating...', total=n_frames):
    
            if is_video(args.driving_info):
                # extract kp info by M
                I_d_i = I_d_lst[i]
                x_d_i_info = self.live_portrait_wrapper.get_kp_info(I_d_i)
                R_d_i = get_rotation_matrix(x_d_i_info['pitch'], x_d_i_info['yaw'], x_d_i_info['roll'])
            else:
                # from template
                x_d_i_info = template_lst[i]
                x_d_i_info = dct2cuda(x_d_i_info, inference_cfg.device_id)
                R_d_i = x_d_i_info['R_d']

            if i == 0:
                R_d_0 = R_d_i
                x_d_0_info = x_d_i_info

            if inference_cfg.flag_relative:
                R_new = (R_d_i @ R_d_0.permute(0, 2, 1)) @ R_s
                delta_new = x_s_info['exp'] + (x_d_i_info['exp'] - x_d_0_info['exp'])
                scale_new = x_s_info['scale'] * (x_d_i_info['scale'] / x_d_0_info['scale'])
                t_new = x_s_info['t'] + (x_d_i_info['t'] - x_d_0_info['t'])
            else:
                R_new = R_d_i
                delta_new = x_d_i_info['exp']
                scale_new = x_s_info['scale']
                t_new = x_d_i_info['t']

            t_new[..., 2].fill_(0) # zero tz
            x_d_i_new = scale_new * (x_c_s @ R_new + delta_new) + t_new

            # Algorithm 1:
            if not inference_cfg.flag_stitching and not inference_cfg.flag_eye_retargeting and not inference_cfg.flag_lip_retargeting:
                # without stitching or retargeting
                if inference_cfg.flag_lip_zero:
                    x_d_i_new += lip_delta_before_animation.reshape(-1, x_s.shape[1], 3)
                else:
                    pass
            elif inference_cfg.flag_stitching and not inference_cfg.flag_eye_retargeting and not inference_cfg.flag_lip_retargeting:
                # with stitching and without retargeting
                if inference_cfg.flag_lip_zero:
                    x_d_i_new = self.live_portrait_wrapper.stitching(x_s, x_d_i_new) + lip_delta_before_animation.reshape(-1, x_s.shape[1], 3)
                else:
                    x_d_i_new = self.live_portrait_wrapper.stitching(x_s, x_d_i_new)
            else:
                eyes_delta, lip_delta = None, None
                if inference_cfg.flag_eye_retargeting:
                    c_d_eyes_i = input_eye_ratio_lst[i]
                    combined_eye_ratio_tensor = self.live_portrait_wrapper.calc_combined_eye_ratio(c_d_eyes_i, source_lmk)
                    # ∆_eyes,i = R_eyes(x_s; c_s,eyes, c_d,eyes,i)
                    eyes_delta = self.live_portrait_wrapper.retarget_eye(x_s, combined_eye_ratio_tensor)
                if inference_cfg.flag_lip_retargeting:
                    c_d_lip_i = input_lip_ratio_lst[i]
                    combined_lip_ratio_tensor = self.live_portrait_wrapper.calc_combined_lip_ratio(c_d_lip_i, source_lmk)
                    # ∆_lip,i = R_lip(x_s; c_s,lip, c_d,lip,i)
                    lip_delta = self.live_portrait_wrapper.retarget_lip(x_s, combined_lip_ratio_tensor)

                if inference_cfg.flag_relative:  # use x_s
                    x_d_i_new = x_s + \
                        (eyes_delta.reshape(-1, x_s.shape[1], 3) if eyes_delta is not None else 0) + \
                        (lip_delta.reshape(-1, x_s.shape[1], 3) if lip_delta is not None else 0)
                else:  # use x_d,i
                    x_d_i_new = x_d_i_new + \
                        (eyes_delta.reshape(-1, x_s.shape[1], 3) if eyes_delta is not None else 0) + \
                        (lip_delta.reshape(-1, x_s.shape[1], 3) if lip_delta is not None else 0)

                if inference_cfg.flag_stitching:
                    x_d_i_new = self.live_portrait_wrapper.stitching(x_s, x_d_i_new)

            out = self.live_portrait_wrapper.warp_decode(f_s, x_s, x_d_i_new)
            I_p_i = self.live_portrait_wrapper.parse_output(out['out'])[0]
            I_p_lst.append(I_p_i)

            if inference_cfg.flag_pasteback:
                I_p_i_to_ori = _transform_img(I_p_i, crop_info['M_c2o'], dsize=(img_rgb.shape[1], img_rgb.shape[0]))
                I_p_i_to_ori_blend = np.clip(mask_ori * I_p_i_to_ori + (1 - mask_ori) * img_rgb, 0, 255).astype(np.uint8)
                out = np.hstack([I_p_i_to_ori, I_p_i_to_ori_blend])
                I_p_paste_lst.append(I_p_i_to_ori_blend)

            pbar.update(1)


        directory, filename = os.path.split(args.output_path)
        if not os.path.exists(directory):
            mkdir(directory)
        wfp_concat = args.output_path_concat
        video_fps = cv2.VideoCapture(args.driving_info).get(cv2.CAP_PROP_FPS)
        if is_video(args.driving_info):
            frames_concatenated = concat_frames(I_p_lst, driving_rgb_lst, img_crop_256x256)
            # save (driving frames, source image, drived frames) result
            # wfp_concat = osp.join(directory, f'{basename(args.source_image)}--{basename(args.driving_info)}_concat.mp4')
            # images2video(frames_concatenated, wfp=wfp_concat)
            images2video(frames_concatenated, wfp=wfp_concat, fps=video_fps)

        # save drived result
        wfp = args.output_path
        if inference_cfg.flag_pasteback:
            images2video(I_p_paste_lst, wfp=wfp, fps=video_fps)
        else:
            images2video(I_p_lst, wfp=wfp, fps=video_fps)

        return wfp, wfp_concat

    def executeForAll(self, args):
        inference_cfg = self.live_portrait_wrapper.cfg  # for convenience
        ######## process reference portrait ########
        # img_rgb = load_image_rgb(args.source_image)
        img_rgb = args.source_image
        # 增加人脸好的
        crop_info_list = args.crop_info
        # eye lip
        __eye__s=[c[0]['__eye__'] for c in crop_info_list]
        __lip__s=[c[0]['__lip__'] for c in crop_info_list]

        # 对齐多个驱动视频的长度
        align_mode=args.align_mode

        img_rgb = resize_to_limit(img_rgb, inference_cfg.ref_max_shape, inference_cfg.ref_shape_n)
        # log(f"Load source image from {args.source_image}")
        # todo 人脸检测并裁切 - 独立一个节点
        crop_info_list = [self.cropper.crop_single_image(img_rgb, src_face=crop_info) for crop_info in crop_info_list]

        video_fps = cv2.VideoCapture(args.driving_info[0]).get(cv2.CAP_PROP_FPS)

        driving_infos=args.driving_info

        ######## process driving info ########
        driving_lmk_lst_s=[]
        n_frames_s=[]
        I_d_lst_s=[]

        pbar = comfy.utils.ProgressBar(len(driving_infos))
        for  z in range(len(driving_infos)):

            driving_info=driving_infos[z]
            crop_info=crop_info_list[z]
            # print('###',z,len(driving_infos),len(crop_info_list))
            # print('#crop_info_list[z]',crop_info_list[z])
            __eye__=__eye__s[z]
            __lip__=__lip__s[z]

            if is_video(driving_info):
                log(f"Load from video file (mp4 mov avi etc...): {driving_info}")
                # TODO: 这里track一下驱动视频 -> 构建模板
                driving_rgb_lst = load_driving_info(driving_info)
                driving_rgb_lst_256 = [cv2.resize(_, (256, 256)) for _ in driving_rgb_lst]
                I_d_lst = self.live_portrait_wrapper.prepare_driving_videos(driving_rgb_lst_256)
                n_frames = I_d_lst.shape[0]
                
                I_d_lst_s.append(I_d_lst)
                n_frames_s.append(n_frames)

                source_lmk = crop_info['lmk_crop']
                if __eye__ or __lip__:
                    driving_lmk_lst = self.cropper.get_retargeting_lmk_info(driving_rgb_lst)
                    driving_lmk_lst_s.append(driving_lmk_lst)
                    input_eye_ratio_lst, input_lip_ratio_lst = self.live_portrait_wrapper.calc_retargeting_ratio(
                        source_lmk, 
                        driving_lmk_lst
                        )
            # elif is_template(args.driving_info):
            #     log(f"Load from video templates {args.driving_info}")
            #     with open(args.driving_info, 'rb') as f:
            #         template_lst, driving_lmk_lst = pickle.load(f)
            #     n_frames = template_lst[0]['n_frames']
            #     # input_eye_ratio_lst, input_lip_ratio_lst = self.live_portrait_wrapper.calc_retargeting_ratio(source_lmk, driving_lmk_lst)
            # else:
            #     raise Exception("Unsupported driving types!")
            #########################################
            # print('#driving_lmk_lst',self.driving_lmk_lst)
            pbar.update(1)

        # 对齐
        max_n_frames = max(n_frames_s)
        n_frames_s=[max_n_frames for i in n_frames_s]
        driving_lmk_lst_s= [create_drivings(d,max_n_frames,align_mode) for d in driving_lmk_lst_s]
        I_d_lst_s= [create_drivings(i,max_n_frames,align_mode) for i in I_d_lst_s]


        # 原图片---视频帧
        img_rgbs=[img_rgb for i in range(n_frames_s[0])]
        

        for index in range(len(crop_info_list)):

            crop_info=crop_info_list[index]
            # 地一张脸

            __eye__=__eye__s[index]
            __lip__=__lip__s[index]
            
            print('#crop_info',crop_info.keys())
            source_lmk = crop_info['lmk_crop']
            img_crop, img_crop_256x256 = crop_info['img_crop'], crop_info['img_crop_256x256']
            if inference_cfg.flag_do_crop:
                I_s = self.live_portrait_wrapper.prepare_source(img_crop_256x256)
          
            x_s_info = self.live_portrait_wrapper.get_kp_info(I_s)
            x_c_s = x_s_info['kp']
            R_s = get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])
            f_s = self.live_portrait_wrapper.extract_feature_3d(I_s)
            x_s = self.live_portrait_wrapper.transform_keypoint(x_s_info)

            if inference_cfg.flag_lip_zero:
                # let lip-open scalar to be 0 at first
                c_d_lip_before_animation = [0.]
                combined_lip_ratio_tensor_before_animation = self.live_portrait_wrapper.calc_combined_lip_ratio(c_d_lip_before_animation, source_lmk)
                if combined_lip_ratio_tensor_before_animation[0][0] < inference_cfg.lip_zero_threshold:
                    inference_cfg.flag_lip_zero = False
                else:
                    lip_delta_before_animation = self.live_portrait_wrapper.retarget_lip(x_s, combined_lip_ratio_tensor_before_animation)
            ############################################

            # 多个驱动视频
            if 0 <= index < len(driving_lmk_lst_s):
                driving_lmk_lst=driving_lmk_lst_s[index]
                
                if driving_lmk_lst!=None:
                    input_eye_ratio_lst, input_lip_ratio_lst = self.live_portrait_wrapper.calc_retargeting_ratio(source_lmk, driving_lmk_lst)

            ######## prepare for pasteback ########
            if inference_cfg.flag_pasteback:
                if inference_cfg.mask_crop is None:
                    inference_cfg.mask_crop = cv2.imread(make_abs_path('./utils/resources/mask_template.png'), cv2.IMREAD_COLOR)
                mask_ori = _transform_img(inference_cfg.mask_crop, crop_info['M_c2o'], dsize=(img_rgb.shape[1], img_rgb.shape[0]))
                mask_ori = mask_ori.astype(np.float32) / 255.
                I_p_paste_lst = []
            #########################################

            I_p_lst = []
            R_d_0, x_d_0_info = None, None

            # 多个驱动视频
            n_frames=n_frames_s[index]
            driving_info=driving_infos[index]
            I_d_lst=I_d_lst_s[index]


            pbar = comfy.utils.ProgressBar(n_frames)

            print('Animating...',  n_frames)
            for i in range(n_frames):
            # track(range(n_frames), description='Animating...', total=n_frames):

                if is_video(driving_info):
                    # extract kp info by M
                    I_d_i = I_d_lst[i]
                    x_d_i_info = self.live_portrait_wrapper.get_kp_info(I_d_i)
                    R_d_i = get_rotation_matrix(x_d_i_info['pitch'], x_d_i_info['yaw'], x_d_i_info['roll'])

                if i == 0:
                    R_d_0 = R_d_i
                    x_d_0_info = x_d_i_info

                if inference_cfg.flag_relative:
                    R_new = (R_d_i @ R_d_0.permute(0, 2, 1)) @ R_s
                    delta_new = x_s_info['exp'] + (x_d_i_info['exp'] - x_d_0_info['exp'])
                    scale_new = x_s_info['scale'] * (x_d_i_info['scale'] / x_d_0_info['scale'])
                    t_new = x_s_info['t'] + (x_d_i_info['t'] - x_d_0_info['t'])
                else:
                    R_new = R_d_i
                    delta_new = x_d_i_info['exp']
                    scale_new = x_s_info['scale']
                    t_new = x_d_i_info['t']

                t_new[..., 2].fill_(0)  # zero tz
                x_d_i_new = scale_new * (x_c_s @ R_new + delta_new) + t_new

                # Algorithm 1:
                if not inference_cfg.flag_stitching and not __eye__ and not __lip__:
                    # without stitching or retargeting
                    if inference_cfg.flag_lip_zero:
                        x_d_i_new += lip_delta_before_animation.reshape(-1, x_s.shape[1], 3)
                    else:
                        pass
                elif inference_cfg.flag_stitching and not __eye__ and not __lip__:
                    # with stitching and without retargeting
                    if inference_cfg.flag_lip_zero:
                        x_d_i_new = self.live_portrait_wrapper.stitching(x_s, x_d_i_new) + lip_delta_before_animation.reshape(-1, x_s.shape[1], 3)
                    else:
                        x_d_i_new = self.live_portrait_wrapper.stitching(x_s, x_d_i_new)
                else:
                    eyes_delta, lip_delta = None, None
                    if __eye__:
                        c_d_eyes_i = input_eye_ratio_lst[i]
                        combined_eye_ratio_tensor = self.live_portrait_wrapper.calc_combined_eye_ratio(c_d_eyes_i, source_lmk)
                        # ∆_eyes,i = R_eyes(x_s; c_s,eyes, c_d,eyes,i)
                        eyes_delta = self.live_portrait_wrapper.retarget_eye(x_s, combined_eye_ratio_tensor)
                    if __lip__:
                        c_d_lip_i = input_lip_ratio_lst[i]
                        combined_lip_ratio_tensor = self.live_portrait_wrapper.calc_combined_lip_ratio(c_d_lip_i, source_lmk)
                        # ∆_lip,i = R_lip(x_s; c_s,lip, c_d,lip,i)
                        lip_delta = self.live_portrait_wrapper.retarget_lip(x_s, combined_lip_ratio_tensor)

                    if inference_cfg.flag_relative:  # use x_s
                        x_d_i_new = x_s + \
                            (eyes_delta.reshape(-1, x_s.shape[1], 3) if eyes_delta is not None else 0) + \
                            (lip_delta.reshape(-1, x_s.shape[1], 3) if lip_delta is not None else 0)
                    else:  # use x_d,i
                        x_d_i_new = x_d_i_new + \
                            (eyes_delta.reshape(-1, x_s.shape[1], 3) if eyes_delta is not None else 0) + \
                            (lip_delta.reshape(-1, x_s.shape[1], 3) if lip_delta is not None else 0)

                    if inference_cfg.flag_stitching:
                        x_d_i_new = self.live_portrait_wrapper.stitching(x_s, x_d_i_new)

                out = self.live_portrait_wrapper.warp_decode(f_s, x_s, x_d_i_new)
                I_p_i = self.live_portrait_wrapper.parse_output(out['out'])[0]
                I_p_lst.append(I_p_i)

                if inference_cfg.flag_pasteback:
                    # img_rgbs=[img_rgb,img_rgb,img_rgb] 新构建的视频帧
                    img_rgb0=img_rgbs[i]

                    I_p_i_to_ori = _transform_img(I_p_i, crop_info['M_c2o'], dsize=(img_rgb0.shape[1], img_rgb0.shape[0]))
                    I_p_i_to_ori_blend = np.clip(mask_ori * I_p_i_to_ori + (1 - mask_ori) * img_rgb0, 0, 255).astype(np.uint8)
                    
                    out = np.hstack([I_p_i_to_ori, I_p_i_to_ori_blend])
                    I_p_paste_lst.append(I_p_i_to_ori_blend)

                    # 更新到img_rgbs
                    img_rgbs[i]=np.copy(I_p_i_to_ori_blend) 

                pbar.update(1)


        directory, filename = os.path.split(args.output_path)
        if not os.path.exists(directory):
            mkdir(directory)
            
            # wfp_concat = args.output_path_concat
            
            # if is_video(args.driving_info):
            #     frames_concatenated = concat_frames(I_p_lst, driving_rgb_lst, img_crop_256x256)
            #     # save (driving frames, source image, drived frames) result
            #     # wfp_concat = osp.join(directory, f'{basename(args.source_image)}--{basename(args.driving_info)}_concat.mp4')
            #     # images2video(frames_concatenated, wfp=wfp_concat)
            #     images2video(frames_concatenated, wfp=wfp_concat, fps=video_fps)

        # save drived result
        wfp = args.output_path
        if inference_cfg.flag_pasteback:
            images2video(I_p_paste_lst, wfp=wfp, fps=video_fps)
        
        return wfp

