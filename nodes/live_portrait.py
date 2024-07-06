import os
import sys
import folder_paths
import numpy as np
import torch
from PIL import Image
import folder_paths
# import comfy.utils




# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)

# 获取当前文件的目录
current_directory = os.path.dirname(current_file_path)

# 添加当前插件的nodes路径，使ChatTTS可以被导入使用
sys.path.append(current_directory)

# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# Convert PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)



from .LivePortrait.src.live_portrait_pipeline import LivePortraitPipeline

def get_model_dir(m):
    try:
        return folder_paths.get_folder_paths(m)[0]
    except:
        return os.path.join(folder_paths.models_dir, m)


class ArgumentConfig:
    def __init__(self,
                    source_image,
                    driving_info,
                    output_path='animations/v.mp4',
                    output_path_concat="",
                    device_id=0, 
                    crop_info =None,
                    face_index=0,
                    align_mode=True,
                    flag_lip_zero=True,
                    flag_eye_retargeting=False,
                    flag_lip_retargeting=False,
                    flag_stitching=True,
                    flag_relative=True,
                    flag_pasteback=True,
                    flag_do_crop=True,
                    flag_do_rot=True,
                    dsize=512,
                    scale=2.3,
                    vx_ratio=0,
                    vy_ratio=-0.125,
                    server_port=8890,
                    share=False,
                    server_name='0.0.0.0'):
        self.source_image = source_image
        self.driving_info = driving_info
        self.output_path = output_path
        self.output_path_concat=output_path_concat
        self.crop_info=crop_info
        self.face_index=face_index
        self.align_mode=align_mode
        self.device_id = device_id
        self.flag_lip_zero = flag_lip_zero
        self.flag_eye_retargeting = flag_eye_retargeting
        self.flag_lip_retargeting = flag_lip_retargeting
        self.flag_stitching = flag_stitching
        self.flag_relative = flag_relative
        self.flag_pasteback = flag_pasteback
        self.flag_do_crop = flag_do_crop
        self.flag_do_rot = flag_do_rot
        self.dsize = dsize
        self.scale = scale
        self.vx_ratio = vx_ratio
        self.vy_ratio = vy_ratio
        self.server_port = server_port
        self.share = share
        self.server_name = server_name

    
class InferenceConfig:
    def __init__(self,
                    models_config,
                    checkpoint_F,
                    checkpoint_M,
                    checkpoint_G,
                    checkpoint_W,
                    checkpoint_S,
                    mask_crop = None,
                    flag_use_half_precision=True,
                    flag_lip_zero=True,
                    lip_zero_threshold=0.03,
                    flag_eye_retargeting=False,
                    flag_lip_retargeting=False,
                    flag_stitching=True,
                    flag_relative=True,
                    anchor_frame=0,
                    input_shape=(256, 256),
                    output_format='mp4',
                    output_fps=30,
                    crf=15,
                    flag_write_result=True,
                    flag_pasteback=True,
                    flag_write_gif=False,
                    size_gif=256,
                    ref_max_shape=1280,
                    ref_shape_n=2,
                    device_id=0,
                    flag_do_crop=True,
                    flag_do_rot=True):
        self.models_config = models_config
        self.checkpoint_F = checkpoint_F
        self.checkpoint_M = checkpoint_M
        self.checkpoint_G = checkpoint_G
        self.checkpoint_W = checkpoint_W
        self.checkpoint_S = checkpoint_S
        self.flag_use_half_precision = flag_use_half_precision
        self.flag_lip_zero = flag_lip_zero
        self.lip_zero_threshold = lip_zero_threshold
        self.flag_eye_retargeting = flag_eye_retargeting
        self.flag_lip_retargeting = flag_lip_retargeting
        self.flag_stitching = flag_stitching
        self.flag_relative = flag_relative
        self.anchor_frame = anchor_frame
        self.input_shape = input_shape
        self.output_format = output_format
        self.output_fps = output_fps
        self.crf = crf
        self.flag_write_result = flag_write_result
        self.flag_pasteback = flag_pasteback
        self.flag_write_gif = flag_write_gif
        self.size_gif = size_gif
        self.ref_max_shape = ref_max_shape
        self.ref_shape_n = ref_shape_n
        self.device_id = device_id
        self.flag_do_crop = flag_do_crop
        self.flag_do_rot = flag_do_rot
        self.mask_crop=mask_crop

class CropConfig:
    def __init__(self, dsize=512, scale=2.3, vx_ratio=0, vy_ratio=-0.125):
        self.dsize = dsize
        self.scale = scale
        self.vx_ratio = vx_ratio
        self.vy_ratio = vy_ratio


liveportrait_model=get_model_dir('liveportrait')
insightface_pretrained_weights=get_model_dir('insightface')

landmark_runner_ckpt=os.path.join(liveportrait_model,'landmark.onnx')
# print('#landmark_runner_ckpt',landmark_runner_ckpt)
inference_cfg = InferenceConfig(
        models_config=os.path.join(current_directory,'LivePortrait','src','config','models.yaml'),
        checkpoint_F=os.path.join(liveportrait_model,'base_models','appearance_feature_extractor.pth'),
        checkpoint_M=os.path.join(liveportrait_model,'base_models','motion_extractor.pth') ,
        checkpoint_G=os.path.join(liveportrait_model,'base_models','spade_generator.pth') ,
        checkpoint_W=os.path.join(liveportrait_model,'base_models','warping_module.pth'),
        checkpoint_S=os.path.join(liveportrait_model,'retargeting_models','stitching_retargeting_module.pth')
    )

crop_cfg = CropConfig()


# 人脸检测并裁切
class FaceCropInfo:
    @classmethod
    def INPUT_TYPES(s):
        
        return {"required": {
                        "source_image": ("IMAGE",),
                        },
                "optional":{ 
                            "face_sorting_direction":(["left-right","large-small"],  {"default": "left-right"}),
                            "face_index":("INT", {"default": 0, "min": -1,"max":200, "step": 1, "display": "number"}),
                            "debug":("BOOLEAN", {"default": False},),
                        }
                }
    
    RETURN_TYPES = ("CROP_INFO","IMAGE",)
    RETURN_NAMES = ("crop_info","debug_image",)

    FUNCTION = "run"

    OUTPUT_NODE = True

    CATEGORY = "♾️Mixlab/Video"

    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = (True,False,) #list 列表 [1,2,3]
  
    def run(self,source_image,face_sorting_direction="left-right",face_index=0,debug=False):

        pil_image=tensor2pil(source_image)
        # Convert PIL image to NumPy array
        opencv_image = np.array(pil_image)

        # print('##---------------------------------#landmark_runner_ckpt',landmark_runner_ckpt)
        live_portrait_pipeline = LivePortraitPipeline(
            inference_cfg=inference_cfg,
            crop_cfg=crop_cfg,
            landmark_runner_ckpt=landmark_runner_ckpt,
            insightface_pretrained_weights=insightface_pretrained_weights
        )

        crop_info,debug_image = live_portrait_pipeline.cropper.crop_all_image(
            opencv_image,
            direction=face_sorting_direction,
            debug=debug
            )
        
        debug_image=pil2tensor(debug_image)

        if face_index>-1:
            #只输出一张 [face]
            crop_info=[crop_info[face_index]]

        return (crop_info,debug_image,)


# 驱动模板制作
# class DriveVideoNode:

#     @classmethod
#     def INPUT_TYPES(s):
        
#         return {"required": {
#                         "driving_video1":("SCENE_VIDEO",),
#                         "driving_video2":("SCENE_VIDEO",),
#                         },
#                 # "optional":{ 
#                 #         "face_index":("INT", {"default": 0, "min": -1,"max":200, "step": 1, "display": "number"}),
                        
#                 #         }
#                 }
    
#     RETURN_TYPES = ("DRIVING_VIDEO",)
#     RETURN_NAMES = ("driving_video",)

#     FUNCTION = "run"

#     OUTPUT_NODE = True

#     CATEGORY = "♾️Mixlab/Video"

#     INPUT_IS_LIST = False
#     OUTPUT_IS_LIST = (True,) #list 列表 [1,2,3]
  
#     def run(self,driving_video1, driving_video2 ):
        
#         return ([driving_video1, driving_video2],)
    

class LivePortraitNode:
    def __init__(self):
        self.speaker = None
    @classmethod
    def INPUT_TYPES(s):
        
        return {"required": {
                        "source_image": ("IMAGE",),
                        "driving_video":("SCENE_VIDEO",),  
                        },
                "optional":{  
                            "crop_info":("CROP_INFO", ),
                            "driving_video_reverse_align":("BOOLEAN", {"default": True},),
                        }
                }
    
    RETURN_TYPES = ("SCENE_VIDEO","SCENE_VIDEO",)
    RETURN_NAMES = ("video","video_concat",)

    FUNCTION = "run"

    CATEGORY = "♾️Mixlab/Video"

    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (False,False,) #list 列表 [1,2,3]
  
    def run(self,source_image,driving_video,crop_info=None,driving_video_reverse_align=True):
        # print('#crop_info',crop_info,isinstance(crop_info, list))
        if crop_info!=None and isinstance(crop_info, list)==False:
            crop_info=[crop_info]
        
        if crop_info!=None:
            crop_info=[ [c] for c in crop_info]

        pil_image=tensor2pil(source_image[0])
        # Convert PIL image to NumPy array
        opencv_image = np.array(pil_image)

        #获取临时目录：temp
        output_dir = folder_paths.get_temp_directory()

        def count_live_portrait_mp4_files(output_dir: str) -> int:
            count = 0
            for filename in os.listdir(output_dir):
                if filename.startswith('live_portrait_') and filename.endswith('.mp4'):
                    count += 1
            return count

        counter=count_live_portrait_mp4_files(output_dir)
        
        v_file = f"live_portrait_{counter:05}.mp4"
        v_file_concat = f"live_portrait_concat_{counter:05}.mp4"

        v_path=os.path.join(output_dir, v_file)
        output_path_concat=os.path.join(output_dir, v_file_concat)

        # print('##---------------------------------#landmark_runner_ckpt',landmark_runner_ckpt)
        live_portrait_pipeline = LivePortraitPipeline(
            inference_cfg=inference_cfg,
            crop_cfg=crop_cfg,
            landmark_runner_ckpt=landmark_runner_ckpt,
            insightface_pretrained_weights=insightface_pretrained_weights
        )

        # run
        if crop_info==None:

            args =  ArgumentConfig(
                source_image=opencv_image,
                driving_info=[driving_video[0]],
                output_path=v_path,
                output_path_concat=output_path_concat,
                crop_info=crop_info, 
            )

            live_portrait_pipeline.execute(args)
        else:

            if len(driving_video)!=len(crop_info):
                last_d=driving_video[-1]
                ds=[]
                #todo 视频的帧要对齐
                for i in range(len(crop_info)):
                    if driving_video[i]:
                        ds.append(driving_video[i])
                    else:
                        ds.append(last_d)
                driving_video=ds
            
                
            args =  ArgumentConfig(
                source_image=opencv_image,
                driving_info=driving_video,
                output_path=v_path,
                output_path_concat=output_path_concat,
                crop_info=crop_info, 
                align_mode=driving_video_reverse_align==False
            )

            # print('#executeForAll',len(crop_info))
            live_portrait_pipeline.executeForAll(args)

        live_portrait_pipeline.live_portrait_wrapper=None

        live_portrait_pipeline=None

        return (v_path,output_path_concat,)

