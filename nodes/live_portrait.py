import os
import sys
import folder_paths
import numpy as np
import torch
from PIL import Image
import folder_paths





# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)

# 获取当前文件的目录
current_directory = os.path.dirname(current_file_path)

# 添加当前插件的nodes路径，使ChatTTS可以被导入使用
sys.path.append(current_directory)

# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


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
                    output_dir='animations/',
                    device_id=0,
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
        self.output_dir = output_dir
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

inference_cfg = InferenceConfig(
        models_config=os.path.join(current_directory,r'LivePortrait\src\config\models.yaml'),
        checkpoint_F=os.path.join(liveportrait_model,r'base_models\appearance_feature_extractor.pth'),
        checkpoint_M=os.path.join(liveportrait_model,r'base_models\motion_extractor.pth') ,
        checkpoint_G=os.path.join(liveportrait_model,r'base_models\spade_generator.pth') ,
        checkpoint_W=os.path.join(liveportrait_model,r'base_models\warping_module.pth'),
        checkpoint_S=os.path.join(liveportrait_model,r'retargeting_models\stitching_retargeting_module.pth')
    )

crop_cfg = CropConfig()


class LivePortraitNode:
    def __init__(self):
        self.speaker = None
    @classmethod
    def INPUT_TYPES(s):
        
        return {"required": {
                        "source_image": ("IMAGE",),
                        "driving_video":("SCENE_VIDEO",),  
                        },
                # "optional":{ 
                #             "skip_refine_text":("BOOLEAN", {"default": False},),
                #         }
                }
    
    RETURN_TYPES = ("SCENE_VIDEO",)
    RETURN_NAMES = ("video",)

    FUNCTION = "run"

    CATEGORY = "♾️Mixlab/Video"

    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = (False,) #list 列表 [1,2,3]
  
    def run(self,source_image,driving_video):

        im=tensor2pil(source_image)

        #获取临时目录：temp
        output_dir = folder_paths.get_temp_directory()
        (
            full_output_folder,
            filename,
            counter,
            subfolder,
            _,
        ) = folder_paths.get_save_image_path('lp_tmp_', output_dir)
        
        image_file = f"{filename}_{counter:05}.png"

        image_path=os.path.join(full_output_folder, image_file)
        # 保存图片
        im.save(image_path,compress_level=6)


        args =  ArgumentConfig(
            source_image=image_path,
            driving_info=r'C:\Users\38957\Documents\ai-lab\ComfyUI_windows_portable\custom_nodes\comfyui-liveportrait\example\d0.mp4',
            output_dir=""
        )
        
        
        live_portrait_pipeline = LivePortraitPipeline(
            inference_cfg=inference_cfg,
            crop_cfg=crop_cfg,
            landmark_runner_ckpt=landmark_runner_ckpt,
            insightface_pretrained_weights=insightface_pretrained_weights
        )

        # run
        live_portrait_pipeline.execute(args)

        return (result,)

