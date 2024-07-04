import os,re
import sys
from pathlib import Path
import torchaudio
import hashlib
import torch
import folder_paths
import comfy.utils


# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)

# 获取当前文件的目录
current_directory = os.path.dirname(current_file_path)

# 添加当前插件的nodes路径，使ChatTTS可以被导入使用
sys.path.append(current_directory)


class LivePortraitNode:
    def __init__(self):
        self.speaker = None
    @classmethod
    def INPUT_TYPES(s):
        
        return {"required": {
                        "image_input": ("IMAGE",),
                        "video_input":("SCENE_VIDEO",),  
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
  
    def run(self,image_input,video_input):
        live_portrait_pipeline = LivePortraitPipeline(
            inference_cfg=inference_cfg,
            crop_cfg=crop_cfg
        )

        # run
        live_portrait_pipeline.execute(args)

        return (result,)

