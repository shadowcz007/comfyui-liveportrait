from .nodes.live_portrait import LivePortraitNode,FaceCropInfo


NODE_CLASS_MAPPINGS = {
    "LivePortraitNode": LivePortraitNode, 
    "FaceCropInfo":FaceCropInfo
}

# dict = { "key":value }

NODE_DISPLAY_NAME_MAPPINGS = {
   "LivePortraitNode":"Live Portrait",
   "FaceCropInfo":"Face Crop Info"
}

# web ui的节点功能
WEB_DIRECTORY = "./web"
