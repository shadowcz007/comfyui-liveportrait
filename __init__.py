from .nodes.live_portrait import LivePortraitNode,FaceCropInfo,Retargeting


NODE_CLASS_MAPPINGS = {
    "LivePortraitNode": LivePortraitNode, 
    "FaceCropInfo":FaceCropInfo,
    "Retargeting":Retargeting
}

# dict = { "key":value }

NODE_DISPLAY_NAME_MAPPINGS = {
   "LivePortraitNode":"Live Portrait",
   "FaceCropInfo":"Face Crop Info",
   "Retargeting":"Retargeting"
}

# web ui的节点功能
WEB_DIRECTORY = "./web"
