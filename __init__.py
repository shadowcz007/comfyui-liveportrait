from .nodes.live_portrait import LivePortraitNode,FaceCropInfo,Retargeting,LivePortraitVideoNode
from .nodes.expression_editor import ExpressionEditor

NODE_CLASS_MAPPINGS = {
    "LivePortraitNode": LivePortraitNode, 
    "LivePortraitVideoNode":LivePortraitVideoNode,
    "FaceCropInfo":FaceCropInfo,
    "Retargeting":Retargeting,
    "ExpressionEditor_":ExpressionEditor
}

# dict = { "key":value }

NODE_DISPLAY_NAME_MAPPINGS = {
   "LivePortraitNode":"Live Portrait",
   "LivePortraitVideoNode":"Live Portrait for Video",
   "FaceCropInfo":"Face Crop Info",
   "Retargeting":"Retargeting",
   "ExpressionEditor_":"Expression Editor"
}

# web ui的节点功能
WEB_DIRECTORY = "./web"
