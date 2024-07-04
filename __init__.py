from .nodes.live_portrait import LivePortraitNode


NODE_CLASS_MAPPINGS = {
    "LivePortraitNode": LivePortraitNode, 
}

# dict = { "key":value }

NODE_DISPLAY_NAME_MAPPINGS = {
   "LivePortraitNode":"LivePortrait"
}

# web ui的节点功能
WEB_DIRECTORY = "./web"
