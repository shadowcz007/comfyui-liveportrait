import os
import sys
import numpy as np
import torch
import cv2
from PIL import Image
import folder_paths
import copy,json
from ultralytics import YOLO

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
sys.path.append(current_directory)
from LivePortrait.src.live_portrait_wrapper import LivePortraitWrapper
from LivePortrait.src.utils.camera import get_rotation_matrix

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)
def rgb_crop(rgb, region):
    return rgb[region[1]:region[3], region[0]:region[2]]

def get_rgb_size(rgb):
    return rgb.shape[1], rgb.shape[0]
def create_transform_matrix(x, y, scale=1):
    return np.float32([[scale, 0, x], [0, scale, y]])

def get_model_dir(m):
    try:
        return folder_paths.get_folder_paths(m)[0]
    except:
        return os.path.join(folder_paths.models_dir, m)

def calc_crop_limit(center, img_size, crop_size):
    pos = center - crop_size / 2
    if pos < 0:
        crop_size += pos * 2
        pos = 0

    pos2 = pos + crop_size

    if img_size < pos2:
        crop_size -= (pos2 - img_size) * 2
        pos2 = img_size
        pos = pos2 - crop_size

    return pos, pos2, crop_size

 


# 修改模型路径 ， 沿用 comfyui-liveportrait
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

liveportrait_model=get_model_dir('liveportrait')

inference_cfg = InferenceConfig(
        models_config=os.path.join(current_directory,'LivePortrait','src','config','models.yaml'),
        checkpoint_F=os.path.join(liveportrait_model,'base_models','appearance_feature_extractor.pth'),
        checkpoint_M=os.path.join(liveportrait_model,'base_models','motion_extractor.pth') ,
        checkpoint_G=os.path.join(liveportrait_model,'base_models','spade_generator.pth') ,
        checkpoint_W=os.path.join(liveportrait_model,'base_models','warping_module.pth'),
        checkpoint_S=os.path.join(liveportrait_model,'retargeting_models','stitching_retargeting_module.pth')
    )

class PreparedSrcImg:
    def __init__(self, src_rgb, crop_trans_m, x_s_info, f_s_user, x_s_user, mask_ori):
        self.src_rgb = src_rgb
        self.crop_trans_m = crop_trans_m
        self.x_s_info = x_s_info
        self.f_s_user = f_s_user
        self.x_s_user = x_s_user
        self.mask_ori = mask_ori

class LP_Engine:
    pipeline = None
    bbox_model = None
    mask_img = None

    def detect_face(self, image_rgb):

        crop_factor = 1.7
        bbox_drop_size = 10

        if self.bbox_model == None:
            bbox_model_path = os.path.join(get_model_dir("ultralytics"), "face_yolov8n.pt")
            
            # 沿用 comfyui-ultralytics-yolo
            for fp in [os.path.join("bbox","face_yolov8m.pt"),os.path.join("bbox","face_yolov8n.pt"),"face_yolov8n.pt","face_yolov8m.pt",]:
                np=os.path.join(get_model_dir("ultralytics"), fp)
                if os.path.isfile(np):
                    bbox_model_path=np

            self.bbox_model = YOLO(bbox_model_path)

        pred = self.bbox_model(image_rgb, conf=0.7, device="")
        bboxes = pred[0].boxes.xyxy.cpu().numpy()

        w, h = get_rgb_size(image_rgb)

        # for x, label in zip(segmasks, detected_results[0]):
        for x1, y1, x2, y2 in bboxes:
            bbox_w = x2 - x1
            bbox_h = y2 - y1

            crop_w = bbox_w * crop_factor
            crop_h = bbox_h * crop_factor

            crop_w = max(crop_h, crop_w)
            crop_h = crop_w

            kernel_x = x1 + bbox_w / 2
            kernel_y = y1 + bbox_h / 2

            new_x1, new_x2, crop_w = calc_crop_limit(kernel_x, w, crop_w)

            if crop_w < crop_h:
                crop_h = crop_w

            new_y1, new_y2, crop_h = calc_crop_limit(kernel_y, h, crop_h)

            if crop_h < crop_w:
                crop_w = crop_h
                new_x1, new_x2, crop_w = calc_crop_limit(kernel_x, w, crop_w)

            return [int(new_x1), int(new_y1), int(new_x2), int(new_y2)]

        print("Failed to detect face!!")
        return [0, 0, w, h]

    def crop_face(self, rgb_img):
        region = self.detect_face(rgb_img)
        face_image = rgb_crop(rgb_img, region)
        return face_image, region

    def get_pipeline(self):
        if self.pipeline == None:
            print("Load pipeline...")
            self.pipeline = LivePortraitWrapper(cfg=inference_cfg)

        return self.pipeline

    def prepare_src_image(self, img):
        h, w = img.shape[:2]
        input_shape = [256,256]
        if h != input_shape[0] or w != input_shape[1]:
            x = cv2.resize(img, (input_shape[0], input_shape[1]), interpolation = cv2.INTER_LINEAR)
        else:
            x = img.copy()

        if x.ndim == 3:
            x = x[np.newaxis].astype(np.float32) / 255.  # HxWx3 -> 1xHxWx3, normalized to 0~1
        elif x.ndim == 4:
            x = x.astype(np.float32) / 255.  # BxHxWx3, normalized to 0~1
        else:
            raise ValueError(f'img ndim should be 3 or 4: {x.ndim}')
        x = np.clip(x, 0, 1)  # clip to 0~1
        x = torch.from_numpy(x).permute(0, 3, 1, 2)  # 1xHxWx3 -> 1x3xHxW
        x = x.cuda()
        return x

    def GetMask(self):
        if self.mask_img is None:
            path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "./LivePortrait/src/utils/resources/mask_template.png")
            self.mask_img = cv2.imread(path, cv2.IMREAD_COLOR)
        return self.mask_img

    def prepare_source(self, source_image, is_video = False):
        print("Prepare source...")
        engine = self.get_pipeline()
        source_image_np = (source_image * 255).byte().numpy()
        img_rgb = source_image_np[0]
        face_img, crop_region = self.crop_face(img_rgb)

        scale = face_img.shape[0] / 512.
        crop_trans_m = create_transform_matrix(crop_region[0], crop_region[1], scale)
        mask_ori = cv2.warpAffine(self.GetMask(), crop_trans_m, get_rgb_size(img_rgb), cv2.INTER_LINEAR)
        mask_ori = mask_ori.astype(np.float32) / 255.

        psi_list = []
        for img_rgb in source_image_np:
            face_img = rgb_crop(img_rgb, crop_region)
            i_s = self.prepare_src_image(face_img)
            x_s_info = engine.get_kp_info(i_s)
            f_s_user = engine.extract_feature_3d(i_s)
            x_s_user = engine.transform_keypoint(x_s_info)
            psi = PreparedSrcImg(img_rgb, crop_trans_m, x_s_info, f_s_user, x_s_user, mask_ori)
            if is_video == False:
                return psi
            psi_list.append(psi)

        return psi_list

    def prepare_driving_video(self, face_images):
        print("Prepare driving video...")
        pipeline = self.get_pipeline()
        f_img_np = (face_images * 255).byte().numpy()

        out_list = []
        for f_img in f_img_np:
            i_d = pipeline.prepare_source(f_img)
            d_info = pipeline.get_kp_info(i_d)
            #out_list.append((d_info, get_rotation_matrix(d_info['pitch'], d_info['yaw'], d_info['roll'])))
            out_list.append(d_info)

        return out_list

    def calc_fe(_, x_d_new, eyes, eyebrow, wink, pupil_x, pupil_y, mouth, eee, woo, smile,
                rotate_pitch, rotate_yaw, rotate_roll):

        x_d_new[0, 20, 1] += smile * -0.01
        x_d_new[0, 14, 1] += smile * -0.02
        x_d_new[0, 17, 1] += smile * 0.0065
        x_d_new[0, 17, 2] += smile * 0.003
        x_d_new[0, 13, 1] += smile * -0.00275
        x_d_new[0, 16, 1] += smile * -0.00275
        x_d_new[0, 3, 1] += smile * -0.0035
        x_d_new[0, 7, 1] += smile * -0.0035

        x_d_new[0, 19, 1] += mouth * 0.001
        x_d_new[0, 19, 2] += mouth * 0.0001
        x_d_new[0, 17, 1] += mouth * -0.0001
        rotate_pitch -= mouth * 0.05

        x_d_new[0, 20, 2] += eee * -0.001
        x_d_new[0, 20, 1] += eee * -0.001
        #x_d_new[0, 19, 1] += eee * 0.0006
        x_d_new[0, 14, 1] += eee * -0.001

        x_d_new[0, 14, 1] += woo * 0.001
        x_d_new[0, 3, 1] += woo * -0.0005
        x_d_new[0, 7, 1] += woo * -0.0005
        x_d_new[0, 17, 2] += woo * -0.0005

        x_d_new[0, 11, 1] += wink * 0.001
        x_d_new[0, 13, 1] += wink * -0.0003
        x_d_new[0, 17, 0] += wink * 0.0003
        x_d_new[0, 17, 1] += wink * 0.0003
        x_d_new[0, 3, 1] += wink * -0.0003
        rotate_roll -= wink * 0.1
        rotate_yaw -= wink * 0.1

        if 0 < pupil_x:
            x_d_new[0, 11, 0] += pupil_x * 0.0007
            x_d_new[0, 15, 0] += pupil_x * 0.001
        else:
            x_d_new[0, 11, 0] += pupil_x * 0.001
            x_d_new[0, 15, 0] += pupil_x * 0.0007

        x_d_new[0, 11, 1] += pupil_y * -0.001
        x_d_new[0, 15, 1] += pupil_y * -0.001
        eyes -= pupil_y / 2.

        x_d_new[0, 11, 1] += eyes * -0.001
        x_d_new[0, 13, 1] += eyes * 0.0003
        x_d_new[0, 15, 1] += eyes * -0.001
        x_d_new[0, 16, 1] += eyes * 0.0003


        if 0 < eyebrow:
            x_d_new[0, 1, 1] += eyebrow * 0.001
            x_d_new[0, 2, 1] += eyebrow * -0.001
        else:
            x_d_new[0, 1, 0] += eyebrow * -0.001
            x_d_new[0, 2, 0] += eyebrow * 0.001
            x_d_new[0, 1, 1] += eyebrow * 0.0003
            x_d_new[0, 2, 1] += eyebrow * -0.0003


        return torch.Tensor([rotate_pitch, rotate_yaw, rotate_roll])
g_engine = LP_Engine()

class ExpressionSet:
    def __init__(self, erst = None, es = None):
        if es != None:
            self.e = copy.deepcopy(es.e)  # [:, :, :]
            self.r = copy.deepcopy(es.r)  # [:]
            self.s = copy.deepcopy(es.s)
            self.t = copy.deepcopy(es.t)
        elif erst != None:
            self.e = erst[0]
            self.r = erst[1]
            self.s = erst[2]
            self.t = erst[3]
        else:
            self.e = torch.from_numpy(np.zeros((1, 21, 3))).float().to(device='cuda')
            self.r = torch.Tensor([0, 0, 0])
            self.s = 0
            self.t = 0
    def div(self, value):
        self.e /= value
        self.r /= value
        self.s /= value
        self.t /= value
    def add(self, other):
        self.e += other.e
        self.r += other.r
        self.s += other.s
        self.t += other.t
    def sub(self, other):
        self.e -= other.e
        self.r -= other.r
        self.s -= other.s
        self.t -= other.t
    def mul(self, value):
        self.e *= value
        self.r *= value
        self.s *= value
        self.t *= value

    #def apply_ratio(self, ratio):        self.exp *= ratio

 

class ExpressionEditor:
    def __init__(self):
        self.sample_image = None
        self.src_image = None

    @classmethod
    def INPUT_TYPES(s):
        display = "number"
        #display = "slider"
        return {
            "required": {
                "src_image": ("IMAGE",), 

                "rotate_pitch": ("FLOAT", {"default": 0, "min": -50, "max": 50, "step": 0.2, "display": display}),
                "rotate_yaw": ("FLOAT", {"default": 0, "min": -50, "max": 50, "step": 0.2, "display": display}),
                "rotate_roll": ("FLOAT", {"default": 0, "min": -50, "max": 50, "step": 0.2, "display": display}),

                "blink": ("FLOAT", {"default": 0, "min": -30, "max": 15, "step": 0.2, "display": display}),
                "eyebrow": ("FLOAT", {"default": 0, "min": -30, "max": 25, "step": 0.2, "display": display}),
                "wink": ("FLOAT", {"default": 0, "min": -10, "max": 25, "step": 0.2, "display": display}),
                
                "pupil_x": ("FLOAT", {"default": 0, "min": -15, "max": 15, "step": 0.2, "display": display}),
                "pupil_y": ("FLOAT", {"default": 0, "min": -15, "max": 15, "step": 0.2, "display": display}),
                
                "aaa": ("FLOAT", {"default": 0, "min": -30, "max": 120, "step": 1, "display": display}),
                "eee": ("FLOAT", {"default": 0, "min": -20, "max": 15, "step": 0.2, "display": display}),
                "woo": ("FLOAT", {"default": 0, "min": -20, "max": 15, "step": 0.2, "display": display}),
                
                "smile": ("FLOAT", {"default": 0, "min": -0.3, "max": 1.3, "step": 0.01, "display": display}),

                "src_weight": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01, "display": display}),
                # "sample_ratio": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01, "display": display}),
            },

            "optional": {
                
                "expression_json":("STRING", {"forceInput": True,"dynamicPrompts": False}),
                # "sample_image": ("IMAGE",), 
            },
        }

    RETURN_TYPES = ("IMAGE","STRING",)
    RETURN_NAMES = ("image","expression_json",)

    FUNCTION = "run"

    OUTPUT_NODE = True

    CATEGORY = "♾️Mixlab/Face"

    # INPUT_IS_LIST = False
    # OUTPUT_IS_LIST = (False,)

    def run(self,src_image, rotate_pitch, rotate_yaw, rotate_roll, blink, eyebrow, wink, pupil_x, pupil_y, aaa,
             eee, woo, smile,
            src_weight, expression_json=None):
        
        if expression_json!=None:
            try:
                dict_obj = json.loads(expression_json)
                if "rotate_pitch" in dict_obj:
                    rotate_pitch=dict_obj["rotate_pitch"]
                if "rotate_yaw" in dict_obj:
                    rotate_yaw=dict_obj["rotate_yaw"]
                if "rotate_roll" in dict_obj:
                    rotate_roll=dict_obj["rotate_roll"]
                if "blink" in dict_obj:
                    blink=dict_obj["blink"]
                if "eyebrow" in dict_obj:
                    eyebrow=dict_obj["eyebrow"]
                if "wink" in dict_obj:
                    wink=dict_obj["wink"]
                if "pupil_x" in dict_obj:
                    pupil_x=dict_obj["pupil_x"]
                if "pupil_y" in dict_obj:
                    pupil_y=dict_obj["pupil_y"]
                if "aaa" in dict_obj:
                    aaa=dict_obj["aaa"]
                if "eee" in dict_obj:
                    eee=dict_obj["eee"]
                if "woo" in dict_obj:
                    woo=dict_obj["woo"]
                if "smile" in dict_obj:
                    smile=dict_obj["smile"]
                if "src_weight" in dict_obj:
                    src_weight=dict_obj["src_weight"]
            except:
                print("#expression_json",expression_json)

        print('#expression_json',expression_json)

        expression_json={
            "rotate_pitch":rotate_pitch,
            "rotate_yaw":rotate_yaw,
            "rotate_roll":rotate_roll, 
            "blink":blink, 
            "eyebrow":eyebrow, 
            "wink":wink, 
            "pupil_x":pupil_x, 
            "pupil_y":pupil_y, 
            "aaa":aaa, 
            "eee":eee, 
            "woo":woo, 
            "smile":smile,
            "src_weight":src_weight
        }

        rotate_yaw = -rotate_yaw

        new_editor_link = None
       
        if id(src_image) != id(self.src_image):
            self.psi = g_engine.prepare_source(src_image)
            self.src_image = src_image
        new_editor_link = []
        new_editor_link.append(self.psi)
        

        pipeline = g_engine.get_pipeline()

        psi = self.psi
        s_info = psi.x_s_info
        #delta_new = copy.deepcopy()
        s_exp = s_info['exp'] * src_weight
        s_exp[0, 5] = s_info['exp'][0, 5]
        s_exp += s_info['kp']

        es = ExpressionSet()

        # if sample_image != None:
        #     if id(self.sample_image) != id(sample_image):
        #         self.sample_image = sample_image
        #         d_image_np = (sample_image * 255).byte().numpy()
        #         d_face, _ = g_engine.crop_face(d_image_np[0])
        #         i_d = pipeline.prepare_source(d_face)
        #         self.d_info = pipeline.get_kp_info(i_d)
        #         self.d_info['exp'][0, 5, 0] = 0
        #         self.d_info['exp'][0, 5, 1] = 0

        #     # delta_new += s_exp * (1 - sample_ratio) + self.d_info['exp'] * sample_ratio
        #     es.e += self.d_info['exp'] * sample_ratio

        es.r = g_engine.calc_fe(es.e, blink, eyebrow, wink, pupil_x, pupil_y, aaa, eee, woo, smile,
                                  rotate_pitch, rotate_yaw, rotate_roll)

        new_rotate = get_rotation_matrix(s_info['pitch'] + es.r[0], s_info['yaw'] + es.r[1],
                                         s_info['roll'] + es.r[2])
        x_d_new = (s_info['scale'] * (1 + es.s)) * ((s_exp + es.e) @ new_rotate) + s_info['t']

        x_d_new = pipeline.stitching(psi.x_s_user, x_d_new)

        crop_out = pipeline.warp_decode(psi.f_s_user, psi.x_s_user, x_d_new)
        crop_out = pipeline.parse_output(crop_out['out'])[0]

        crop_with_fullsize = cv2.warpAffine(crop_out, psi.crop_trans_m, get_rgb_size(psi.src_rgb), cv2.INTER_LINEAR)
        out = np.clip(psi.mask_ori * crop_with_fullsize + (1 - psi.mask_ori) * psi.src_rgb, 0, 255).astype(np.uint8)

        out_img = pil2tensor(out)

        return (out_img,json.dumps(expression_json) ,)

