'''Here we testing the model's localization ability.

We consider multiple factors that could affect models' localization ability:
- The size of the object.
- The color of the object.
- The shape of the object.
- The background texture.
- The location of the object.
'''

from pydantic import BaseModel, Field
from PIL import Image, ImageDraw
import random
import math
import numpy as np

VALID_RESOLUTIONS = [
    # regular desktop resolution
    (1024, 768),
    (1280, 720),
    (1280, 800),
    (1920, 1080),
    (2560, 1440),
    (3840, 2160),
    (5120, 2880),
    (7680, 4320),
    # phone resolution
    (750, 1334),  # iPhone 6,7,8
    (828, 1792),  # iPhone XR
    (1080, 2256),  # vivo nex 3
    (1080, 2340),  # huawei p30
    (1080, 2376),  # vivo x50
    (1080, 2400),  # galaxy s20
    (1125, 2436),  # iPhone X, XS, 11 Pro
    (1170, 2532),  # iPhone 12, 12 Pro
    (1176, 2400),  # huawei mate30
    (1179, 2556),  # iphone 15
    (1200, 2670),  # xiaomi 15
    (1216, 2688),  # huawei mate 60
    (1228, 2700),  # huawei p60
    (1240, 2772),  # oppo find x6
    (1242, 2688),  # iPhone XS Max, 11 Pro Max
    (1260, 2800),  # vivo x90 pro
    (1284, 2778),  # iPhone 12 Pro Max
    (1290, 2796),  # iPhone 15 Pro Max
    (1344, 2992),  # google pixel 8 pro
    (1440, 3200),  # galaxy s20u
    (1440, 3088),  # galaxy note 20u
    (1440, 3168),  # oppo find x2
    (1440, 3168),  # oneplus 8 pro
    (1440, 3216),  # oneplus 11
]

COMMON_RESOLUTIONS = [
    (1280, 720),
    (828, 1792),  # iPhone XR
    (1920, 1080),
    (1080, 2400),  # galaxy s20
    (1179, 2556),  # iphone 15
    (1242, 2688),  # iPhone XS Max, 11 Pro Max
    (1440, 3216),  # oneplus 11
    (3840, 2160),
]


class Resolution(BaseModel):
    width: int
    height: int


class Position(BaseModel):
    x: int = Field(..., title='The x coordinate of the object.')
    y: int = Field(..., title='The y coordinate of the object.')


class BasicObject(BaseModel):
    color: str | tuple[int, int,
                       int] = Field(..., title='The color of the object.')
    bbox: list[int, int, int,
               int] = Field(..., title='The bounding box of the object.')
    shape: str = "Unknown"

    def draw(self, img: Image.Image, pos: Position):
        raise NotImplementedError

    def validate_point(self, pos: Position):
        raise NotImplementedError


class RectangleObject(BasicObject):
    shape: str = "Rectangle"
    width: int = Field(..., title='The width of the object.')
    height: int = Field(..., title='The height of the object.')

    def draw(self, img: Image.Image, pos: Position):
        draw = ImageDraw.Draw(img)
        x1, y1 = pos.x, pos.y
        x2, y2 = x1+self.width, y1+self.height
        draw.rectangle([x1, y1, x2, y2], fill=self.color)
        return img

    def validate_point(self, pos: Position):
        return 0 <= pos.x and pos.x <= self.bbox[2] and 0 <= pos.y and pos.y <= self.bbox[3]


class CircleObject(BasicObject):
    shape: str = "Circle"
    radius: int = Field(..., title='The radius of the object.')

    def draw(self, img: Image.Image, pos: Position):
        draw = ImageDraw.Draw(img)
        x1, y1 = pos.x, pos.y
        x2, y2 = x1+self.radius*2, y1+self.radius*2
        draw.ellipse([x1, y1, x2, y2], fill=self.color)
        return img

    def validate_point(self, pos: Position):
        # first calculate the distance between the pos to bbox center
        d = ((pos.x-self.bbox[2]/2)**2+(pos.y-self.bbox[3]/2)**2)**(0.5)
        if d <= self.radius:
            return True
        else:
            return False


class LocaliationTestItem(BaseModel):
    res: Resolution
    obj: BasicObject
    topleft_pos: Position
    background: str | tuple[int, int,
                            int] = Field(..., title='The background color of the image.')

    @property
    def center(self):
        return Position(x=self.topleft_pos.x+self.obj.bbox[2]//2, y=self.topleft_pos.y+self.obj.bbox[3]//2)

    def draw(self):
        back = Image.new(
            'RGB', (self.res.width, self.res.height), color=self.background)
        img = self.obj.draw(back, self.topleft_pos)
        return img

    def validate_point(self, pos: Position):
        delta_pos = Position(x=pos.x-self.topleft_pos.x,
                             y=pos.y-self.topleft_pos.y)
        return self.obj.validate_point(delta_pos)

    def validate_bbox(self, bbox: list[int, int, int, int], threshold: float = 0.5):
        """True if IoU is larger than threshold"""
        x1, y1 = self.topleft_pos.x, self.topleft_pos.y
        x2, y2 = x1+self.obj.bbox[2], y1+self.obj.bbox[3]
        x3, y3 = bbox[0], bbox[1]
        x4, y4 = bbox[0]+bbox[2], bbox[1]+bbox[3]
        # calculate the IoU
        x5 = max(x1, x3)
        y5 = max(y1, y3)
        x6 = min(x2, x4)
        y6 = min(y2, y4)
        inter_area = max(0, x6-x5)*max(0, y6-y5)
        union_area = (x2-x1)*(y2-y1) + (x4-x3)*(y4-y3) - inter_area
        return inter_area/union_area >= threshold


def build_full_localization_test(
        resolutions: list[tuple[int, int]] = COMMON_RESOLUTIONS,
        windows_ratio: list[float] = [0.1, 0.05, 0.01, 0.005],
        max_repeat_times: int = 1,
        delta_size=0.2,
        colorful=False
        ) -> dict[float, dict[tuple[int, int], list[LocaliationTestItem]]]:
    dataset = {}
    total_size = 0
    for ws in windows_ratio:
        for res in resolutions:
            if res not in dataset:
                dataset[res] = {}
            dataset[res][ws] = []
            width,height = res
            # calculate how many samples is sufficient for the ws
            windows_area = int(ws*width*height)
            minimum_square_len = math.floor(math.sqrt(windows_area))
            
            # 计算接近min_samples的行数和列数
            cols = math.ceil(2*width/minimum_square_len)
            rows = math.ceil(2*height/minimum_square_len)
            
            # 生成网格点
            x = np.linspace(0, width, cols)
            y = np.linspace(0, height, rows)

            # 计算网格间距
            x_spacing = width / (cols - 1)
            y_spacing = height / (rows - 1)
            
            # 创建网格
            xv, yv = np.meshgrid(x, y)
            grid_points = np.column_stack([xv.ravel(), yv.ravel()])
            
            # 生成随机偏移
            shifts_x = (np.random.rand(*grid_points[:, 0].shape) - 0.5) * x_spacing
            shifts_y = (np.random.rand(*grid_points[:, 1].shape) - 0.5) * y_spacing
            
            # 应用偏移
            shifted_points = grid_points + np.column_stack([shifts_x, shifts_y])
            for _ in range(max_repeat_times-1):
                shifts_x_new = (np.random.rand(*grid_points[:, 0].shape) - 0.5) * x_spacing
                shifts_y_new = (np.random.rand(*grid_points[:, 1].shape) - 0.5) * y_spacing
                shifted_new = grid_points + np.column_stack([shifts_x_new, shifts_y_new])
                shifted_new[:, 0] = np.clip(shifted_new[:, 0], 0, width - 1)
                shifted_new[:, 1] = np.clip(shifted_new[:, 1], 0, height - 1)
                shifted_points = np.vstack([shifted_points, shifted_new])

            # 确保偏移后的点在边界内
            shifted_points[:, 0] = np.clip(shifted_points[:, 0], 0, width-1)
            shifted_points[:, 1] = np.clip(shifted_points[:, 1], 0, height-1)

            def sample_objects():
                objs_type = np.random.randint(2,size=len(shifted_points))
                objs_area = ws*(
                    1+np.random.uniform(
                        -delta_size,
                        delta_size,
                        len(shifted_points))
                        )*width*height
                if colorful:
                    color1 = np.random.randint(0,255,(len(shifted_points),3))
                    color2 = np.random.randint(0,255,(len(shifted_points),3))
                else:
                    color1 = []
                    color2 = []
                    for bright in np.random.randint(0,1,(len(shifted_points))):
                        if bright:
                            color1.append((255,255,255))
                            color2.append((0,0,0))
                        else:
                            color1.append((0,0,0))
                            color2.append((255,255,255))
                # print(len(objs_type),len(shifted_points),len(objs_area),len(color1),len(color2))

                for otype,point,area,c1,c2 in zip(objs_type,shifted_points,objs_area,color1,color2):
                    if otype == 0:
                        # rectangle
                        # center point must be the `point`
                        max_width = min(point[0]*2,2*(width-point[0]),area/4)
                        min_width = max(4,math.sqrt(0.2*area))
                        if min_width > max_width:
                            # print(min_width,max_width)
                            continue

                        # Calculate width and height with a more balanced approach
                        aspect_ratio = random.uniform(0.5, 2)  # Adjust the range as needed for balance
                        obj_width = int(math.sqrt(area / aspect_ratio))
                        obj_height = int(area / obj_width)

                        # Ensure width and height are within bounds
                        obj_width = max(int(min_width), min(int(max_width), obj_width))
                        obj_height = int(area / obj_width)
                        obj = RectangleObject(width=obj_width, height=obj_height, color=c1, bbox=[
                                            0, 0, obj_width, obj_height])

                    if otype == 1:
                        # circle
                        max_radius = int(min(point[0],(width-point[0]),point[1],(height-point[1])))

                        if max_radius<=0:
                            continue
                            
                        radius = min(int((area/math.pi)**0.5),max_radius)
                        obj = CircleObject(radius=radius, color=c1, bbox=[
                               0, 0, 2*radius, 2*radius])
                
                    # print(max(point[0]-obj.bbox[2]//2,0))
                    dataset[res][ws].append(LocaliationTestItem(
                        res=Resolution(width=width,height=height),
                        obj=obj,
                        topleft_pos=Position(
                            x=int(max(point[0]-obj.bbox[2]//2,0)),
                            y=int(max(point[1]-obj.bbox[3]//2,0))
                        ),
                        background=c2
                    ))

            sample_objects()
            # print(f"Res: {res}, ws: {ws}, total: {len(dataset[res][ws])}")
            total_size += len(dataset[res][ws])
    # print("Total: ",total_size)
    return dataset

