
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



from pydantic import BaseModel, Field
from PIL import Image, ImageDraw

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


class MBVATItem(BaseModel):
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
