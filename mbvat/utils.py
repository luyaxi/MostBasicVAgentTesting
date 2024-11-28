import numpy as np
from pydantic import BaseModel, Field
from PIL import Image, ImageDraw

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



class NoneObject(BasicObject):
    shape: str = "NotFound"
    def draw(self, img: Image.Image, pos: Position):
        return img

class TriangleObject(BasicObject):
    shape: str = "Triangle"
    width: int = Field(..., title='The width of the object.')
    height: int = Field(..., title='The height of the object.')
    
    def draw(self, img: Image.Image, pos: Position):
        draw = ImageDraw.Draw(img)
        x1, y1 = pos.x+self.width//2, pos.y
        x2, y2 = pos.x+self.width, pos.y+self.height
        x3, y3 = pos.x, pos.y+self.height
        draw.polygon([x1, y1, x2, y2, x3, y3], fill=self.color)
        return img
    
    def validate_point(self, pos: Position):
        # validate point in the drawn triangle
        x1, y1 = pos.x+self.width//2, pos.y
        x2, y2 = pos.x+self.width, pos.y+self.height
        x3, y3 = pos.x, pos.y+self.height
        # calculate the area of the triangle
        area = abs((x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2))/2)
        area1 = abs((pos.x*(y2-y3)+x2*(y3-pos.y)+x3*(pos.y-y2))/2)
        area2 = abs((x1*(pos.y-y3)+pos.x*(y3-y1)+x3*(y1-pos.y))/2)
        area3 = abs((x1*(y2-pos.y)+x2*(pos.y-y1)+pos.x*(y1-y2))/2)
        return area == area1+area2+area3
        
        
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
                            int] = Field(..., description='The background color of the image.')
    delta_e: float = Field(description="The color difference between the object and the background.")

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

def delta_e_cie2000(labc1, labc2, Kl=1, Kc=1, Kh=1):
    """Calculates the Delta E (CIE2000) of the colors.
    
    Args:
        labc1: The L*a*b* vectors. Shape: (N,3).
        labc2: The L*a*b* vectors. Shape: (N,3).
    
    Returns:
        The Delta E (CIE2000). Shape: (N,).
    """
    
    labc1 = np.array(labc1,dtype=np.float64)
    labc2 = np.array(labc2,dtype=np.float64)


    avg_Lp = (labc1[:,0] + labc2[:,0]) / 2.0 # N
    
    C1 = np.sqrt(np.sum(np.power(labc1[:, 1:], 2), axis=-1)) # N
    C2 = np.sqrt(np.sum(np.power(labc2[:, 1:], 2), axis=-1)) # N
    
    avg_C1_C2 = (C1 + C2) / 2.0 # N
    
    G = 0.5 * (1 - np.sqrt(np.power(avg_C1_C2, 7.0) / (np.power(avg_C1_C2, 7.0) + np.power(25.0, 7.0)))) # N
    a1p = (1.0 + G) * labc1[:, 1] # N
    a2p = (1.0 + G) * labc2[:, 1] # N
    
    C1p = np.sqrt(np.power(a1p, 2) + np.power(labc1[:,2], 2)) # N
    C2p = np.sqrt(np.power(a2p, 2) + np.power(labc2[:,2], 2)) # N
    avg_C1p_C2p = (C1p + C2p) / 2.0 # N
    
    h1p = np.degrees(np.arctan2(labc1[:,2], a1p)) # N
    h1p += (h1p < 0) * 360

    h2p = np.degrees(np.arctan2(labc2[:,2], a2p)) # N
    h2p += (h2p < 0) * 360

    diff_h2p_h1p = h2p - h1p
    delta_hp = diff_h2p_h1p + (np.fabs(diff_h2p_h1p) > 180) * 360
    delta_hp -= (h2p > h1p) * 720
    
    avg_Hp = (((np.fabs(h1p - h2p) > 180) * 360) + h1p + h2p) / 2.0
    
    T = 1 - 0.17 * np.cos(np.radians(avg_Hp - 30)) + \
        0.24 * np.cos(np.radians(2 * avg_Hp)) + \
        0.32 * np.cos(np.radians(3 * avg_Hp + 6)) - \
        0.2 * np.cos(np.radians(4 * avg_Hp - 63)) # N
                
    delta_Lp = labc1[:,0] - labc2[:,0] # N
    delta_Cp = C2p - C1p # N
    delta_Hp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(delta_hp) / 2.0)

    S_L = 1 + ((0.015 * np.power(avg_Lp - 50, 2)) / np.sqrt(20 + np.power(avg_Lp - 50, 2.0)))
    S_C = 1 + 0.045 * avg_C1p_C2p
    S_H = 1 + 0.015 * avg_C1p_C2p * T

    delta_ro = 60 * np.exp(-(np.power(((avg_Hp - 275) / 25), 2.0)))
    R_C = np.sqrt(np.power(avg_C1p_C2p, 7.0) / (np.power(avg_C1p_C2p, 7.0) + np.power(25.0, 7.0)))
    R_T = -2 * R_C * np.sin(np.radians(delta_ro))
    
    return np.sqrt(
        np.power(delta_Lp / (S_L * Kl), 2) +
        np.power(delta_Cp / (S_C * Kc), 2) +
        np.power(delta_Hp / (S_H * Kh), 2) +
        R_T * (delta_Cp / (S_C * Kc)) * (delta_Hp / (S_H * Kh))
    )

    

def cross_delta_e_cie2000(labc1,labc2, Kl=1, Kc=1, Kh=1):
    """
    Calculates the Delta E (CIE2000) matrix of the colors.
    
    Args:
        labc1: The L*a*b* vectors. Shape: (N,3).
        labc2: The L*a*b* vectors. Shape: (N,3).
    
    Returns:
        The Delta E (CIE2000) matrix. Shape: (N, N).
    """
    
    labc1 = np.array(labc1,dtype=np.float64)
    labc2 = np.array(labc2,dtype=np.float64)

    
    avg_Lp = (labc1[:,0].reshape(1,-1) + labc2[:,0].reshape(-1,1)) / 2.0 # N,N
    
    C1 = np.sqrt(np.sum(np.power(labc1[:, 1:], 2), axis=-1,keepdims=True)) # N,1
    C2 = np.sqrt(np.sum(np.power(labc2[:, 1:], 2), axis=-1,keepdims=True)) # N,1
    
    avg_C1_C2 = (C1.reshape(1,-1) + C2) / 2.0 # N,N
    
    G = 0.5 * (1 - np.sqrt(np.power(avg_C1_C2, 7.0) / (np.power(avg_C1_C2, 7.0) + np.power(25.0, 7.0)))) # N,N
    a1p = (1.0 + G) * labc1[:, 1].reshape(-1,1) # N,N
    a2p = (1.0 + G) * labc2[:, 1].reshape(1,-1) # N,N
    
    C1p = np.sqrt(np.power(a1p, 2) + np.power(labc1[:,2].reshape(-1,1), 2))
    C2p = np.sqrt(np.power(a2p, 2) + np.power(labc2[:,2].reshape(1,-1), 2))
    avg_C1p_C2p = (C1p + C2p) / 2.0
    
    h1p = np.degrees(np.arctan2(labc1[:,2].reshape(-1,1), a1p)) # N,N
    h1p += (h1p < 0) * 360

    h2p = np.degrees(np.arctan2(labc2[:,2].reshape(1,-1), a2p)) # N,N
    h2p += (h2p < 0) * 360

    diff_h2p_h1p = h2p - h1p
    delta_hp = diff_h2p_h1p + (np.fabs(diff_h2p_h1p) > 180) * 360
    delta_hp -= (h2p > h1p) * 720
    
    avg_Hp = (((np.fabs(h1p - h2p) > 180) * 360) + h1p + h2p) / 2.0
    
    T = 1 - 0.17 * np.cos(np.radians(avg_Hp - 30)) + \
        0.24 * np.cos(np.radians(2 * avg_Hp)) + \
        0.32 * np.cos(np.radians(3 * avg_Hp + 6)) - \
        0.2 * np.cos(np.radians(4 * avg_Hp - 63)) # N,N
                
    delta_Lp = labc1[:,0].reshape(1,-1) - labc2[:,0].reshape(-1,1) # N,N
    delta_Cp = C2p - C1p # N,N
    delta_Hp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(delta_hp) / 2.0)

    S_L = 1 + ((0.015 * np.power(avg_Lp - 50, 2)) / np.sqrt(20 + np.power(avg_Lp - 50, 2.0)))
    S_C = 1 + 0.045 * avg_C1p_C2p
    S_H = 1 + 0.015 * avg_C1p_C2p * T

    delta_ro = 60 * np.exp(-(np.power(((avg_Hp - 275) / 25), 2.0)))
    R_C = np.sqrt(np.power(avg_C1p_C2p, 7.0) / (np.power(avg_C1p_C2p, 7.0) + np.power(25.0, 7.0)))
    R_T = -2 * R_C * np.sin(np.radians(delta_ro))
    
    return np.sqrt(
        np.power(delta_Lp / (S_L * Kl), 2) +
        np.power(delta_Cp / (S_C * Kc), 2) +
        np.power(delta_Hp / (S_H * Kh), 2) +
        R_T * (delta_Cp / (S_C * Kc)) * (delta_Hp / (S_H * Kh))
    )
