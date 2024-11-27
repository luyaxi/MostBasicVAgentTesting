import io
import base64
import random
from typing import Literal
from mbvat.visualization import MBVATItem,Position
from mbvat.errors import NoPointFoundError

def create_localization_messages(item:MBVATItem, type: Literal["qwenvl"] = "qwenvl"):
    img= item.draw()
    # convert to jpeg codec
    img_io = io.BytesIO()
    img.save(img_io, "JPEG")
    img_io.seek(0)

    match type:
        case "qwenvl":
            prompt = f"Given the image, please give a point's coordinate in the {item.obj.shape} in form of <point>x,y</point>, where x and y range from 0 to 1000."
            return [
                {"role":"user",
                    "content":[
                    {"type":"text","text":prompt},
                    {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+base64.b64encode(img_io.read()).decode()}},
                ]}
            ]


        case _:
            raise ValueError(f"Unknown type: {type}")

def create_color_messages(item:MBVATItem, type: Literal["qwenvl"] = "qwenvl"):
    img= item.draw()
    # convert to jpeg codec
    img_io = io.BytesIO()
    img.save(img_io, "JPEG")
    img_io.seek(0)

    match type:
        case "qwenvl":
            prompt = f"Describe the shape of elements shown in the image in format of <shape>any_shapes_here</shape>. The validated shapes are: Rectangle, Circle."
            return [
                {"role":"user",
                    "content":[
                    {"type":"text","text":prompt},
                    {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+base64.b64encode(img_io.read()).decode()}},
                ]}
            ]


        case _:
            raise ValueError(f"Unknown type: {type}")


def extract_point(res:str, width:int, height:int ,type:Literal["qwenvl"] = "qwenvl"):
    match type:
        case "qwenvl":
            try:
                s = res.split("<point>")[-1].split("</point>")[0]
                x = s.split(",")
                if len(x) == 2:
                    x,y = x
                    return Position(x=int(int(x)/1000*width),y=int(int(y)/1000*height))
                else:
                    raise ValueError(f"Unkown point format: {x}")
            except:
                raise NoPointFoundError(f"No point found: {res}")
        
        case _:
            raise ValueError(f"Unknown type: {type}")
        
        
def extract_shape(res:str, type:Literal["qwenvl"] = "qwenvl"):
    match type:
        case "qwenvl":
            try:
                s = res.split("<shape>")[-1].split("</shape>")[0]
                return s
            except:
                raise ValueError(f"No shape found: {res}")
        
        case _:
            raise ValueError(f"Unknown type: {type}")
        

async def random_pointer(item:MBVATItem):
    import random
    x = random.randint(0,item.res.width)
    y = random.randint(0,item.res.height)
    return Position(x=x,y=y)

async def random_shape(item:MBVATItem):
    shapes = ["Rectangle","Circle"]
    return random.choice(shapes)