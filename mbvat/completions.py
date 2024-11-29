import io
import base64
import random
import json
from typing import Literal
from mbvat.visualization import MBVATItem,Position
from mbvat.errors import NoPointFoundError

def create_localization_messages(item:MBVATItem, type: Literal["qwen","minicpm"] = "qwen"):
    img= item.draw()
    # convert to jpeg codec
    img_io = io.BytesIO()
    img.save(img_io, "JPEG")
    img_io.seek(0)

    match type:
        case "qwen":
            prompt = f"Given the image, please give a point's coordinate in the {item.obj.shape} in form of <point>x,y</point>, where x and y range from 0 to 1000.\nExample Responses:\n<point>123,456</point>"
            return [
                {"role":"user",
                    "content":[
                    {"type":"text","text":prompt},
                    {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+base64.b64encode(img_io.read()).decode()}},
                ]}
            ]

        case "minicpm":
            prompt = f"Given the image, please give a point's coordinate in the {item.obj.shape} in form of "+json.dumps({"action": {"name": "POINT", "args": {"coordinate": {"x": "integer, x index", "y": "integer, y index"}}}})+", where x and y range from 0 to 1000.\nExample Responses:\n"+json.dumps({"action": {"name": "POINT", "args": {"coordinate": {"x": 629, "y": 95}}}})
            return [
                {"role":"user",
                    "content":[
                    {"type":"text","text":prompt},
                    {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+base64.b64encode(img_io.read()).decode()}},
                ]}
            ]

        case _:
            raise ValueError(f"Unknown type: {type}")

def create_color_messages(item:MBVATItem, type: Literal["qwen"] = "qwen"):
    img= item.draw()
    # convert to jpeg codec
    img_io = io.BytesIO()
    img.save(img_io, "JPEG")
    img_io.seek(0)

    match type:
        case "qwen":
            prompt = f"Describe the shape of elements shown in the image in format of <shape>any_shapes_here</shape>. The validated shapes are: Rectangle, Circle, NotFound. If you does not find any shape, output <shape>NotFound</shape>\nExample Responses:\n<shape>Rectangle</shape>\n<shape>Circle</shape>"
            return [
                {"role":"user",
                    "content":[
                    {"type":"text","text":prompt},
                    {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+base64.b64encode(img_io.read()).decode()}},
                ]}
            ]


        case _:
            raise ValueError(f"Unknown type: {type}")


def extract_point(res:str, width:int, height:int ,type:Literal["qwen","minicpm"] = "qwen"):
    match type:
        case "qwen":
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
        
        case "minicpm":
            try:
                s = json.loads(res)
                x = s["action"]["args"]["coordinate"]
                return Position(x=int(x["x"]),y=int(x["y"]))
            except:
                raise NoPointFoundError(f"No point found: {res}")
        
        case _:
            raise ValueError(f"Unknown type: {type}")
        
        
def extract_shape(res:str, type:Literal["qwen"] = "qwen"):
    match type:
        case "qwen":
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