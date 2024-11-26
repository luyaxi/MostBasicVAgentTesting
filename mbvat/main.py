import fire
import io
import asyncio
from typing import Literal
from openai import AsyncOpenAI
import base64
import json
import re
import random
from .loc_test import LocalizationTester,LocaliationTestItem,Position,draw_res_correctness


sem = asyncio.Semaphore(32)
async def completions(item:LocaliationTestItem)->Position:
    async with sem:
        img= item.draw()
        # convert to jpeg codec
        img_io = io.BytesIO()
        img.save(img_io, "JPEG")
        img_io.seek(0)

        prompt = f"Given the image, please give a point's coordinate in the {item.obj.shape} in form of <point>x,y</point>, where x and y range from 0 to 1000."
        client = AsyncOpenAI(
            api_key="sk-1234",
            base_url="http://localhost:8000/v1"
        )
        response = await client.chat.completions.create(
            messages=[
                {"role":"user",
                    "content":[
                    {"type":"text","text":prompt},
                    {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+base64.b64encode(img_io.read()).decode()}},
                ]}
            ],
            model="qwen2vl"
        )
        response = response.choices[0].message.content
        # extract the point
        try:
            s = response.split("<point>")[-1].split("</point>")[0]
            x = s.split(",")
            if len(x) == 2:
                x,y = x
                return Position(x=int(int(x)/1000*item.res.width),y=int(int(y)/1000*item.res.height))
            else:
                raise ValueError(f"Unkown point format: {x}")
        except:

            # print("No point found: ",response)
            x = random.randint(0,item.res.width)
            y = random.randint(0,item.res.height)
            return Position(x=x,y=y)


async def random_pointer(item:LocaliationTestItem):
    import random
    x = random.randint(0,item.res.width)
    y = random.randint(0,item.res.height)
    return Position(x=x,y=y)

def main(
    test:Literal["localization"]="localization",
    random_test:bool = False,
    save_path:str = "results"
):
    match test:
        case "localization":
            tester = LocalizationTester(max_repeat_times=5)


            if random_test:
                results = asyncio.run(tester.run(random_pointer,save_path=save_path))
            else:
                results = asyncio.run(tester.run(completions,save_path=save_path))
        
        case _:
            raise ValueError(f"Unkown test {test}")
    
if __name__=="__main__":
    fire.Fire(main)