import fire
import io
import asyncio
from typing import Literal
from openai import AsyncOpenAI
import base64
import json
from .loc_test import LocalizationTester,LocaliationTestItem,Position,draw_res_correctness

sem = asyncio.Semaphore(1)
async def completions(item:LocaliationTestItem)->Position:
    async with sem:
        img= item.draw()
        # convert to jpeg codec
        img_io = io.BytesIO()
        img.save(img_io, "JPEG")
        img_io.seek(0)

        prompt = f"Please point out where the {item.obj.shape} is in form of <point>x,y</point>"
        client = AsyncOpenAI()
        response = await client.chat.completions.create(
            messages=[
                {"role":"user",
                    "content":[
                    {"type":"text","text":prompt},
                    {"type":"image_url","image_url":"data:image/jpeg;base64,"+base64.b64encode(img_io).decode()},
                ]}
            ],
            model="Qwen2-VL"
        )
        response = response.choices[0].message.content

async def random_pointer(item:LocaliationTestItem):
    import random
    x = random.randint(0,item.res.width)
    y = random.randint(0,item.res.height)
    return Position(x=x,y=y)

def main(
    test:Literal["localization"]="localization",
    random_test:bool = True
):
    match test:
        case "localization":
            tester = LocalizationTester()


            if random_test:
                results = asyncio.run(tester.run(random_pointer))
            else:
                results = asyncio.run(tester.run(completions))

            for i,item in enumerate(results):
                draw_res_correctness(item, save_path=f"results/{i}.png")
                ret = item
                for idx,t in enumerate(ret["results"]["test"]):
                    ret["results"]["test"][idx] = t.model_dump()
                    
                json.dump(ret,open(f"results/{i}.json","w"))
        
        case _:
            raise ValueError(f"Unkown test {test}")
    
if __name__=="__main__":
    fire.Fire(main)