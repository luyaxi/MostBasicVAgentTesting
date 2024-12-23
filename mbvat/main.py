import fire
import asyncio
from typing import Literal
import json
import os
from mbvat.build_testset import LocalizationTester,MBVATItem,Position,ColorTester
from mbvat.completions import create_localization_messages,extract_point,create_color_messages,extract_shape,random_pointer,random_shape
from mbvat.visualization import draw_color_coverage,draw_res_correctness

BASE_URL = "http://localhost:8000/v1"
API_KEY = "sk-1234"
MODEL_NAME = "qwen2vl"
TEMPLATE_TYPE = "qwen"

sem = asyncio.Semaphore(32)
async def localization_completions(item:MBVATItem)->Position:
    async with sem:
        
        messages = create_localization_messages(item,TEMPLATE_TYPE)
        from openai import AsyncOpenAI
        client = AsyncOpenAI(
            api_key=API_KEY,
            base_url=BASE_URL
        )
        response = await client.chat.completions.create(
            messages=messages,
            model=MODEL_NAME,
            temperature=0,
            # logprobs=True,
            # top_logprobs=20,
        )

        response = response.choices[0].message.content
        # extract the point
        point = extract_point(response,item.res.width,item.res.height,type=TEMPLATE_TYPE)
        return point

async def color_completions(item:MBVATItem)->str:
    async with sem:
        messages = create_color_messages(item,type=TEMPLATE_TYPE)
        from openai import AsyncOpenAI
        client = AsyncOpenAI(
            api_key=API_KEY,
            base_url=BASE_URL
        )
        response = await client.chat.completions.create(
            messages=messages,
            model=MODEL_NAME,
            temperature=0,
            # logprobs=True,
            # top_logprobs=20,
        )

        response = response.choices[0].message.content
        # extract the shape
        shape = extract_shape(response,type=TEMPLATE_TYPE)
        return shape


def main(
    test:Literal["localization","localization_colorful","color"]="localization",
    random_test:bool = False,
    save_path:str = "results",
    max_repeat_times:int = 5,
    base_url: str = None,
    api_key: str = None,
    model:str = None,
    template_type:str = None
):
    global BASE_URL,API_KEY,MODEL_NAME,TEMPLATE_TYPE
    if base_url is not None:
        BASE_URL = base_url
    if api_key is not None:
        API_KEY = api_key
    if model is not None:
        MODEL_NAME = model
    if template_type is not None:
        TEMPLATE_TYPE = template_type

    os.makedirs(save_path,exist_ok=True)

    match test:
        case "localization":
            tester = LocalizationTester(max_repeat_times=max_repeat_times)
            completion_func = localization_completions if not random_test else random_pointer

        case "localization_colorful":
            tester = LocalizationTester(max_repeat_times=max_repeat_times,colorful=True)
            completion_func = localization_completions if not random_test else random_pointer

        case "color":
            tester = ColorTester(max_repeat_times=max_repeat_times)
            completion_func = color_completions if not random_test else random_shape

        case _:
            raise ValueError(f"Unkown test {test}")
        

        
    async def run_test():
        idx = 0
        async for statics in tester.run(completion_func):
            idx+=1
            print(statics["meta"])
            if len(statics["results"]["test"])==0:
                print("Warning: No test results")
                continue
            draw_res_correctness(statics, save_path=os.path.join(save_path,f"{idx}_loc.png"))
            if tester.colorful:
                draw_color_coverage(statics, save_path=os.path.join(save_path,f"{idx}_color.png"))

            for i,t in enumerate(statics["results"]["test"]):
                statics["results"]["test"][i] = t.model_dump()
            json.dump(statics,open(os.path.join(save_path,f"{idx}.json"),"w"))
            
    asyncio.run(run_test())

if __name__=="__main__":
    fire.Fire(main)