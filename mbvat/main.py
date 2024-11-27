import fire
import asyncio
from typing import Literal
import json
import os
from mbvat.build_testset import LocalizationTester,MBVATItem,Position,ColorTester
from mbvat.completions import create_localization_messages,extract_point,create_color_messages,extract_shape,random_pointer,random_shape
from mbvat.visualization import draw_color_coverage,draw_res_correctness

sem = asyncio.Semaphore(32)
async def qwen_localization_completions(item:MBVATItem)->Position:
    async with sem:
        messages = create_localization_messages(item)
        from openai import AsyncOpenAI
        client = AsyncOpenAI(
            api_key="sk-1234",
            base_url="http://localhost:8000/v1"
        )
        response = await client.chat.completions.create(
            messages=messages,
            model="qwen2vl",
            # logprobs=True,
            # top_logprobs=20,
        )

        response = response.choices[0].message.content
        # extract the point
        point = extract_point(response,item.res.width,item.res.height)
        return point

async def qwen_color_completions(item:MBVATItem)->str:
    async with sem:
        messages = create_color_messages(item)
        from openai import AsyncOpenAI
        client = AsyncOpenAI(
            api_key="sk-1234",
            base_url="http://localhost:8000/v1"
        )
        response = await client.chat.completions.create(
            messages=messages,
            model="qwen2vl",
            # logprobs=True,
            # top_logprobs=20,
        )

        response = response.choices[0].message.content
        # extract the shape
        shape = extract_shape(response)
        return shape


def main(
    test:Literal["localization","localization_colorful","color"]="localization",
    random_test:bool = False,
    save_path:str = "results",
    max_repeat_times:int = 5,
):
    
    match test:
        case "localization":
            tester = LocalizationTester(max_repeat_times=max_repeat_times)
            completion_func = qwen_localization_completions if not random_test else random_pointer

        case "localization_colorful":
            tester = LocalizationTester(max_repeat_times=max_repeat_times,colorful=True)
            completion_func = qwen_localization_completions if not random_test else random_pointer

        case "color":
            tester = ColorTester(max_repeat_times=max_repeat_times)
            completion_func = qwen_color_completions if not random_test else random_shape

        case _:
            raise ValueError(f"Unkown test {test}")
        

        
    async def run_test():
        idx = 0
        async for statics in tester.run(completion_func):
            idx+=1
            print(statics["meta"])
            draw_res_correctness(statics, save_path=os.path.join(save_path,f"{idx}_loc.png"))
            if tester.colorful:
                draw_color_coverage(statics, save_path=os.path.join(save_path,f"{idx}_color_failed.png"),plot_success=False)
                draw_color_coverage(statics, save_path=os.path.join(save_path,f"{idx}_color_pass.png"),plot_success=True)

            for i,t in enumerate(statics["results"]["test"]):
                statics["results"]["test"][i] = t.model_dump()
            json.dump(statics,open(os.path.join(save_path,f"{idx}.json"),"w"))
            
    asyncio.run(run_test())

if __name__=="__main__":
    fire.Fire(main)