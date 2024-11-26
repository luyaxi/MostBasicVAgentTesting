import asyncio
import matplotlib.pyplot as plt
from typing import List, Literal, Callable, Coroutine, Any
from scipy.ndimage import gaussian_filter
import numpy as np
import math
import json
import os
from tqdm import tqdm

from .build_loc_testset import COMMON_RESOLUTIONS,VALID_RESOLUTIONS,build_full_localization_test,LocaliationTestItem,Position
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

def calculate_color_delta(color1,color2):
    c1 = sRGBColor(color1[0],color1[1],color1[2])
    c2 = sRGBColor(color2[0],color2[1],color2[2])
    c1_lab = convert_color(c1,LabColor)
    c2_lab = convert_color(c2,LabColor)
    return delta_e_cie2000(c1_lab, c2_lab)

class LocalizationTester:
    def __init__(
            self,
            resolutions: list[tuple[int, int]] = COMMON_RESOLUTIONS,
            windows_ratio: list[float] = [0.1, 0.05, 0.01, 0.005],
            max_repeat_times: int = 1
    ):
        self.data = build_full_localization_test(
            resolutions=resolutions,
            windows_ratio=windows_ratio,
            max_repeat_times=max_repeat_times
        )

    def __len__(self):
        total = 0
        for res, res_testset in  self.data.items():
            for ws,subset  in res_testset.items():
                total+=len(subset)
        return total
    
    def __getitem__(self, idx):
        for res, res_testset in  self.data.items():
            for ws,subset  in res_testset.items():
                if idx < len(subset):
                    return subset[idx]
                else:
                    idx -= len(subset)
        raise IndexError

    async def run(
        self,
        completion_func: Callable[[LocaliationTestItem], Coroutine[Any,Any,Position | list[int, int, int, int]]],
        save_path:str = None
    ):
        statics = []
        subset_idx = 0
        for res, res_testset in  self.data.items():
            for ws,subset  in res_testset.items():
                subset_idx+=1
                meta = {
                    "Windows Ratio": ws,
                    "Resolution": res,
                }
                tasks = [asyncio.create_task(completion_func(
                    item), name=str(idx)) for idx, item in enumerate(subset)]


                test = []
                labels = []
                pbar = tqdm(total=len(tasks),ncols=150)
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                while pending:
                    for item in done:
                        try:
                            ret = await item
                        except Exception as e:
                            print(f"Error: {e}")
                            continue
                        idx = int(item.get_name())
                        pbar.update(1)
                        if idx < len(subset):
                            t = subset[idx]
                            test.append(t)
                            if isinstance(ret, Position):
                                labels.append(t.validate_point(ret))
                            else:
                                labels.append(t.validate_bbox(ret))
                        else:
                            print(f"Unkown index: {idx}")
                    done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                

                meta["Discernible"] = sum(labels)/len(labels) > 0.5
                statics.append({
                    "meta": meta,
                    "results": {
                        "test":test,
                        "labels": labels
                    }
                })
                print(meta)
                draw_res_correctness(statics[-1], save_path=os.path.join(save_path,f"{subset_idx}.png"))
                for idx,t in enumerate(statics[-1]["results"]["test"]):
                    statics[-1]["results"]["test"][idx] = t.model_dump()
                json.dump(statics[-1],open(os.path.join(save_path,f"{subset_idx}.json"),"w"))
        return statics
    

def draw_res_correctness(statics, sigma:float = None, save_path:str = None):
    res_x,res_y = statics["meta"]["Resolution"]
    ws_ratio:float = statics["meta"]["Windows Ratio"]
    x = []
    y = []

    for t,label in zip(statics["results"]["test"],statics["results"]["labels"]):
        t:LocaliationTestItem
        x.append(t.center.x)
        y.append(t.center.y)
    
    x=  np.array(x)
    y = np.array(y)
    labels = np.array(statics["results"]["labels"])

    windows_area = int(ws_ratio*res_x*res_y)
    minimum_square_len = math.sqrt(windows_area)
    fig_res_x = math.ceil(res_x/minimum_square_len)
    fig_res_y = math.ceil(res_y/minimum_square_len)

    
    x_bins = np.linspace(0,res_x,fig_res_x+1)
    y_bins = np.linspace(0,res_y,fig_res_y+1)

    x_inds = np.digitize(x,x_bins) - 1
    y_inds = np.digitize(y,y_bins) - 1

    # 初始化计数数组
    true_counts = np.zeros((fig_res_x, fig_res_y))
    total_counts = np.zeros((fig_res_x, fig_res_y))

    # 计算每个网格单元内的 True 和总数
    for xi, yi, label in zip(x_inds, y_inds, labels):
        # 确保索引在合理范围内
        if 0 <= xi < res_x and 0 <= yi < res_y:
            total_counts[xi, yi] += 1
            if label:
                true_counts[xi, yi] += 1

    # 计算正确率：True 的比例
    accuracy = np.divide(
        true_counts, 
        total_counts, 
        out=np.zeros_like(true_counts),  # 避免除以零
        where=total_counts != 0
    )
    if sigma is not None:
        accuracy = gaussian_filter(accuracy, sigma=sigma)

    accuracy[total_counts==0] = np.nan

    plt.figure(figsize=(8,8*(res_y/res_x)+0.5))
    cmap = plt.get_cmap('RdYlGn')
    img = plt.imshow(
        accuracy.T,  # 转置以匹配 x 和 y 轴
        origin='lower',       # 原点在左下角
        extent=(0, res_x, 0, res_y),  # 定义坐标范围
        cmap=cmap,
        vmin=0, vmax=1,        # 正确率范围从0到1
        aspect='auto'          # 自动调整纵横比
    )

    # 绘制图例
    plt.colorbar(img, orientation='vertical')

    plt.gca().set_aspect('equal', adjustable='box')  # 设置纵横比为1:1
    plt.title(f'Windows Ratio: {ws_ratio}\nResolution: {res_x}x{res_y}')

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()