'''Here we testing the model's localization ability.

We consider multiple factors that could affect models' localization ability:
- The size of the object.
- The color of the object.
- The shape of the object.
- The background texture.
- The location of the object.
'''

import random
import math
import numpy as np
import asyncio
from tqdm import tqdm
from typing import List, Literal, Callable, Coroutine, Any

from mbvat.utils import COMMON_RESOLUTIONS,MBVATItem,CircleObject,RectangleObject,Position,Resolution,NoneObject,BasicObject,TriangleObject


def build_full_localization_test(
        resolutions: list[tuple[int, int]] = COMMON_RESOLUTIONS,
        windows_ratio: list[float] = [0.1, 0.05, 0.01, 0.005],
        max_repeat_times: int = 1,
        delta_size=0.2,
        colorful=False,
        shapes = [RectangleObject,CircleObject]
        ) -> dict[float, dict[tuple[int, int], list[MBVATItem]]]:
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
                objs_type = np.random.choice(shapes, size=len(shifted_points))
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
                    if otype == RectangleObject:
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

                    if otype == CircleObject:
                        # circle
                        max_radius = int(min(point[0],(width-point[0]),point[1],(height-point[1])))

                        if max_radius<=0:
                            continue
                            
                        radius = min(int((area/math.pi)**0.5),max_radius)
                        obj = CircleObject(radius=radius, color=c1, bbox=[
                               0, 0, 2*radius, 2*radius])

                    if otype == NoneObject:
                        obj = NoneObject(color=c1, bbox=[0, 0, 0, 0])
                    
                    if otype == TriangleObject:
                        # triangle
                        max_width = min(point[0]*2,2*(width-point[0]),area/4)
                        min_width = max(4,math.sqrt(0.2*area))
                        if min_width > max_width:
                            # print(min_width,max_width)
                            continue

                        # Calculate width and height with a more balanced approach
                        aspect_ratio = random.uniform(0.5, 2)
                        obj_width = int(math.sqrt(area / aspect_ratio))
                        obj_height = int(area / obj_width)
                        
                        # Ensure width and height are within bounds
                        obj_width = max(int(min_width), min(int(max_width), obj_width))
                        obj_height = int(area / obj_width)
                        obj = TriangleObject(width=obj_width, height=obj_height, color=c1, bbox=[
                                            0, 0, obj_width, obj_height])
                    
                    # print(max(point[0]-obj.bbox[2]//2,0))
                    dataset[res][ws].append(MBVATItem(
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



class LocalizationTester:
    def __init__(
        self,
        resolutions: list[tuple[int, int]] = COMMON_RESOLUTIONS,
        windows_ratio: list[float] = [0.1, 0.05, 0.01, 0.005],
        colorful:bool=False,
        max_repeat_times: int = 1
    ):
        self.data = build_full_localization_test(
            resolutions=resolutions,
            windows_ratio=windows_ratio,
            max_repeat_times=max_repeat_times,
            colorful=colorful
        )
        self.resolutions = resolutions
        self.windows_ratio = windows_ratio
        self.colorful = colorful
        self.max_repeat_times = max_repeat_times

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
        completion_func: Callable[[MBVATItem], Coroutine[Any,Any,Position | list[int, int, int, int]]],
    ):

        for res, res_testset in  self.data.items():
            for ws,subset  in res_testset.items():
                meta = {
                    "Windows Ratio": ws,
                    "Resolution": res,
                    "Colorful": self.colorful
                }
                error_counts = {}
                tasks = [asyncio.create_task(
                    completion_func(
                    item), name=str(idx)) for idx, item in enumerate(subset)]

                test = []
                labels = []
                preds = []
                pbar = tqdm(total=len(tasks), ncols=150)
                timeout_counter = 0
                timeout_limit = 3  # Set the limit for timeout occurrences

                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED, timeout=15)
                while pending or done:
                    if not done:
                        timeout_counter += 1
                        if timeout_counter >= timeout_limit:
                            print("Timeout limit reached, exiting loop.")
                            break
                    else:
                        timeout_counter = 0  # Reset counter if tasks are done

                    for item in done:
                        try:
                            ret = await item
                        except Exception as e:
                            error_counts[type(e)] = error_counts.get(type(e), 0) + 1
                            print(f"Error: {e}")
                            continue
                        idx = int(item.get_name())
                        pbar.update(1)
                        if idx < len(subset):
                            t = subset[idx]
                            test.append(t)
                            if isinstance(ret, Position):
                                labels.append(t.validate_point(ret))
                                preds.append([ret.x, ret.y])
                            else:
                                labels.append(t.validate_bbox(ret))
                                preds.append(ret)
                        else:
                            print(f"Unknown index: {idx}")
                    if not pending:
                        break
                    done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED, timeout=15)

                meta["Discernible"] = sum(labels)/(len(labels)+1e-6)> 0.5
                
                yield {
                    "meta": meta,
                    "results": {
                        "test":test,
                        "labels": labels,
                        "preds": preds
                    }
                }



class ColorTester:
    def __init__(
        self,
        resolutions: list[tuple[int, int]] = COMMON_RESOLUTIONS,
        windows_ratio: list[float] = [0.1, 0.05, 0.01, 0.005],
        max_repeat_times: int = 1
    ):
        self.data = build_full_localization_test(
            resolutions=resolutions,
            windows_ratio=windows_ratio,
            max_repeat_times=max_repeat_times,
            colorful=True,
            shapes=[NoneObject,RectangleObject,CircleObject,TriangleObject]
        )
        self.resolutions = resolutions
        self.windows_ratio = windows_ratio
        self.max_repeat_times = max_repeat_times

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
        completion_func: Callable[[MBVATItem], Coroutine[Any,Any,Literal["rectangle","circle"]]],
    ):

        for res, res_testset in  self.data.items():
            for ws,subset  in res_testset.items():
                meta = {
                    "Windows Ratio": ws,
                    "Resolution": res,
                    "Colorful": True
                }
                error_counts = {}
                tasks = [asyncio.create_task(
                    completion_func(
                    item), name=str(idx)) for idx, item in enumerate(subset)]

                test = []
                labels = []
                preds = []
                pbar = tqdm(total=len(tasks), ncols=150)
                timeout_counter = 0
                timeout_limit = 3  # Set the limit for timeout occurrences

                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED, timeout=15)
                while pending or done:
                    if not done:
                        timeout_counter += 1
                        if timeout_counter >= timeout_limit:
                            print("Timeout limit reached, exiting loop.")
                            break
                    else:
                        timeout_counter = 0  # Reset counter if tasks are done

                    for item in done:
                        try:
                            ret = await item
                        except Exception as e:
                            error_counts[type(e)] = error_counts.get(type(e), 0) + 1
                            print(f"Error: {e}")
                            continue
                        idx = int(item.get_name())
                        pbar.update(1)
                        if idx < len(subset):
                            t = subset[idx]
                            test.append(t)
                            labels.append(t.obj.shape == ret)
                            preds.append(ret)
                            
                        else:
                            print(f"Unknown index: {idx}")
                    if not pending:
                        break
                    done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED, timeout=15)

                meta["Discernible"] = sum(labels)/(len(labels)+1e-6)> 0.5
                
                yield {
                    "meta": meta,
                    "results": {
                        "test":test,
                        "labels": labels,
                        "preds": preds
                    }
                }
    