import colour
from colour.plotting import *
colour_style()
import numpy as np
import random
# 1  colour-science函数画出CIE1976UCS色域图
plot_chromaticity_diagram_CIE1976UCS(standalone=False)

import matplotlib.pyplot as plt
import matplotlib.patches as patches

ax = plt.gca()       # 获取CIE1976UCS的坐标系
# 2  将CIE1931 xy坐标转换成CIE1976 u'v'坐标
# 坐标转换函数
# colour.xy_to_Luv_uv        convert  CIE 1931 xy to CIE 1976UCS u'v'
# colour.xy_to_UCS_uv       convert CIE 1931 xy to CIE 1960UCS uv            
#  xy: CIE 1931 xy;    UCS_uv: CIE 1960UCS uv;  Luv_uv: CIE 1976 u'v'
ITUR_709=([[.64, .33], [.3, .6], [.15, .06]])
ITUR_709_uv=colour.xy_to_Luv_uv(ITUR_709)
ITUR_2020=([[.708, .292], [.170, .797], [.131, .046]])
ITUR_2020_uv=colour.xy_to_Luv_uv(ITUR_2020)
DCI_P3=([[.68, .32], [.265, .69], [.15, .06]])
DCI_P3_uv=colour.xy_to_Luv_uv(DCI_P3)
pointer_bound= ([[ 0.508, 0.226], [ 0.538, 0.258], [ 0.588, 0.280], [ 0.637, 0.298], [ 0.659, 0.316], 
                                      [ 0.634, 0.351], [ 0.594, 0.391], [ 0.557, 0.427], [ 0.523, 0.462], [ 0.482, 0.491], 
                                      [ 0.444, 0.515], [ 0.409, 0.546], [ 0.371, 0.558], [ 0.332, 0.573], [ 0.288, 0.584], 
                                      [ 0.242, 0.576], [ 0.202, 0.530 ], [ 0.177, 0.454], [ 0.151, 0.389],[ 0.151, 0.330 ],
                                      [ 0.162, 0.295], [ 0.157, 0.266], [ 0.159, 0.245], [ 0.142, 0.214], [ 0.141, 0.195], 
                                      [ 0.129, 0.168], [ 0.138, 0.141], [ 0.145, 0.129], [ 0.145, 0.106], [ 0.161, 0.094], 
                                      [ 0.188, 0.084], [ 0.252, 0.104], [ 0.324, 0.127], [ 0.393, 0.165], [ 0.451, 0.199], [ 0.508, 0.226]])
pointer_bound_uv=colour.xy_to_Luv_uv(pointer_bound)
# 3 matplotlib绘制ITU-R BT.709，ITU-R BT.2020，DCI-P3色域空间多边形，以及pointer 色域空间
gamut_709=patches.Polygon(ITUR_709_uv, linewidth=2, color='green', fill=False)
gamut_2020=patches.Polygon(ITUR_2020_uv, linewidth=2, color='yellow', fill=False)
gamut_DCI_P3=patches.Polygon(DCI_P3_uv, linewidth=1, color='blue', fill=False)
gamut_pointer=patches.Polygon(pointer_bound_uv, linewidth=2, color='white', fill=False)
ax.add_patch(gamut_709)
ax.add_patch(gamut_2020)
ax.add_patch(gamut_DCI_P3)
ax.add_patch(gamut_pointer)

# 4 附加曲线标注，修改坐标范围
plt.legend([gamut_709,gamut_2020, gamut_DCI_P3, gamut_pointer],
    ['ITU-R BT.709','ITU-R BT.2020', 'DCI-P3', 'pointer gamut'],
    loc='upper right')  # 对曲线的标注

# 给定的sRGB数据：列表包含[sRGB元组, 布尔值]
srgb_data = []
for _ in range(1000):
    color = tuple(random.randint(0, 255) for _ in range(3))
    srgb_data.append((color, False))
    
# 处理和过滤sRGB数据
false_colors_uv = []

def linearize_srgb(srgb):
    """
    将标准化的非线性sRGB值转换为线性RGB值。

    参数：
        srgb (array-like): 标准化后的sRGB值（范围0到1）。

    返回：
        array-like: 线性RGB值。
    """
    srgb = np.array(srgb)
    linear = np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        ((srgb + 0.055) / 1.055) ** 2.4
    )
    return linear

def xy_to_UCS_uv(xy):
    x = xy[:, 0]
    y = xy[:, 1]
    denominator = (-2 * x) + (12 * y) + 3
    u_prime = (4 * x) / denominator
    v_prime = (9 * y) / denominator
    return np.vstack((u_prime, v_prime)).T


for color, flag in srgb_data:
    # 修正超出范围的sRGB值
    color_corrected = tuple(np.clip(color, 0, 255))
    if not flag:
        # 将sRGB值标准化到[0, 1]范围
        srgb_normalized = [c / 255 for c in color_corrected]
        # # 线性化sRGB（假设sRGB值是非线性的，需要转换为线性空间）
        srgb_linear = linearize_srgb(srgb_normalized)
        # 转换到XYZ
        XYZ = colour.sRGB_to_XYZ(srgb_linear)
        xy = colour.XYZ_to_xy(XYZ)
        
        u_prime, v_prime = xy_to_UCS_uv(xy[np.newaxis, :])[0]
        false_colors_uv.append([u_prime, v_prime])


# 转换为NumPy数组以便绘制
false_colors_uv = np.array(false_colors_uv)

# 在图上绘制灰色点
ax.plot(false_colors_uv[:, 0], false_colors_uv[:, 1], 'o', color='gray', label='False Colors')


# 5 显示
plt.axis([-0.1, 0.7, -0.1, 0.7])    #改变坐标轴范围
plt.show()
