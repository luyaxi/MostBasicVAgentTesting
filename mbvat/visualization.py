import numpy as np
import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from scipy.ndimage import zoom,gaussian_filter

import colour
from colour.plotting import colour_style,plot_chromaticity_diagram_CIE1976UCS
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

from mbvat.utils import MBVATItem,Position


def calculate_color_delta(color1,color2):
    c1 = sRGBColor(color1[0],color1[1],color1[2])
    c2 = sRGBColor(color2[0],color2[1],color2[2])
    c1_lab = convert_color(c1,LabColor)
    c2_lab = convert_color(c2,LabColor)
    return delta_e_cie2000(c1_lab, c2_lab)


def draw_res_correctness(
    statics, 
    sigma:float = None, 
    save_path:str = None,
    super_res_factor = 1  # Default to 4x resolution increase
    ):
    res_x,res_y = statics["meta"]["Resolution"]
    ws_ratio:float = statics["meta"]["Windows Ratio"]
    x = []
    y = []

    for t,label in zip(statics["results"]["test"],statics["results"]["labels"]):
        if isinstance(t,MBVATItem):
            x.append(t.center.x)
            y.append(t.center.y)
        elif isinstance(t,dict):
            x.append(t["center"]["x"])
            y.append(t["center"]["y"])
    
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
    if sigma is None and super_res_factor > 1:
        sigma = super_res_factor / 2  # Add appropriate smoothing
    if super_res_factor > 1:
        accuracy = zoom(accuracy, zoom=super_res_factor, order=3)
    else:
        accuracy[total_counts==0] = np.nan
    
    if sigma is not None:
        accuracy = gaussian_filter(accuracy, sigma=sigma)


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
    plt.title(f'Windows Ratio: {ws_ratio}\nResolution: {res_x}x{res_y}\nCorrectness: {sum(labels)}/{len(labels)}')

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
        
def draw_color_coverage(
    statics,    
    save_path:str = None,
):
    # 1  colour-science函数画出CIE1976UCS色域图
    colour_style()
    plot_chromaticity_diagram_CIE1976UCS(show=False)
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

    # 5 读取Static颜色数据
    res_x,res_y = statics["meta"]["Resolution"]
    ws_ratio:float = statics["meta"]["Windows Ratio"]
    labels = np.array(statics["results"]["labels"])
    foreground_colors = []
    background_colors = []

    for t in statics["results"]["test"]:
        if isinstance(t,MBVATItem):
            foreground_colors.append(list(t.obj.color))
            background_colors.append(list(t.background))
        elif isinstance(t,dict):
            foreground_colors.append(t["obj"]["color"])
            background_colors.append(t["background"])
    
    background_colors = np.array(background_colors)
    foreground_colors = np.array(foreground_colors)
    
    foreground_colors_corrected = np.clip(foreground_colors,0,255)
    foreground_colors_normalized = foreground_colors_corrected/255
    foreground_colors_linear = np.where(
        foreground_colors_normalized <= 0.04045,
        foreground_colors_normalized / 12.92,
        ((foreground_colors_normalized + 0.055) / 1.055) ** 2.4
    )
    foreground_colors_XYZ = colour.sRGB_to_XYZ(foreground_colors_linear)
    foreground_colors_xy = colour.XYZ_to_xy(foreground_colors_XYZ)
    foreground_colors_uv = colour.xy_to_Luv_uv(foreground_colors_xy)
    
    # 6. Add a global grey mask and remove it where labels are True
    # Step 6a: Add a semi-transparent grey overlay covering the entire plot
    # grey_mask = patches.Rectangle(
    #     (-0.1, -0.1),  # (x,y) of the lower left corner
    #     0.8,  # width
    #     0.8,  # height
    #     facecolor='grey',
    #     alpha=0.7,
    #     zorder=100  # Ensure it's on top of all other plots
    # )
    # ax.add_patch(grey_mask)
    # Step 6b: Plot the 'True' labels on top to "remove" the grey mask in those areas
    # Assuming 'labels' is a boolean array where True indicates coverage
    passed_indices = labels  # Adjust based on your actual label structure
    failed_indices = ~labels

    # Plot failed foreground colors with grey color and some transparency
    ax.scatter(
        foreground_colors_uv[failed_indices, 0],
        foreground_colors_uv[failed_indices, 1],
        color='grey',
        alpha=0.3,
        s=30,
        label='Failed Foreground Colors',
        edgecolors='black',
        # zorder=2  # Ensure it's above the grey mask
    )
    
    # Plot passed foreground colors with distinct color to highlight them
    # ax.scatter(
    #     foreground_colors_uv[passed_indices, 0],
    #     foreground_colors_uv[passed_indices, 1],
    #     c=foreground_colors_normalized[passed_indices],  # Use CIE colors for coloring
    #     edgecolors=foreground_colors_normalized[passed_indices],
    #     s=15,
    #     alpha=1,
    #     label='Passed Foreground Colors',
    #     zorder=1000  # Ensure it's above everything
    # )
    
    # 7 显示
    plt.title(f'Windows Ratio: {ws_ratio}\nResolution: {res_x}x{res_y}\nCoverage: {sum(labels)}/{len(labels)}')
    plt.axis([-0.1, 0.7, -0.1, 0.7])
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
        