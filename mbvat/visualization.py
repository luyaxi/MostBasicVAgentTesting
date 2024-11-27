import numpy as np
import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from scipy.ndimage import zoom,gaussian_filter

import colour
from colour.plotting import colour_style,plot_chromaticity_diagram_CIE1976UCS

from mbvat.utils import MBVATItem,Position,calculate_color_delta



def draw_res_correctness(
    statics, 
    sigma:float = None, 
    save_path:str = None,
    super_res_factor = 1  # Default to 4x resolution increase
    ):
    res_x,res_y = statics["meta"]["Resolution"]
    ws_ratio:float = statics["meta"]["Windows Ratio"]

    if res_x >= res_y:
        plt.figure(figsize=(10,8*(res_y/res_x)))
    else:
        plt.figure(figsize=(10,8*(res_y/res_x)))
    
    
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

    # 获取每个单元内最小delta e并绘制在图上
    for xj in range(max(x_inds)+1):
        for yj in range(max(y_inds)+1):
            x_center = (x_bins[xj] + x_bins[xj+1]) / 2
            y_center = (y_bins[yj] + y_bins[yj+1]) / 2
            
            # obtain the minimum delta e in the window
            window_colors = []
            box_elems = np.logical_and(x_inds == xj, y_inds == yj)
            for idx in np.where(box_elems)[0]:
                item = statics["results"]["test"][idx]
                window_colors.append(item.delta_e)
            
            if len(window_colors) == 0:
                continue
            min_delta_e = np.min(window_colors)
            
            plt.text(x_center, y_center, f'{min_delta_e:0.1f}', ha='center', va='center', fontsize=8, color='black')


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
    plt.title(f'Windows Ratio: {ws_ratio} Resolution: {res_x}x{res_y} Correctness: {sum(labels)}/{len(labels)}')
    # 去掉网格线
    plt.grid(False)

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
        
def draw_color_coverage(
    statics,    
    save_path:str = None,
    plot_success:bool = False,
    similarity_threshold:float = 10
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
    

    if plot_success:
        passed_indices = labels  # Adjust based on your actual label structure
        # Step 6b: Add a semi-transparent grey overlay covering the entire plot
        grey_mask = patches.Rectangle(
            (-0.1, -0.1),  # (x,y) of the lower left corner
            0.8,  # width
            0.8,  # height
            facecolor='grey',
            alpha=0.7,
            zorder=100  # Ensure it's on top of all other plots
        )
        ax.add_patch(grey_mask)

        
        # Plot passed foreground colors with distinct color to highlight them
        ax.scatter(
            foreground_colors_uv[passed_indices, 0],
            foreground_colors_uv[passed_indices, 1],
            c=foreground_colors_normalized[passed_indices],  # Use CIE colors for coloring
            edgecolors=foreground_colors_normalized[passed_indices],
            s=15,
            alpha=1,
            label='Passed Foreground Colors',
            zorder=1000  # Ensure it's above everything
        )
    
    else:
        # only be considered as failed if all similiar color are failed
        failed_indices = np.logical_not(labels)
        
        # color_similarities = np.zeros((len(foreground_colors),len(foreground_colors)))
        # for i in range(len(foreground_colors_normalized)):
        #     for j in range(i+1,len(foreground_colors_normalized)):
        #         color_similarities[i][j] = calculate_color_delta(foreground_colors_normalized[i],foreground_colors_normalized[j])
        #         color_similarities[j][i] = color_similarities[i][j]
        
        # for i in range(len(failed_indices)):
        #     if failed_indices[i]:
        #         # check all similar color
        #         for j in np.where(color_similarities[i] < similarity_threshold)[0]:
        #             if labels[j]:
        #                 failed_indices[i] = False
        #                 break
        
        
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

    # 7 显示
    plt.title(f'Windows Ratio: {ws_ratio} Resolution: {res_x}x{res_y} Coverage: {sum(labels)}/{len(labels)}')
    plt.axis([-0.1, 0.7, -0.1, 0.7])
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
        