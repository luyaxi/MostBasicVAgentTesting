import colour.plotting
import numpy as np
import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from scipy.ndimage import zoom, gaussian_filter

import colour
from colour.plotting import colour_style, plot_chromaticity_diagram_CIE1976UCS

from mbvat.utils import MBVATItem, Position, cross_delta_e_cie2000


def draw_res_correctness(
    statics,
    sigma: float = None,
    save_path: str = None,
    upscale=1  # Default to 4x resolution increase
):
    res_x, res_y = statics["meta"]["Resolution"]
    ws_ratio: float = statics["meta"]["Windows Ratio"]

    if res_x >= res_y:
        plt.figure(figsize=(10, 8*(res_y/res_x)))
    else:
        plt.figure(figsize=(10, 8*(res_y/res_x)))

    x = []
    y = []

    for idx, (t, label) in enumerate(zip(statics["results"]["test"], statics["results"]["labels"])):
        if not isinstance(t, MBVATItem):
            t = MBVATItem(**t)
            statics["results"]["test"][idx] = t
        x.append(t.center.x)
        y.append(t.center.y)

    x = np.array(x)
    y = np.array(y)
    labels = np.array(statics["results"]["labels"])

    windows_area = int(ws_ratio*res_x*res_y)
    minimum_square_len = math.sqrt(windows_area)
    fig_res_x = math.ceil(res_x/minimum_square_len)
    fig_res_y = math.ceil(res_y/minimum_square_len)

    x_bins = np.linspace(0, res_x, fig_res_x+1)
    y_bins = np.linspace(0, res_y, fig_res_y+1)

    x_inds = np.digitize(x, x_bins) - 1
    y_inds = np.digitize(y, y_bins) - 1

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
    for xj in range(len(x_bins)-1):
        for yj in range(len(y_bins)-1):
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

            plt.text(x_center, y_center, f'{min_delta_e:0.1f}',
                     ha='center', va='center', fontsize=8, color='black')

    # 计算正确率：True 的比例
    accuracy = np.divide(
        true_counts,
        total_counts,
        out=np.zeros_like(true_counts),  # 避免除以零
        where=total_counts != 0
    )
    if sigma is None and upscale > 1:
        sigma = upscale / 2  # Add appropriate smoothing
    if upscale > 1:
        accuracy = zoom(accuracy, zoom=upscale, order=3)
    else:
        accuracy[total_counts == 0] = np.nan

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
    plt.title(
        f'Windows Ratio: {ws_ratio} Resolution: {res_x}x{res_y} Correctness: {sum(labels)}/{len(labels)}')
    # 去掉网格线
    plt.grid(False)

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()


def merge_similar_colors(
    colors: np.ndarray,
    colors_similarity: np.ndarray,
    colors_labels: np.ndarray,
    threshold: float,
):
    merged_colors = []
    merged_colors_size = []
    merged_colors_labels = []
    similar_counts = np.sum(colors_similarity < threshold, axis=-1)
    while True:
        # merge the most similar color
        if np.max(similar_counts) < 2:
            break
        max_idx = np.argmax(similar_counts)
        similar_counts[max_idx] = -1
        similar_indices = np.where(colors_similarity[max_idx] < threshold)[0]
        merged_colors.append(np.mean(colors[similar_indices], axis=0))
        merged_colors_labels.append(
            np.mean(colors_labels[similar_indices], axis=0))
        merged_colors_size.append(len(similar_indices))

    for idx in np.where(similar_counts >= 0)[0]:
        merged_colors.append(colors[idx])
        merged_colors_labels.append(colors_labels[idx])
        merged_colors_size.append(1)

    return np.asarray(merged_colors), np.array(merged_colors_labels), np.array(merged_colors_size)


def draw_colorspace_line(
    ax,
):
    # 1 将CIE1931 xy坐标转换成CIE1976 u'v'坐标
    # 坐标转换函数
    # colour.xy_to_Luv_uv        convert  CIE 1931 xy to CIE 1976UCS u'v'
    # colour.xy_to_UCS_uv       convert CIE 1931 xy to CIE 1960UCS uv
    #  xy: CIE 1931 xy;    UCS_uv: CIE 1960UCS uv;  Luv_uv: CIE 1976 u'v'

    ITUR_709 = ([[.64, .33], [.3, .6], [.15, .06]])
    ITUR_709_uv = colour.xy_to_Luv_uv(ITUR_709)
    ITUR_2020 = ([[.708, .292], [.170, .797], [.131, .046]])
    ITUR_2020_uv = colour.xy_to_Luv_uv(ITUR_2020)
    DCI_P3 = ([[.68, .32], [.265, .69], [.15, .06]])
    DCI_P3_uv = colour.xy_to_Luv_uv(DCI_P3)
    # pointer_bound = ([[0.508, 0.226], [0.538, 0.258], [0.588, 0.280], [0.637, 0.298], [0.659, 0.316],
    #                   [0.634, 0.351], [0.594, 0.391], [0.557, 0.427], [
    #                       0.523, 0.462], [0.482, 0.491],
    #                   [0.444, 0.515], [0.409, 0.546], [0.371, 0.558], [
    #                       0.332, 0.573], [0.288, 0.584],
    #                   [0.242, 0.576], [0.202, 0.530], [0.177, 0.454], [
    #                       0.151, 0.389], [0.151, 0.330],
    #                   [0.162, 0.295], [0.157, 0.266], [0.159, 0.245], [
    #                       0.142, 0.214], [0.141, 0.195],
    #                   [0.129, 0.168], [0.138, 0.141], [0.145, 0.129], [
    #                       0.145, 0.106], [0.161, 0.094],
    #                   [0.188, 0.084], [0.252, 0.104], [0.324, 0.127], [0.393, 0.165], [0.451, 0.199], [0.508, 0.226]])
    # pointer_bound_uv = colour.xy_to_Luv_uv(pointer_bound)

    # 2 matplotlib绘制ITU-R BT.709，ITU-R BT.2020，DCI-P3色域空间多边形，以及pointer 色域空间
    gamut_709 = patches.Polygon(
        ITUR_709_uv, linewidth=2, color='green', fill=False)
    gamut_2020 = patches.Polygon(
        ITUR_2020_uv, linewidth=2, color='yellow', fill=False)
    gamut_DCI_P3 = patches.Polygon(
        DCI_P3_uv, linewidth=1, color='blue', fill=False)
    # gamut_pointer = patches.Polygon(
    #     pointer_bound_uv, linewidth=2, color='white', fill=False)
    ax.add_patch(gamut_709)
    ax.add_patch(gamut_2020)
    ax.add_patch(gamut_DCI_P3)
    # ax.add_patch(gamut_pointer)

    # 4 附加曲线标注，修改坐标范围
    plt.legend([
        gamut_709,
        gamut_2020,
        gamut_DCI_P3,
        # gamut_pointer
    ],
        ['ITU-R BT.709',
         'ITU-R BT.2020',
         'DCI-P3',
         #  'pointer gamut'
         ],
        loc='upper right')  # 对曲线的标注


def draw_color_coverage(
    statics,
    save_path: str = None,
    similarity_threshold: float = 10,
    resolution: int = 1024,
    inverse: bool = True
):
    # colour-science函数画出CIE1976UCS色域图
    colour_style()
    fig, ax = plot_chromaticity_diagram_CIE1976UCS(show=False)

    # 读取Static颜色数据
    res_x, res_y = statics["meta"]["Resolution"]
    ws_ratio: float = statics["meta"]["Windows Ratio"]
    labels = np.array(statics["results"]["labels"])
    fg_colors = []
    bg_colors = []

    for idx, t in enumerate(statics["results"]["test"]):
        if not isinstance(t, MBVATItem):
            t = MBVATItem(**t)
            statics["results"]["test"][idx] = t
        fg_colors.append(list(t.obj.color))
        bg_colors.append(list(t.background))

    bg_colors = np.array(bg_colors)
    fg_colors = np.array(fg_colors)

    fg_colors_corrected = np.clip(fg_colors, 0, 255)
    fg_colors_normalized = fg_colors_corrected/255
    fg_colors_linear = np.where(
        fg_colors_normalized <= 0.04045,
        fg_colors_normalized / 12.92,
        ((fg_colors_normalized + 0.055) / 1.055) ** 2.4
    )
    fg_colors_XYZ = colour.sRGB_to_XYZ(fg_colors_linear)
    fg_colors_lab = colour.XYZ_to_Lab(fg_colors_XYZ)

    fg_colors_uv = colour.xy_to_Luv_uv(colour.XYZ_to_xy(fg_colors_XYZ))

    # print("calculating color similarity matrix")
    
    color_similarity_matrix = cross_delta_e_cie2000(
        fg_colors_lab, fg_colors_lab)
    # merge the similar colors first
    # print("merging similar color points")
    fg_colors_uv_merged, fg_colors_labels, fg_colors_count = merge_similar_colors(
        colors=fg_colors_uv,
        colors_similarity=color_similarity_matrix,
        colors_labels=labels,
        threshold=similarity_threshold
    )
    # print("Merged Points:", len(fg_colors_uv_merged))

    # colour.plotting.plot_RGB_chromaticities_in_chromaticity_diagram_CIE1976UCS
    # should better than random guess
    similar_success = fg_colors_labels > (
        1/len(statics["meta"].get("Tested Shapes", [1, 2])))

    passed_colors_uv = fg_colors_uv_merged[similar_success]
    passed_colors_count = fg_colors_count[similar_success]
    failed_colors_uv = fg_colors_uv_merged[~similar_success]
    failed_colors_count = fg_colors_count[~similar_success]

    x = np.linspace(-0.1, 0.7, resolution)
    y = np.linspace(-0.1, 0.7, resolution)
    X, Y = np.meshgrid(x, y)

    # print("calculating pass masks")
    radius = 0.002
    valid_masks = np.zeros((resolution, resolution))
    invalid_masks = np.zeros((resolution, resolution))
    for passed_color_uv, passed_color_count in zip(passed_colors_uv, passed_colors_count):
        valid_masks += ((X - passed_color_uv[0])**2 +
                        (Y - passed_color_uv[1])**2) <= np.sqrt(passed_color_count)*(radius**2)
        
    # print("calculating fail masks")
    for failed_color_uv, failed_color_count in zip(failed_colors_uv, failed_colors_count):
        invalid_masks += ((X - failed_color_uv[0])**2 +
                          (Y - failed_color_uv[1])**2) <= np.sqrt(failed_color_count)*(radius**2)

    # print("calculating mean masks")
    # 综合判定范围内是否有效，若有效范围被无效范围覆盖，则进行加权平均;0为完全无效，1为完全有效
    mean_masks = np.zeros((resolution, resolution))
    mean_masks[(valid_masks > 0) & (invalid_masks > 0)] = (valid_masks / (valid_masks+invalid_masks+1e-6))[(valid_masks > 0) & (invalid_masks > 0)]
    mean_masks[(valid_masks > 0) & (invalid_masks == 0)] = 1
    mean_masks[(valid_masks == 0) & (invalid_masks > 0)] = 0
    
    if inverse:
        # print("build gray background")
        gray_bg = np.zeros((resolution, resolution, 4))
        gray_bg[..., 3] = 1
        # dig gray holes in the background
        gray_bg[..., 3][mean_masks > 0] = 1 - mean_masks[mean_masks > 0]
        ax.imshow(gray_bg, extent=(-0.1, 0.7, -0.1, 0.7),
                alpha=0.9, origin='lower',)

    else:
        gray_fg = np.zeros((resolution, resolution, 4))
        gray_fg[..., 3] = 0
        gray_fg[..., 3][mean_masks > 0] = mean_masks[mean_masks > 0]
        ax.imshow(gray_fg, extent=(-0.1, 0.7, -0.1, 0.7),
                alpha=0.7, origin='lower',)

    # 显示
    draw_colorspace_line(ax)

    plt.title(
        f'Windows Ratio: {ws_ratio} Resolution: {res_x}x{res_y} Coverage: {sum(labels)}/{len(labels)}')
    plt.axis([-0.1, 0.7, -0.1, 0.7])
    ax.grid(False)
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()
