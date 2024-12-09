import datetime
import os
import napari
from PySide6.QtWidgets import QVBoxLayout, QWidget
from magicgui import magicgui
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.measure import regionprops, label
from CellCounter import get_binary_map, apply_opening, find_median_cell_size, apply_watershed

selem = 7
remove = 1000


def process_image(image_data, mode, viewer, sl=selem, rm=remove):
    """
    处理单张图像并返回其处理结果，且在每个处理步骤后将图像显示在napari的layers中。
    """
    # 1. 获取二值化图像
    binary_img = get_binary_map(image_data, mode)
    viewer.add_image(binary_img, name='Binary Image', colormap='gray', blending='additive')

    # 2. 应用开运算
    opened_image = apply_opening(binary_img, sl,rm)
    viewer.add_image(opened_image, name='Opened Image', colormap='gray', blending='additive')

    # 3. 标记图像
    labeled_image = label(opened_image)
    viewer.add_image(labeled_image, name='Labeled Image', colormap='tab20', blending='additive')

    # 4. 计算中位数细胞大小
    median_size = find_median_cell_size(labeled_image)

    # 5. 判断是否使用分水岭算法
    cell_number = len(np.unique(labeled_image)) - 1
    final_image = apply_watershed(labeled_image, median_size) if cell_number > 150 else labeled_image

    if cell_number > 150:
        # 6. 显示分水岭处理后的图像（如果使用了分水岭）
        viewer.add_image(final_image, name='Final Image (Watershed)', colormap='viridis', blending='additive')

    # 返回处理后的图像以及相关数据
    return {
        'processed_image': final_image,
        'cell_count': cell_number,
        'median_size': median_size
    }


def extract_visualization_info(final_image, median_size):
    """
    从标记图像中提取用于可视化展示的细胞信息，包括中心点坐标、颜色以及边界框信息。
    """
    points = []
    colors = []
    bboxes = []
    for region in regionprops(final_image):
        y, x = region.centroid
        points.append([y, x])
        if region.area >= 2 * median_size:
            minr, minc, maxr, maxc = region.bbox
            bbox_rect = np.array([[minr, minc], [maxr, minc], [maxr, maxc], [minr, maxc]])
            colors.append('lime')  # 大细胞用绿
            bboxes.append(bbox_rect)
        elif region.area < median_size / 2:
            colors.append('magenta')  # 小细胞用品红
        else:
            colors.append('cyan')  # 中等大小的细胞用青色
    return np.array(points), colors, np.array(bboxes)


def add_visualization_layers(viewer, points, colors, bboxes):
    """
    在napari查看器中添加用于展示细胞信息的点图层和形状（边界框）图层。

    参数:
    viewer (napari.Viewer): napari查看器对象。
    points (numpy.ndarray): 细胞中心点坐标数组。
    colors (list): 细胞对应的颜色列表。
    bboxes (numpy.ndarray): 细胞边界框坐标数组。
    """
    if len(bboxes) > 0:
        viewer.add_shapes(bboxes,
                          face_color='transparent',
                          edge_color='magenta',
                          name='Bounding Boxes',
                          edge_width=5)
    if len(points) > 0:
        viewer.add_points(points,
                          face_color=colors,
                          size=20,
                          name='Cell Points')


filepath = None
df_results = pd.DataFrame(columns=['name', 'cell_count', 'median_size'])  # Initialize df_results


def save_results_to_excel(df, file_path=filepath):
    """
    保存处理结果到 Excel 文件，允许用户选择保存路径。
    """
    if not file_path:
        # 如果未提供文件路径，基于当前时间生成默认文件名
        current_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
        file_path = f"CountData/At_{current_time}.xlsx"

    try:
        if not os.path.exists(file_path):
            with pd.ExcelWriter(file_path, mode='w', engine='openpyxl') as writer:
                df.to_excel(writer, index=False, header=True)
            print(f'新文件已创建并保存结果到 {file_path}！')
        else:
            with pd.ExcelWriter(file_path, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
                existing_df = pd.read_excel(file_path)
                df.to_excel(writer, index=False, header=False, startrow=len(existing_df) + 1)
            print(f'结果已追加到现有文件 {file_path}！')

    except PermissionError:
        print(f"没有权限将结果保存到 {file_path}，请检查文件路径的权限设置或选择其他保存路径。")
    except Exception as e:
        print(f"保存结果到Excel文件时出现未知错误: {e}")


def napari_gui_interaction():
    viewer = napari.Viewer()
    viewer.window.resize(1920, 1080)

    @magicgui(call_button="Confirm",
              image_path={"label": "Choose Image",
                          "widget_type": "FileEdit",
                          "tooltip": "Select an image file for preprocessing"})
    def start_preprocessing(image_path):
        if not image_path:
            print("请输入有效的图像路径！")
            return
        try:
            # Clear all layers
            viewer.layers.clear()

            # 只处理单张图像
            if os.path.isfile(image_path):
                img = imread(image_path)
                viewer.add_image(img, name='Image')
                print(f"预处理的图像：{image_path}")
            else:
                print("请选择有效的图片文件！")

        except Exception as e:
            print(f"图像加载时出现错误: {e}")

    @magicgui(call_button="Set Value",
              slider1={"label": "Set selem_para.", "widget_type": "Slider", "min": 1, "max": 14, "value": 7},
              slider2={"label": "Set a remove_objects#", "widget_type": "Slider", "min": 100, "max": 5000, "value": 1000})
    def set_value(slider1: int, slider2: int):
        global selem, remove
        selem = slider1
        remove = slider2

    @magicgui(call_button="Start Processing",
              mode={"label": "Choose Binary Mode ('b' for default)", "widget_type": "ComboBox",
                    "choices": ["b", "r", "g"], "value": "b"},
              nm={"label": "Give a name", "widget_type": "Text"}
              )
    def set_mode(mode, nm):
        try:
            image_data = viewer.layers['Image'].data  # 获取预处理图层的数据

            # 假设 process_image 返回一个包含处理结果的字典
            results = process_image(image_data, mode, viewer)

            # 获取处理后的图像和细胞信息
            final_image = results['processed_image']
            viewer.add_image(final_image, name='Result')
            median_size = results['median_size']

            # 提取细胞的可视化信息
            points, colors, bboxes = extract_visualization_info(final_image, median_size)

            # 添加细胞的图层
            add_visualization_layers(viewer, points, colors, bboxes)

            if not nm:
                nm = f"At {datetime.datetime.now().strftime('%H%M%S')}"
            # 创建新的行数据
            new_row = pd.DataFrame([[nm, results['cell_count'], results['median_size']]],
                                   columns=['name', 'cell_count', 'median_size'])

            # 将新行添加到 df_results 中
            global df_results  # Ensure we modify the global df_results
            df_results = pd.concat([df_results, new_row], ignore_index=True)

        except Exception as e:
            print(f"图像处理时出错: {e}")

    @magicgui(call_button="Save Results to Excel", )
    def save_results():
        global filepath
        save_results_to_excel(df_results, filepath)
        viewer.status = "Results have been successfully saved!"

    @magicgui(call_button="Screenshot")
    def save_screenshot():
        viewer.screenshot(f'images/SSs/Screenshot{datetime.datetime.now().strftime('%H%M%S')}.png')
        viewer.status = "Screenshot have been successfully saved!"

    @magicgui(call_button="Save Current Layers")
    def save_current_layer():
        viewer.export_figure(f'images/CLs/Layer{datetime.datetime.now().strftime('%H%M%S')}.png')
        viewer.status = "Current Layer have been successfully saved!"

    # Create custom widget layout for better control
    container = QWidget()
    layout = QVBoxLayout()

    # Add widgets to the layout manually
    layout.addWidget(start_preprocessing.native)
    layout.addWidget(set_value.native)
    layout.addWidget(set_mode.native)
    layout.addWidget(save_screenshot.native)
    layout.addWidget(save_current_layer.native)
    layout.addWidget(save_results.native)

    layout.setSpacing(30)  # Adjust the space between widgets
    container.setLayout(layout)
    viewer.window.add_dock_widget(container, name='Controls', area='right')

    napari.run()


if __name__ == '__main__':
    napari_gui_interaction()
