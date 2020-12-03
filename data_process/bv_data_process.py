import numpy as np
import os


def SelectPointsNearGround(points, bv_common_settings):
    height_shift = bv_common_settings['train_height_shift']
    shifted_min_height = bv_common_settings['shifted_min_height']
    shifted_max_height = bv_common_settings['shifted_max_height']

    p_h = points[:, 2] + height_shift

    points1 = points[(p_h >= shifted_min_height) & (p_h <= shifted_max_height), :]

    return points1


def AdjustIntensity(points, bv_common_settings):
    truncation_max_intensiy= bv_common_settings["truncation_max_intensiy"]
    points[:, 3] = points[:, 3] / 255.0
    p_intensity = points[:, 3].copy()
    points[p_intensity > truncation_max_intensiy, 3] = truncation_max_intensiy
    points[:, 3] = points[:, 3] / truncation_max_intensiy


def ProduceBVData(points, bv_common_settings, bv_range_settings, if_square_dilate = True):

    height_shift = bv_common_settings["train_height_shift"]
    distance_resolution = bv_common_settings["distance_resolution_train"]
    width_resolution = bv_common_settings["width_resolution_train"]
    vis_point_radius = bv_common_settings["point_radius_train"]
    background_intensity_shift = bv_common_settings["train_background_intensity_shift"]

    max_distance = bv_range_settings["max_distance"]
    min_distance = bv_range_settings["min_distance"]
    left_distance = bv_range_settings["left_distance"]
    right_distance = bv_range_settings["right_distance"]

    bv_im_width = int((right_distance + left_distance)*width_resolution)
    bv_im_height = int((max_distance - min_distance) * distance_resolution)
    im_intensity = np.zeros((bv_im_height, bv_im_width, 1)).astype(np.float32)
    height_map = np.zeros((bv_im_height, bv_im_width, 1)).astype(np.float32)

    points[:, 2] = points[:, 2] + height_shift
    points[:, 3] = points[:, 3] + background_intensity_shift

    max_distance_in_pixel = max_distance * distance_resolution
    left_distance_in_pixel = left_distance * width_resolution

    point_num = points.shape[0]
    for i in range(point_num):
        x = points[i, 0]*distance_resolution
        y = points[i, 1]*width_resolution

        im_x = int(-y + left_distance_in_pixel)
        im_y = int(max_distance_in_pixel - x)

        if im_x >= 0 and im_x < bv_im_width and im_y >= 0 and im_y < bv_im_height:
            start_x = max(0, np.ceil(im_x - vis_point_radius))
            end_x = min(bv_im_width -1, np.ceil(im_x + vis_point_radius) - 1) + 1
            start_y = max(0, np.ceil(im_y - vis_point_radius))
            end_y = min(bv_im_height -1, np.ceil(im_y + vis_point_radius) - 1) + 1
            start_x = int(start_x)
            start_y = int(start_y)
            end_x = int(end_x)
            end_y = int(end_y)
            if if_square_dilate:
                im_intensity[start_y:end_y, start_x:end_x, 0] = np.maximum(points[i, 3], im_intensity[start_y:end_y, start_x:end_x, 0])
                height_map[start_y:end_y, start_x:end_x, 0] = np.maximum(points[i, 2], height_map[start_y:end_y, start_x:end_x, 0])

    concat_data = np.concatenate((im_intensity, height_map), axis=2)

    return concat_data


def GetPointsClassFromBV(points_input_set, bv_label_map, bv_common_settings, bv_range_settings):
    lidar_id_list = points_input_set.keys()

    height_shift = bv_common_settings['train_height_shift']
    shifted_min_height = bv_common_settings['shifted_min_height']
    shifted_max_height = bv_common_settings['shifted_max_height']

    distance_resolution = bv_common_settings["distance_resolution_train"]
    width_resolution = bv_common_settings["width_resolution_train"]

    max_distance = bv_range_settings["max_distance"]
    min_distance = bv_range_settings["min_distance"]
    left_distance = bv_range_settings["left_distance"]
    right_distance = bv_range_settings["right_distance"]
    max_distance_in_pixel = max_distance * distance_resolution
    left_distance_in_pixel = left_distance * width_resolution

    bv_im_width = int((right_distance + left_distance)*width_resolution)
    bv_im_height = int((max_distance - min_distance) * distance_resolution)

    points_class_set = {}
    for lidar_id in lidar_id_list:
        points = points_input_set[lidar_id]
        point_num = points.shape[0]
        point_classes = np.zeros([point_num, 1]).astype("float32")
        h = points[:, 2] + height_shift

        for i in range(point_num):
            x = points[i, 0]*distance_resolution
            y = points[i, 1]*width_resolution
            z = h[i]
            if (z < shifted_min_height) or (z > shifted_max_height):
                continue

            im_x = int(-y + left_distance_in_pixel)
            im_y = int(max_distance_in_pixel - x)

            if im_x >= 0 and im_x < bv_im_width and im_y >= 0 and im_y < bv_im_height:
                point_classes[i] = bv_label_map[im_y, im_x]

        points_class_set[lidar_id] = point_classes

    return points_class_set




