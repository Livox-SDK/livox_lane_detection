import numpy as np
import math
import sys
import cv2

sys.path.append("..")
from util.color_table import color_table
from util.color_table_for_class import color_table_for_class


def ProjectPointsToWorld(points_input_set, parameters):
    lidar_id_list = points_input_set.keys()
    points_output_set = {}

    for lidar_id in lidar_id_list:
        P_world = parameters[lidar_id]["P_world"]

        points0 = points_input_set[lidar_id][:, :4].copy()
        points0[:, 3] = 1
        points1 = P_world.dot(points0.T)
        points1 = points1.T

        points_output_set[lidar_id] = points_input_set[lidar_id].copy()
        points_output_set[lidar_id][:, :3] = points1[:, :3]

    return points_output_set


def MergePoints(points_input_set):
    lidar_id_list = points_input_set.keys()
    points_list = []

    for lidar_id in lidar_id_list:
        points_list.append(points_input_set[lidar_id])

    points_output = np.concatenate(points_list, axis = 0)

    return points_output


# R = RzRxRy
def GetWorldToCamMatrix(new_angle, T):
    angle_x, angle_y, angle_z = new_angle
    R_x = [[1,         0,                 0         ],
           [0, math.cos(angle_x), -math.sin(angle_x)],
           [0, math.sin(angle_x),  math.cos(angle_x)]]
    R_y = [[ math.cos(angle_y), 0, math.sin(angle_y)],
           [0,                  1,        0        ],
           [-math.sin(angle_y), 0, math.cos(angle_y)]]
    R_z = [[math.cos(angle_z), -math.sin(angle_z), 0],
           [math.sin(angle_z),  math.cos(angle_z), 0],
           [       0,                   0,         1]]
    R_x = np.array(R_x)
    R_y = np.array(R_y)
    R_z = np.array(R_z)
    R = np.dot(R_z, np.dot(R_x, R_y))
    P = np.zeros([4, 4])
    P[:3, :3] = R
    P[:3, 3] = T
    P[3, 3] = 1
    return P


def GetMatrices(img_shape):
    img_h, img_w = img_shape
    P_world = np.array([[0.999972   ,           0,  0.00750485,           0],
                        [0          ,           1,           0,           0],
                        [-0.00750485,           0,    0.999972,       1.909],
                        [0          ,           0,          0,           1]])

    K = np.array([[ 380.0,      0, float(img_w / 2), 0],
                  [     0,  380.0, float(img_h / 2), 0], 
                  [     0,      0,                1, 0]])

    #if you want to change the view point, you can change T and angle
    T = np.array([0, 0, 38]).T
    angle = [0, -math.pi*(16.0/18.0), math.pi/2]
    P = GetWorldToCamMatrix(angle, T)

    # print(K)
    # print(P)
    # print(P_world)  
    return K, P, P_world


def DrawPointOnImg(img, im_x, im_y, color):
    vis_point_radius = 1

    im_height, im_width = img.shape[:2]

    start_x = max(0, np.ceil(im_x - vis_point_radius))
    end_x = min(im_width -1, np.ceil(im_x + vis_point_radius) - 1) + 1
    start_y = max(0, np.ceil(im_y - vis_point_radius))
    end_y = min(im_height -1, np.ceil(im_y + vis_point_radius) - 1) + 1
    start_x = int(start_x)
    start_y = int(start_y)
    end_x = int(end_x)
    end_y = int(end_y)

    for yy in range(start_y, end_y):
        for xx in range(start_x, end_x):
            img[yy, xx, :] = color


def GetProjectImage(points, color_info, img_shape, K, P, P_world, color_info_table, draw_virtual_car=True):
    BLANK_COLOR = [10, 10, 10]

    img_h, img_w = img_shape

    N = points.shape[0]

    points_pos = np.concatenate([points[:, :3].copy(), np.ones([N, 1])], axis=1)
    points2 = np.dot(np.linalg.inv(P_world), np.transpose(points_pos))
    points3 = np.dot(P, points2)
    points_cam = np.dot(K, points3)

    img_show = (np.ones([img_h, img_w, 3])*BLANK_COLOR[0]).astype(np.uint8)

    for i in range(N):
        if points_cam[2, i] < 0:
            continue
        im_x = points_cam[0, i] / points_cam[2, i]
        im_y = points_cam[1, i] / points_cam[2, i]

        if 0 <= im_x < img_w and 0 <= im_y < img_h:
            color_info_cur = color_info[i]
            point_color = color_info_table[int(color_info_cur)]
            DrawPointOnImg(img_show, im_x, im_y, point_color)

    if draw_virtual_car:
        car_ctx = img_w / 2
        car_cty = img_h / 2 
        car_w = 10
        car_h = 15
        car_points = [(int(car_ctx-car_w), int(car_cty)),
                      (int(car_ctx-car_w/2), int(car_cty-car_h)),
                      (int(car_ctx+car_w/2), int(car_cty-car_h)),
                      (int(car_ctx+car_w), int(car_cty)), 
                      (int(car_ctx+car_w/2), int(car_cty+car_h)),
                      (int(car_ctx-car_w/2), int(car_cty+car_h))]
        img_class = cv2.fillPoly(img_show, [np.array(car_points)], [255, 255, 255])

    return img_show


def VisualizePointsClass(points_input):
    output_img_h = 1080
    output_img_w = 1920
    output_img_w1 = int(output_img_w / 2)

    K, P, P_world = GetMatrices([output_img_h, output_img_w1])

    intensity_show = GetProjectImage(points_input, 
                                     points_input[:, 3]*255, 
                                     [output_img_h, output_img_w1], 
                                     K, P, P_world, color_table)
    class_show = GetProjectImage(points_input, 
                                     points_input[:, 4], 
                                     [output_img_h, output_img_w1], 
                                     K, P, P_world, color_table_for_class)

    vis_img = np.concatenate([intensity_show, class_show], axis = 1)

    return vis_img