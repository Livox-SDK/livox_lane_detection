import os
import glob
import numpy as np
import shutil
import cv2

def GetListFormTxt(list_file):
    result = []
    if not os.path.exists(list_file):
        return result
    f = open(list_file, 'r')
    result = [line.strip() for line in f.readlines()]
    f.close()
    return result

def ReadPcdFile(file_name):
    f = open(file_name, "r")
    lines = [line.strip().split(" ") for line in f.readlines()]
    lines = lines[11:]

    point_dim = len(lines[0])
    points = np.array(lines).astype("float32").reshape(-1, point_dim)

    return points

def GetFullPath(path_split):
    if len(path_split) == 0:
        return ""
    path = "/" + path_split[0]
    for i in range(1, len(path_split)):
        path = path + "/" + path_split[i]
    return path 

def WritePointsToFile(points, filename):
    f = open(filename, "w")
    N, dim = points.shape
    if dim == 4:
        f.write("""# .PCD v.7 - Point Cloud Data file format\n""")
        f.write("VERSION .7\nFIELDS x y z intensity")
    elif dim == 5:
        f.write("""# .PCD v.7 - Point Cloud Data file format\n""")
        f.write("VERSION .7\nFIELDS x y z intensity class\n")

    f.write("SIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1")
    f.write("WIDTH %d\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS %d\nDATA ascii\n" % (N, N))

    for i in range(N):
        if dim == 4:
            f.write("%.3f %.3f %.3f %d\n" % (points[i, 0], points[i, 1], points[i, 2], points[i, 3]))
        elif dim == 5:
            f.write("%.3f %.3f %.3f %.3f %d\n" % (points[i, 0], points[i, 1], points[i, 2], points[i, 3], points[i, 4]))
    f.close()


def GetTestDataList(folder, lidar_id_list):
    data_num = len(glob.glob(os.path.join(folder, "lidar", lidar_id_list[0], "*.pcd")))
    lidar_name_set_list = []
    img_name_set_list = []
    for j in range(data_num):
        lidar_name_set_list.append(dict())
        img_name_set_list.append(dict())
    
    para_name_list = {}

    for lidar_id in lidar_id_list:
        lidar_list_temp = glob.glob(os.path.join(folder, "lidar", lidar_id, "*.pcd"))
        lidar_list_temp.sort()
        img_list_temp = glob.glob(os.path.join(folder, "image", lidar_id, "*.jpg"))
        img_list_temp.sort()

        para_name_list[lidar_id] = os.path.join(folder, "parameter", lidar_id, "config.txt")

        for j in range(data_num):
            lidar_name_set_list[j][lidar_id] = lidar_list_temp[j]
            if j < len(img_list_temp):
                img_name_set_list[j][lidar_id] = img_list_temp[j]


    return lidar_name_set_list, img_name_set_list, para_name_list



def GetNumFromOneLine(one_line):
    one_line = one_line.strip()
    nums = []
    flag = 1
    num_str = [""]
    for i in range(len(one_line)):
        if flag == 1:
            if one_line[i] != " " and one_line[i] != "\t":
                num_str[-1] = num_str[-1] + one_line[i]
            else:
                flag = 0
        else:
            if one_line[i] != " " and one_line[i] != "\t":
                num_str.append("")
                num_str[-1] = num_str[-1] + one_line[i]
                flag = 1
    for item in num_str:
        if item != "":
            nums.append(float(item))

    return nums

def ReadSelectedPara(para_name_list):
    K_default = {}
    P_default = {}
    P_world_default = {}
    K_default["1"] = np.array([[957.994,   0,        790.335,  0],
                               [0,         955.3280, 250.6631, 0],
                               [0,         0,        1,        0]]).astype("float32")
    K_default["2"] = np.array([[949.038,   0,        774.967,  0],
                               [0,         946.466,  283.137,  0],
                               [0,         0,        1,        0]]).astype("float32")
    K_default["3"] = np.array([[927.697,   0,        749.535,  0],
                               [0,         919.007 , 312.66 ,  0],
                               [0,         0,        1,        0]]).astype("float32")
    K_default["4"] = np.array([[918.696 ,  0,        789.368,  0],
                               [0,         925.347,  310.116 , 0],
                               [0,         0,        1,        0]]).astype("float32")
    K_default["5"] = np.array([[929.028,   0,        812.443,  0],
                               [0,         925.386 , 295.572,  0],
                               [0,         0,        1,        0]]).astype("float32")
    K_default["6"] = np.array([[4728.8 ,   0,        876.11,  0],
                               [0,         4717,     411.5,   0],
                               [0,         0,        1,       0]]).astype("float32")
    P_default["1"] = np.array([[0.00554604,      -0.999971,       -0.00523653,     0.0316362],
                       [-0.000379382,    0.00523451,      -0.999986,       0.0380934],
                       [0.999985,        0.00554795,      -0.000350341,    0.409066],
                       [0,               0,                0,              1]]).astype("float32")
    P_default["2"] = np.array([[-0.00494621,     -0.999944,       -0.00931772,     0.160904],
                       [-0.0124084,      0.00937849,      -0.999879,       0.126189],
                       [0.999911,        -0.00483,        -0.0124541,      0.356469],
                       [0,               0,               0,               1]]).astype("float32")
    P_default["3"] = np.array([[0.01302,         -0.999907,       -0.00412561,     -0.0141273],
                       [-0.0110091,      0.00398236,      -0.999931,       0.0552548],
                       [0.999855,        0.0130645,       -0.0109562,      -0.0442258],
                       [0,               0,               0,               1]]).astype("float32")
    P_default["4"] = np.array([[-0.00981236,     -0.999946 ,      -0.00344587,     0.0113077],
                       [-0.0189915,      0.00363177,      -0.999813,       0.0466016],
                       [0.999771,        -0.00974508,     -0.0190261,      -0.0685527],
                       [0 ,              0,               0,               1]]).astype("float32")
    P_default["5"] = np.array([[-0.00667688,     -0.999945,       -0.00814291,     -0.0127165],
                        [-0.0047921,      0.00817499,      -0.999955,       0.0400943],
                        [0.999966,        -0.00663756,     -0.00484642,     -0.0503617],
                        [0,               0,               0,               1]]).astype("float32")
    P_default["6"] = np.array([[-0.000170169,    -0.999914,       0.0130839,       0.104728],
                        [0.029933,        -0.0130831,      -0.999466,       0.165281],
                        [0.999552,        0.000221562,     0.0299326,       7.34219],
                        [0,               0,               0,               1]]).astype("float32")
    P_world_default["1"] = np.array([[   0.999972,           0,  0.00750485,           0],
                                    [           0,           1,           0,           0],
                                    [ -0.00750485,           0,    0.999972,       1.909],
                                    [           0,           0,           0,           1]]).astype("float32")
    P_world_default["2"] = np.array([[   0.338535,   -0.939803,   0.0465286,      -0.001],
                                     [   0.929107,    0.341685,    0.141463,       0.338],
                                     [  -0.148845, -0.00465988,     0.98885,       1.909],
                                     [          0,           0,           0,           1]]).astype("float32")
    P_world_default["3"] = np.array([[  -0.742701,   -0.656677,   -0.131041,     -1.109],
                                     [    0.64927,   -0.754084,   0.0990242,       0.15],
                                     [  -0.163842,  -0.0115354,    0.986419,      1.909],
                                     [          0,           0,           0,          1]]).astype("float32")
    P_world_default["4"] = np.array([[  -0.770801,     0.62401,   -0.128361,       -1.2],
                                     [  -0.617308,   -0.781369,  -0.0916196,     -0.149],
                                     [  -0.157469,  0.00861766,    0.987486,      1.909],
                                     [          0,           0,           0,          1]]).astype("float32")
    P_world_default["5"] = np.array([[    0.33825,    0.939574,   0.0528062,      -0.009],
                                     [  -0.929838,    0.342329,   -0.134952,      -0.301],
                                     [  -0.144874, -0.00345383,    0.989444,       1.909],
                                     [          0,           0,           0,           1]]).astype("float32")
    P_world_default["6"] = np.array([[   0.999931, -0.000861352,    0.0117086,       -0.037],
                                     [ 0.00104713,     0.999874,   -0.0158696,        0.019],
                                     [ -0.0116934,    0.0158807,     0.999806,        1.909],
                                     [          0,            0,            0,            1]]).astype("float32")

    lidar_id_list = para_name_list.keys()
    para = {}

    for lidar_id in lidar_id_list:
        para_file = para_name_list[lidar_id]
        para[lidar_id] = {}
        if not os.path.exists(para_file):
            para[lidar_id]["K"] = K_default[lidar_id]
            para[lidar_id]["P"] = P_default[lidar_id]
            para[lidar_id]["P_world"] = P_world_default[lidar_id]
        else:
            with open(para_file) as f:
                info = f.readlines()

            nums_line1 = GetNumFromOneLine(info[1]) + [0.0]
            nums_line2 = GetNumFromOneLine(info[2]) + [0.0]
            nums_line3 = GetNumFromOneLine(info[3]) + [0.0]
            para[lidar_id]["K"] = np.array([nums_line1, nums_line2, nums_line3]).astype("float32")

            nums_line9 = GetNumFromOneLine(info[9])
            nums_line10 = GetNumFromOneLine(info[10])
            nums_line11 = GetNumFromOneLine(info[11])
            nums_line12 = GetNumFromOneLine(info[12])
            para[lidar_id]["P"] = np.array([nums_line9, nums_line10, nums_line11, nums_line12]).astype("float32")

            nums_line15 = GetNumFromOneLine(info[15])
            nums_line16 = GetNumFromOneLine(info[16])
            nums_line17 = GetNumFromOneLine(info[17])
            nums_line18 = GetNumFromOneLine(info[18])
            para[lidar_id]["P_world"] = np.array([nums_line15, nums_line16, nums_line17, nums_line18]).astype("float32")

    return para

def ReadSelectedPoints(file_name_set):
    lidar_id_list = file_name_set.keys()
    points_result = {}

    for lidar_id in lidar_id_list:
        pc_file_name = file_name_set[lidar_id]
        points = ReadPcdFile(pc_file_name)
        points_result[lidar_id] = points
    return points_result

def OutputPointsWithClass(points_set, points_class_set, output_path, pc_name_set, para_name_set):
    lidar_id_list = pc_name_set.keys()
    output_path_lidar = os.path.join(output_path, "lidar")
    output_path_para = os.path.join(output_path, "parameter")
    if not os.path.exists(output_path_lidar):
        os.mkdir(output_path_lidar)
    if not os.path.exists(output_path_para):
        os.mkdir(output_path_para)

    for lidar_id in lidar_id_list:
        output_folder = os.path.join(output_path_lidar, lidar_id)
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        pc_name0 = pc_name_set[lidar_id]
        pure_name = pc_name0.split("/")[-1]
        output_name = os.path.join(output_folder, pure_name)

        points_with_class = np.concatenate([points_set[lidar_id], points_class_set[lidar_id]], axis=1)
        WritePointsToFile(points_with_class, output_name)

    for lidar_id in lidar_id_list:
        para_file = para_name_set[lidar_id]
        if not os.path.exists(para_file):
            continue
        pure_name = para_file.split("/")[-1]
        output_folder = os.path.join(output_path_para, lidar_id)
        output_name = os.path.join(output_folder, pure_name)

        if os.path.exists(output_name):
            continue

        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        shutil.copyfile(para_file, output_name)


def SaveVisImg(vis_img, pc_name_set, output_path):
    one_pc_name = pc_name_set[list(pc_name_set)[0]]

    pure_name = one_pc_name.split("/")[-1][:-4]
    output_name = os.path.join(output_path, pure_name+".jpg")

    cv2.imwrite(output_name, vis_img)