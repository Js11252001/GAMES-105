import numpy as np
from scipy.spatial.transform import Rotation as R

def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i+1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1,-1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data



def part1_calculate_T_pose(bvh_file_path):
    """请填写以下内容
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量

    Tips:
        joint_name顺序应该和bvh一致
    """
    # Open BVH file
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()

    joint_name = []
    joint_parent = []
    joint_offset = []
    joint_stack = []

    for i in range(len(lines)):
        if lines[i].startswith('MOTION'):
            break
        if lines[i].startswith('ROOT'):
            joint_name.append(lines[i].split()[1])
            joint_parent.append(-1)
        elif lines[i].strip().startswith('JOINT'):
            joint_name.append(lines[i].split()[1])
            joint_parent.append(joint_stack[-1])
        elif lines[i].strip().startswith('End'):
            # name with parent name + '_end'
            joint_name.append(joint_name[joint_stack[-1]] + '_end')
            joint_parent.append(joint_stack[-1])
        elif lines[i].strip().startswith('OFFSET'):
            # Joint offset
            tmp_line = lines[i].split()
            joint_offset.append(np.array([float(x) for x in tmp_line[1:4]]).reshape(1, -1))
        elif lines[i].strip().startswith('{'):
            joint_stack.append(len(joint_name) - 1)
        elif lines[i].strip().startswith('}'):
            # End of joint
            joint_stack.pop()
    joint_offset = np.concatenate(joint_offset, axis=0)
    return joint_name, joint_parent, joint_offset


def part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
        2. from_euler时注意使用大写的XYZ
    """
    joint_positions = np.zeros((len(joint_name), 3))
    joint_orientations = np.zeros((len(joint_name), 4))
    root_pos = motion_data[frame_id, :3]
    rot_data = motion_data[frame_id, 3:]
    channel_offset = 1
    for i in range(len(joint_name)):
        if i > 0:
            # 计算全局旋转 注意，end没有旋转
            # 从运动数据中获取关节的旋转角度
            if 'end' in joint_name[i]:
                angles = [0.,0.,0.]
            else:
                angles = rot_data[channel_offset * 3:channel_offset * 3 + 3]
                channel_offset += 1

            # 将欧拉角转换为四元数
            rotation = R.from_euler('XYZ', angles, degrees=True)
            quat = rotation.as_quat()

            # 将父节点的全局旋转与当前关节的旋转相乘，以获得当前关节的全局旋转
            parent_orient = joint_orientations[joint_parent[i]]
            joint_orientations[i] = R.as_quat(R.from_quat(parent_orient) * R.from_quat(quat))
            # 非根节点的全局位置
            parent_pos = joint_positions[joint_parent[i]]
            joint_positions[i] = parent_pos + R.from_quat(parent_orient).apply(joint_offset[i])
        else:
            # 根关节的旋转为单位四元数
            angles = rot_data[:3]
            # 将欧拉角转换为四元数
            rotation = R.from_euler('XYZ', angles, degrees=True)
            quat = rotation.as_quat()
            joint_orientations[i] = quat
            # 根节点的全局位置
            joint_positions[i] = np.array(root_pos) + joint_offset[i]

    return joint_positions, joint_orientations


def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
        as_euler时也需要大写的XYZ
    """
    # 获取T、A pose数据
    T_pose_name,_,_ = part1_calculate_T_pose(T_pose_bvh_path)
    A_pose_name,_,_ = part1_calculate_T_pose(A_pose_bvh_path)
    A_pose_motion_data = load_motion_data(A_pose_bvh_path)
    motion_data = np.zeros_like(A_pose_motion_data)
    A_joint_map = {}
    count = 0
    for i in range(len(A_pose_name)):
        if '_end' in A_pose_name[i]:
            count += 1
        A_joint_map[A_pose_name[i]] = i - count

    # 更新将A-pose的动画转为T-pose动画
    for i in range(A_pose_motion_data.shape[0]):
        data = []
        for joint in T_pose_name:
            index = A_joint_map[joint]
            if joint == 'lShoulder':
                Rot = (R.from_euler('XYZ', list(A_pose_motion_data[i][index * 3 + 3: index*3 + 6]), degrees=True) * R.from_euler('XYZ', [0., 0., -45.], degrees=True)).as_euler('XYZ',True)
                data += (list(Rot))
            elif joint == 'rShoulder':
                Rot = (R.from_euler('XYZ', list(A_pose_motion_data[i][index * 3 + 3: index*3 + 6]), degrees=True) * R.from_euler('XYZ', [0., 0.,  45.], degrees=True)).as_euler('XYZ',True)
                data += (list(Rot))
            elif '_end' in joint:
                continue
            elif joint == 'RootJoint':
                data += (list(A_pose_motion_data[i][0:6]))
            else:
                data += (list(A_pose_motion_data[i][index * 3 + 3: index * 3 + 6]))
        motion_data[i] = data
    return motion_data
