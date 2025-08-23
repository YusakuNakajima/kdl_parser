#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import os
from ament_index_python.packages import get_package_share_directory
# pyKDL is installed in system Python, so we need to add the path
import sys
sys.path.append('/usr/lib/python3/dist-packages')
import PyKDL
# Use kdl_parser_py from same package
from kdl_parser_py.urdf import treeFromFile, treeFromString
# 補助関数 (変更なし)
def frame_to_list(frame):
    pos = frame.p
    rot = frame.M.GetQuaternion()
    return np.array([pos[0], pos[1], pos[2], rot[0], rot[1], rot[2], rot[3]])

def quaternion_matrix(quaternion):
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < np.finfo(float).eps * 4.0:
        return np.identity(4)
    q *= np.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3], 0.0],
        [    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], 0.0],
        [    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], 0.0],
        [                0.0,                 0.0,                 0.0, 1.0]])


class KDLHelper:
    """
    URDFからロードされた任意のロボットの運動学をPyKDLで計算するための汎用クラス。
    URDFファイルパスまたは文字列から初期化する。
    """
    def __init__(self, logger, urdf_path=None, urdf_string=None, base_link="base_link", ee_link="tool0"):
        self.logger = logger
        
        # URDFファイルパスが指定されている場合はtreeFromFileを使用
        if urdf_path is not None:
            self.logger.info(f"Loading URDF from file: {urdf_path}")
            ok, self._kdl_tree = treeFromFile(urdf_path)
        elif urdf_string is not None:
            self.logger.info("Loading URDF from string")
            ok, self._kdl_tree = treeFromString(urdf_string)
        else:
            self.logger.error("Either urdf_path or urdf_string must be provided.")
            raise ValueError("Either urdf_path or urdf_string must be provided.")
            
        if not ok:
            self.logger.error("Failed to parse URDF to KDL tree.")
            raise ValueError("Failed to parse URDF.")

        self._base_link = base_link
        self._ee_link = ee_link

        # base_linkからee_linkまでの運動学チェーンを生成
        self._arm_chain = self._kdl_tree.getChain(self._base_link, self._ee_link)

        # チェーンから可動関節の名前と数を自動的に抽出
        self.joint_names = []
        for i in range(self._arm_chain.getNrOfSegments()):
            joint = self._arm_chain.getSegment(i).getJoint()
            if joint.getType() != PyKDL.Joint.Fixed:  # Fixed joints are not movable
                self.joint_names.append(joint.getName())
        
        self._num_jnts = len(self.joint_names)
        if self._num_jnts == 0:
            raise ValueError(f"No non-fixed joints found in chain from '{base_link}' to '{ee_link}'.")

        # KDLソルバーの初期化
        self._fk_p_kdl = PyKDL.ChainFkSolverPos_recursive(self._arm_chain)
        self._ik_v_kdl = PyKDL.ChainIkSolverVel_pinv(self._arm_chain)
        self._ik_p_kdl = PyKDL.ChainIkSolverPos_NR(self._arm_chain, self._fk_p_kdl, self._ik_v_kdl)
        self._jac_kdl = PyKDL.ChainJntToJacSolver(self._arm_chain)
        self._dyn_kdl = PyKDL.ChainDynParam(self._arm_chain, PyKDL.Vector.Zero())
        
        self.print_robot_description()

    def print_robot_description(self):
        """ロボットの情報をROS2ロガーで表示します。"""
        self.logger.info("-" * 30)
        self.logger.info("KDL Chain Details:")
        self.logger.info(f"  Base Link: {self._base_link}")
        self.logger.info(f"  EE Link:   {self._ee_link}")
        self.logger.info(f"  Joints:    {self._num_jnts} -> {self.joint_names}")
        self.logger.info("-" * 30)

    # _joints_to_kdl, _kdl_to_mat, forward_kinematics, inverse_kinematics, 
    # jacobian, jacobian_pseudo_inverse, inertia_matrix の各メソッドは
    # 変更がないため、元のコードをそのまま使用します。
    # (ここでは簡潔さのため省略しますが、実際にはクラス内に含めてください)
    def _joints_to_kdl(self, joint_values):
        if len(joint_values) != self._num_jnts:
            raise ValueError(f"Invalid number of joint values. Expected {self._num_jnts}, but got {len(joint_values)}")
        kdl_array = PyKDL.JntArray(self._num_jnts)
        for idx, val in enumerate(joint_values):
            kdl_array[idx] = val
        return kdl_array
    
    def _kdl_to_mat(self, kdl_data):
        mat = np.zeros((kdl_data.rows(), kdl_data.columns()))
        for i in range(kdl_data.rows()):
            for j in range(kdl_data.columns()):
                mat[i, j] = kdl_data[i, j]
        return mat

    def forward_kinematics(self, joint_values, get_transform=False):
        kdl_joints = self._joints_to_kdl(joint_values)
        end_frame = PyKDL.Frame()
        self._fk_p_kdl.JntToCart(kdl_joints, end_frame)
        pose_list = frame_to_list(end_frame)
        if get_transform:
            transform = quaternion_matrix(pose_list[3:])
            transform[:3, 3] = pose_list[:3]
            return transform
        else:
            return pose_list

    def inverse_kinematics(self, position, orientation=None, seed=None):
        pos = PyKDL.Vector(position[0], position[1], position[2])
        rot = PyKDL.Rotation()
        if orientation is not None:
            rot = PyKDL.Rotation.Quaternion(orientation[0], orientation[1], orientation[2], orientation[3])
        goal_pose = PyKDL.Frame(rot, pos)
        seed_array = PyKDL.JntArray(self._num_jnts)
        if seed is not None:
            if len(seed) != self._num_jnts:
                 raise ValueError(f"Invalid seed length. Expected {self._num_jnts}, but got {len(seed)}")
            for i in range(self._num_jnts):
                seed_array[i] = seed[i]
        result_angles = PyKDL.JntArray(self._num_jnts)
        if self._ik_p_kdl.CartToJnt(seed_array, goal_pose, result_angles) >= 0:
            return np.array(list(result_angles))
        else:
            return None

    def jacobian(self, joint_values):
        kdl_joints = self._joints_to_kdl(joint_values)
        jacobian = PyKDL.Jacobian(self._num_jnts)
        self._jac_kdl.JntToJac(kdl_joints, jacobian)
        return self._kdl_to_mat(jacobian)

    def jacobian_pseudo_inverse(self, joint_values):
        return np.linalg.pinv(self.jacobian(joint_values))

    def inertia_matrix(self, joint_values):
        kdl_joints = self._joints_to_kdl(joint_values)
        inertia = PyKDL.JntSpaceInertiaMatrix(self._num_jnts)
        self._dyn_kdl.JntToMass(kdl_joints, inertia)
        return self._kdl_to_mat(inertia)


class KdlExampleNode(Node):
    def __init__(self):
        super().__init__('kdl_helper_example_node')

        
        # パラメータを取得
        urdf_path = get_package_share_directory('grinding_robot_control') + '/config/ur5e.urdf'
        base_link = "base_link"
        ee_link = "tool0"
        
        if not urdf_path or not os.path.exists(urdf_path):
            self.get_logger().fatal(f"URDF file not found at path: {urdf_path}")
            return
            
        self.get_logger().info(f"Loading URDF from: {urdf_path}")
        
        try:
            self.kin = KDLHelper(self.get_logger(), urdf_path=urdf_path, base_link=base_link, ee_link=ee_link)
            self.run_tests()
        except Exception as e:
            self.get_logger().error(f"Failed to initialize KDLHelper or run tests: {e}")

    def run_tests(self):
        self.get_logger().info("--- Running Kinematics Tests ---")
        
        # UR5eの関節角度 (ラジアン) - 6軸
        joint_angles = [0.0, -np.pi/2, np.pi/2, 0.0, np.pi/2, 0.0]
        self.get_logger().info(f"Calculating FK for joint angles: {joint_angles}")
        
        # 順運動学 (FK)
        ee_pose = self.kin.forward_kinematics(joint_angles)
        self.get_logger().info(f"Forward Kinematics (Pose): {np.round(ee_pose, 3)}")

        # 逆運動学 (IK)
        target_pos = [ee_pose[0], ee_pose[1], ee_pose[2]]
        target_ori = [ee_pose[3], ee_pose[4], ee_pose[5], ee_pose[6]]
        initial_guess = [0.1, -np.pi/2, np.pi/2, 0.1, np.pi/2, 0.1]
        
        ik_solution = self.kin.inverse_kinematics(target_pos, target_ori, seed=initial_guess)
        if ik_solution is not None:
            self.get_logger().info(f"Inverse Kinematics (Solution): {np.round(ik_solution, 3)}")
        else:
            self.get_logger().warn("IK solution not found.")

        # ヤコビアン
        jacobian_matrix = self.kin.jacobian(joint_angles)
        self.get_logger().info(f"Jacobian Matrix:\n{np.round(jacobian_matrix, 3)}")
        

def main(args=None):
    rclpy.init(args=args)
    node = KdlExampleNode()
    # この例ではテスト実行後にシャットダウン
    rclpy.shutdown()

if __name__ == '__main__':
    main()