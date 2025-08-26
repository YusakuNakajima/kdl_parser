#!/usr/bin/env python3
"""
KDL Helper - PyKDLを使用した汎用運動学計算ライブラリ

このモジュールは、任意のURDFロボットモデルに対して運動学計算を行うための
包括的なソリューションを提供します。

主な機能:
- URDFファイルまたは文字列からのロボットモデル読み込み
- 順運動学 (Forward Kinematics)
- 逆運動学 (Inverse Kinematics) 
- ヤコビアン行列計算
- ヤコビアン疑似逆行列計算
- 慣性行列計算
- 詳細なデバッグ情報出力
- エラーハンドリング

使用例:
    # URDFファイルから初期化
    helper = KDLHelper(logger, urdf_path="/path/to/robot.urdf", 
                      base_link="base_link", ee_link="end_effector")
    
    # 順運動学
    joint_angles = [0.1, 0.2, 0.3]
    pose = helper.forward_kinematics(joint_angles, debug=True)
    
    # 逆運動学
    target_pos = [0.5, 0.0, 0.3]
    target_ori = [0, 0, 0, 1]  # クォータニオン
    solution = helper.inverse_kinematics(target_pos, target_ori, debug=True)

著者: OnoLab
バージョン: 1.1
最終更新: 2025-01-26
"""

import rclpy
from rclpy.node import Node
import numpy as np
import os
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
        self.logger.info("-" * 50)
        self.logger.info("KDL Chain Details:")
        self.logger.info(f"  Base Link: {self._base_link}")
        self.logger.info(f"  EE Link:   {self._ee_link}")
        self.logger.info(f"  Chain Segments: {self._arm_chain.getNrOfSegments()}")
        self.logger.info(f"  DOF (関節数): {self._num_jnts}")
        self.logger.info(f"  Joint Names: {self.joint_names}")
        
        # 各セグメントの詳細情報を表示
        self.logger.info("  Segment Details:")
        for i in range(self._arm_chain.getNrOfSegments()):
            segment = self._arm_chain.getSegment(i)
            joint = segment.getJoint()
            joint_name = joint.getName()
            joint_type = self._get_joint_type_name(joint.getType())
            self.logger.info(f"    [{i}] '{segment.getName()}' -> Joint: '{joint_name}' ({joint_type})")
        
        # KDLソルバーの初期化状態
        self.logger.info("  KDL Solvers initialized:")
        self.logger.info(f"    - Forward Kinematics (FK): {'✓' if self._fk_p_kdl else '✗'}")
        self.logger.info(f"    - Inverse Kinematics (IK): {'✓' if self._ik_p_kdl else '✗'}")
        self.logger.info(f"    - Jacobian Solver: {'✓' if self._jac_kdl else '✗'}")
        self.logger.info(f"    - Dynamics (Inertia): {'✓' if self._dyn_kdl else '✗'}")
        self.logger.info("-" * 50)
    
    def _get_joint_type_name(self, joint_type):
        """関節タイプの名前を取得"""
        type_names = {
            PyKDL.Joint.Fixed: "Fixed",
            PyKDL.Joint.RotAxis: "Revolute",
            PyKDL.Joint.RotX: "RotX", 
            PyKDL.Joint.RotY: "RotY",
            PyKDL.Joint.RotZ: "RotZ",
            PyKDL.Joint.TransAxis: "Prismatic",
            PyKDL.Joint.TransX: "TransX",
            PyKDL.Joint.TransY: "TransY", 
            PyKDL.Joint.TransZ: "TransZ"
        }
        return type_names.get(joint_type, f"Unknown({joint_type})")

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

    def forward_kinematics(self, joint_values, get_transform=False, debug=False):
        """
        順運動学を計算します。
        
        Args:
            joint_values: 関節角度のリスト/配列
            get_transform: Trueの場合4x4変換行列を返す、Falseの場合7要素ポーズを返す
            debug: デバッグ情報を出力するかどうか
        
        Returns:
            get_transform=Trueの場合: 4x4変換行列
            get_transform=Falseの場合: [x,y,z,qx,qy,qz,qw]のポーズ
        """
        if debug:
            self.logger.info(f"FK計算開始: 関節角度={joint_values}")
            
        kdl_joints = self._joints_to_kdl(joint_values)
        end_frame = PyKDL.Frame()
        result = self._fk_p_kdl.JntToCart(kdl_joints, end_frame)
        
        if result < 0:
            self.logger.error(f"FK計算エラー: KDLソルバーがエラーコード {result} を返しました")
            return None
            
        pose_list = frame_to_list(end_frame)
        
        if debug:
            self.logger.info(f"FK結果: 位置=[{pose_list[0]:.4f}, {pose_list[1]:.4f}, {pose_list[2]:.4f}]")
            self.logger.info(f"FK結果: 姿勢=[{pose_list[3]:.4f}, {pose_list[4]:.4f}, {pose_list[5]:.4f}, {pose_list[6]:.4f}]")
        
        if get_transform:
            transform = quaternion_matrix(pose_list[3:])
            transform[:3, 3] = pose_list[:3]
            if debug:
                self.logger.info(f"FK結果: 変換行列=\n{np.round(transform, 4)}")
            return transform
        else:
            return pose_list

    def inverse_kinematics(self, position, orientation=None, seed=None, debug=False):
        """
        逆運動学を計算します。
        
        Args:
            position: ターゲット位置 [x, y, z]
            orientation: ターゲット姿勢のクォータニオン [x, y, z, w] (オプション)
            seed: 初期推定値 (オプション)
            debug: デバッグ情報を出力するかどうか
        
        Returns:
            解が見つかった場合: 関節角度の配列
            解が見つからない場合: None
        """
        if debug:
            self.logger.info(f"IK計算開始:")
            self.logger.info(f"  ターゲット位置: [{position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}]")
            if orientation is not None:
                self.logger.info(f"  ターゲット姿勢: [{orientation[0]:.4f}, {orientation[1]:.4f}, {orientation[2]:.4f}, {orientation[3]:.4f}]")
            if seed is not None:
                self.logger.info(f"  初期推定値: {seed}")
        
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
        ik_result = self._ik_p_kdl.CartToJnt(seed_array, goal_pose, result_angles)
        
        if ik_result >= 0:
            solution = np.array(list(result_angles))
            if debug:
                self.logger.info(f"IK成功: 解={solution}")
                # 検証のためにFKを実行
                verify_pose = self.forward_kinematics(solution)
                pos_error = np.linalg.norm(verify_pose[:3] - position)
                self.logger.info(f"IK検証: 位置誤差={pos_error:.6f}")
            return solution
        else:
            if debug:
                self.logger.warn(f"IK失敗: KDLソルバーがエラーコード {ik_result} を返しました")
            return None

    def jacobian(self, joint_values, debug=False):
        """
        ヤコビアン行列を計算します。
        
        Args:
            joint_values: 関節角度のリスト/配列
            debug: デバッグ情報を出力するかどうか
        
        Returns:
            6xN のヤコビアン行列 (N=関節数)
        """
        if debug:
            self.logger.info(f"ヤコビアン計算開始: 関節角度={joint_values}")
            
        kdl_joints = self._joints_to_kdl(joint_values)
        jacobian = PyKDL.Jacobian(self._num_jnts)
        result = self._jac_kdl.JntToJac(kdl_joints, jacobian)
        
        if result < 0:
            self.logger.error(f"ヤコビアン計算エラー: KDLソルバーがエラーコード {result} を返しました")
            return None
            
        jac_matrix = self._kdl_to_mat(jacobian)
        
        if debug:
            self.logger.info(f"ヤコビアン結果: 形状={jac_matrix.shape}")
            self.logger.info(f"ヤコビアン行列:\n{np.round(jac_matrix, 4)}")
            
        return jac_matrix

    def jacobian_pseudo_inverse(self, joint_values, debug=False):
        """
        ヤコビアン行列の疑似逆行列を計算します。
        
        Args:
            joint_values: 関節角度のリスト/配列
            debug: デバッグ情報を出力するかどうか
        
        Returns:
            Nx6 のヤコビアン疑似逆行列 (N=関節数)
        """
        jac = self.jacobian(joint_values, debug=debug)
        if jac is None:
            return None
            
        pinv_jac = np.linalg.pinv(jac)
        
        if debug:
            self.logger.info(f"ヤコビアン疑似逆行列: 形状={pinv_jac.shape}")
            # 特異値分解の情報も表示
            u, s, vh = np.linalg.svd(jac)
            self.logger.info(f"ヤコビアン特異値: {np.round(s, 6)}")
            
        return pinv_jac

    def inertia_matrix(self, joint_values, debug=False):
        """
        慣性行列を計算します。
        
        Args:
            joint_values: 関節角度のリスト/配列
            debug: デバッグ情報を出力するかどうか
        
        Returns:
            NxN の慣性行列 (N=関節数)
        """
        if debug:
            self.logger.info(f"慣性行列計算開始: 関節角度={joint_values}")
            
        kdl_joints = self._joints_to_kdl(joint_values)
        inertia = PyKDL.JntSpaceInertiaMatrix(self._num_jnts)
        result = self._dyn_kdl.JntToMass(kdl_joints, inertia)
        
        if result < 0:
            self.logger.error(f"慣性行列計算エラー: KDLソルバーがエラーコード {result} を返しました")
            return None
            
        inertia_matrix = self._kdl_to_mat(inertia)
        
        if debug:
            self.logger.info(f"慣性行列結果: 形状={inertia_matrix.shape}")
            self.logger.info(f"慣性行列:\n{np.round(inertia_matrix, 4)}")
            
            # 行列の特性をチェック
            eigenvals = np.linalg.eigvals(inertia_matrix)
            is_positive_definite = np.all(eigenvals > 0)
            is_symmetric = np.allclose(inertia_matrix, inertia_matrix.T, atol=1e-6)
            
            self.logger.info(f"慣性行列特性: 対称={is_symmetric}, 正定値={is_positive_definite}")
            self.logger.info(f"固有値: {np.round(eigenvals, 6)}")
            
        return inertia_matrix


class KdlExampleNode(Node):
    def __init__(self):
        super().__init__('kdl_helper_example_node')
        
        # テスト用URDFファイルを使用 (kdl_parser_pyパッケージ内のもの)
        current_package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        urdf_path = os.path.join(current_package_dir, 'test', 'test.urdf')
        base_link = "base_link"
        ee_link = "right_gripper"
        
        if not os.path.exists(urdf_path):
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
        
        # テスト用関節角度 (ラジアン) - test.urdfの場合は2軸
        joint_angles = [0.1, 0.2]  # gripper_extension, right_gripper_joint
        self.get_logger().info(f"Calculating FK for joint angles: {joint_angles}")
        
        # 順運動学 (FK)
        ee_pose = self.kin.forward_kinematics(joint_angles)
        self.get_logger().info(f"Forward Kinematics (Pose): {np.round(ee_pose, 3)}")

        # 逆運動学 (IK)
        target_pos = [ee_pose[0], ee_pose[1], ee_pose[2]]
        target_ori = [ee_pose[3], ee_pose[4], ee_pose[5], ee_pose[6]]
        initial_guess = [0.0, 0.0]  # 2軸用の初期推定値
        
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