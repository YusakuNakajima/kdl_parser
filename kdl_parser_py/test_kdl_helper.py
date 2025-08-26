#!/usr/bin/env python3
"""
KDLHelperクラスの包括的なテストスクリプト
- 全ての機能が正常に動作することを確認
- デバッグ情報の表示
- エラーハンドリングのテスト
"""

import rclpy
from rclpy.node import Node
import numpy as np
import os
import sys
import traceback
from kdl_parser_py.kdl_helper import KDLHelper, frame_to_list, quaternion_matrix

class TestKDLHelper:
    """KDLHelperの包括的テストクラス"""
    
    def __init__(self, logger):
        self.logger = logger
        self.test_results = []
        
    def run_all_tests(self):
        """全てのテストを実行"""
        self.logger.info("=" * 50)
        self.logger.info("KDLHelper 包括的テスト開始")
        self.logger.info("=" * 50)
        
        # テスト用URDFファイルのパス
        test_urdf_path = os.path.join(os.path.dirname(__file__), "test", "test.urdf")
        if not os.path.exists(test_urdf_path):
            self.logger.error(f"テスト用URDFファイルが見つかりません: {test_urdf_path}")
            return False
        
        try:
            # 基本的な初期化テスト
            self.test_initialization(test_urdf_path)
            
            # 補助関数のテスト
            self.test_helper_functions()
            
            # 各機能のテスト
            self.test_forward_kinematics()
            self.test_inverse_kinematics() 
            self.test_jacobian()
            self.test_inertia_matrix()
            self.test_error_handling()
            
            # デバッグ機能のテスト
            self.test_debug_functionality()
            
        except Exception as e:
            self.logger.error(f"テスト実行中にエラー: {e}")
            self.logger.error(traceback.format_exc())
            return False
            
        # 結果のサマリー
        self.print_test_summary()
        return all(result[1] for result in self.test_results)
    
    def test_initialization(self, urdf_path):
        """初期化テスト"""
        self.logger.info("テスト1: 初期化テスト")
        
        try:
            # ファイルから初期化
            self.kdl_helper = KDLHelper(
                self.logger,
                urdf_path=urdf_path,
                base_link="base_link", 
                ee_link="right_gripper"
            )
            
            # 基本情報の確認
            self.logger.info(f"関節数: {self.kdl_helper._num_jnts}")
            self.logger.info(f"関節名: {self.kdl_helper.joint_names}")
            
            # URDFから期待される関節数を確認 (test.urdfから)
            expected_joints = ["gripper_extension", "right_gripper_joint"]
            
            if self.kdl_helper._num_jnts == len(expected_joints):
                self.add_test_result("初期化", True, "成功")
            else:
                self.add_test_result("初期化", False, f"関節数不一致: 期待値{len(expected_joints)}, 実際{self.kdl_helper._num_jnts}")
                
        except Exception as e:
            self.add_test_result("初期化", False, f"例外発生: {e}")
    
    def test_forward_kinematics(self):
        """順運動学テスト"""
        self.logger.info("テスト2: 順運動学 (FK) テスト")
        
        try:
            # テスト用関節角度 (関節数分)
            joint_angles = [0.0, 0.0]  # gripper_extension, right_gripper_joint
            
            # 通常のFK
            pose = self.kdl_helper.forward_kinematics(joint_angles)
            self.logger.info(f"FK結果 (pose): {np.round(pose, 4)}")
            
            if len(pose) == 7:  # [x,y,z,qx,qy,qz,qw]
                self.add_test_result("FK (pose)", True, "成功")
            else:
                self.add_test_result("FK (pose)", False, f"ポーズ長さ異常: {len(pose)}")
            
            # 変換行列版のFK
            transform = self.kdl_helper.forward_kinematics(joint_angles, get_transform=True)
            self.logger.info(f"FK結果 (transform): \n{np.round(transform, 4)}")
            
            if transform.shape == (4, 4):
                self.add_test_result("FK (transform)", True, "成功")
            else:
                self.add_test_result("FK (transform)", False, f"変換行列サイズ異常: {transform.shape}")
                
        except Exception as e:
            self.add_test_result("FK", False, f"例外発生: {e}")
    
    def test_inverse_kinematics(self):
        """逆運動学テスト"""
        self.logger.info("テスト3: 逆運動学 (IK) テスト")
        
        try:
            # まずFKで目標ポーズを取得
            joint_angles = [0.1, 0.2]
            target_pose = self.kdl_helper.forward_kinematics(joint_angles)
            target_pos = target_pose[:3]
            target_ori = target_pose[3:]
            
            self.logger.info(f"目標位置: {np.round(target_pos, 4)}")
            self.logger.info(f"目標姿勢: {np.round(target_ori, 4)}")
            
            # IKを実行
            seed = [0.0, 0.0]  # 初期推定値
            ik_solution = self.kdl_helper.inverse_kinematics(target_pos, target_ori, seed=seed)
            
            if ik_solution is not None:
                self.logger.info(f"IK解: {np.round(ik_solution, 4)}")
                
                # IK解をFKで検証
                check_pose = self.kdl_helper.forward_kinematics(ik_solution)
                pos_error = np.linalg.norm(check_pose[:3] - target_pos)
                
                if pos_error < 1e-3:  # 1mm以下の誤差なら成功
                    self.add_test_result("IK", True, f"成功 (誤差: {pos_error:.6f})")
                else:
                    self.add_test_result("IK", False, f"精度不足 (誤差: {pos_error:.6f})")
            else:
                self.add_test_result("IK", False, "解が見つからない")
                
        except Exception as e:
            self.add_test_result("IK", False, f"例外発生: {e}")
    
    def test_jacobian(self):
        """ヤコビアン行列テスト"""
        self.logger.info("テスト4: ヤコビアン行列テスト")
        
        try:
            joint_angles = [0.05, 0.1]
            
            jacobian = self.kdl_helper.jacobian(joint_angles)
            self.logger.info(f"ヤコビアン行列サイズ: {jacobian.shape}")
            self.logger.info(f"ヤコビアン行列:\n{np.round(jacobian, 4)}")
            
            # ヤコビアンのサイズ確認 (6x関節数)
            expected_shape = (6, self.kdl_helper._num_jnts)
            if jacobian.shape == expected_shape:
                self.add_test_result("ヤコビアン", True, "成功")
            else:
                self.add_test_result("ヤコビアン", False, f"サイズ異常: 期待{expected_shape}, 実際{jacobian.shape}")
            
            # 疑似逆行列テスト
            pinv_jacobian = self.kdl_helper.jacobian_pseudo_inverse(joint_angles)
            expected_pinv_shape = (self.kdl_helper._num_jnts, 6)
            if pinv_jacobian.shape == expected_pinv_shape:
                self.add_test_result("ヤコビアン疑似逆行列", True, "成功")
            else:
                self.add_test_result("ヤコビアン疑似逆行列", False, f"サイズ異常: 期待{expected_pinv_shape}, 実際{pinv_jacobian.shape}")
                
        except Exception as e:
            self.add_test_result("ヤコビアン", False, f"例外発生: {e}")
    
    def test_inertia_matrix(self):
        """慣性行列テスト"""
        self.logger.info("テスト5: 慣性行列テスト")
        
        try:
            joint_angles = [0.02, 0.05]
            
            inertia = self.kdl_helper.inertia_matrix(joint_angles)
            self.logger.info(f"慣性行列サイズ: {inertia.shape}")
            self.logger.info(f"慣性行列:\n{np.round(inertia, 4)}")
            
            # 慣性行列のサイズ確認 (関節数x関節数)
            expected_shape = (self.kdl_helper._num_jnts, self.kdl_helper._num_jnts)
            if inertia.shape == expected_shape:
                # 慣性行列は正定値対称行列であるべき
                is_symmetric = np.allclose(inertia, inertia.T, atol=1e-6)
                eigenvals = np.linalg.eigvals(inertia)
                is_positive_definite = np.all(eigenvals > 0)
                
                if is_symmetric and is_positive_definite:
                    self.add_test_result("慣性行列", True, "成功 (正定値対称)")
                else:
                    self.add_test_result("慣性行列", False, f"行列特性異常 (対称: {is_symmetric}, 正定値: {is_positive_definite})")
            else:
                self.add_test_result("慣性行列", False, f"サイズ異常: 期待{expected_shape}, 実際{inertia.shape}")
                
        except Exception as e:
            self.add_test_result("慣性行列", False, f"例外発生: {e}")
    
    def test_error_handling(self):
        """エラーハンドリングテスト"""
        self.logger.info("テスト6: エラーハンドリングテスト")
        
        try:
            # 間違った関節数でのテスト
            wrong_joint_angles = [0.0, 0.0, 0.0]  # 3つの値 (正しくは2つ)
            
            try:
                self.kdl_helper.forward_kinematics(wrong_joint_angles)
                self.add_test_result("エラーハンドリング (FK)", False, "例外が発生しなかった")
            except ValueError:
                self.add_test_result("エラーハンドリング (FK)", True, "適切にValueErrorが発生")
            except Exception as e:
                self.add_test_result("エラーハンドリング (FK)", False, f"予期しない例外: {e}")
            
            # 無効なIK目標でのテスト
            impossible_pos = [100.0, 100.0, 100.0]  # 到達不可能な位置
            ik_result = self.kdl_helper.inverse_kinematics(impossible_pos, seed=[0.0, 0.0])
            
            if ik_result is None:
                self.add_test_result("エラーハンドリング (IK)", True, "到達不可能位置でNone返却")
            else:
                self.add_test_result("エラーハンドリング (IK)", False, "到達不可能位置で解が返された")
                
        except Exception as e:
            self.add_test_result("エラーハンドリング", False, f"予期しない例外: {e}")
    
    def test_helper_functions(self):
        """補助関数のテスト"""
        self.logger.info("テスト1.5: 補助関数テスト")
        
        try:
            # quaternion_matrix関数のテスト
            unit_quat = [0.0, 0.0, 0.0, 1.0]  # 単位クォータニオン
            matrix = quaternion_matrix(unit_quat)
            
            expected_matrix = np.eye(4)
            if np.allclose(matrix, expected_matrix, atol=1e-6):
                self.add_test_result("quaternion_matrix (単位)", True, "成功")
            else:
                self.add_test_result("quaternion_matrix (単位)", False, "単位クォータニオンで単位行列にならない")
            
            # 他のクォータニオンでのテスト
            test_quat = [0.0, 0.0, 0.7071, 0.7071]  # 90度回転（Z軸周り）
            matrix2 = quaternion_matrix(test_quat)
            
            # 回転行列の基本的性質をチェック
            det = np.linalg.det(matrix2[:3, :3])
            is_orthogonal = np.allclose(matrix2[:3, :3] @ matrix2[:3, :3].T, np.eye(3), atol=1e-6)
            
            if np.isclose(det, 1.0, atol=1e-6) and is_orthogonal:
                self.add_test_result("quaternion_matrix (回転)", True, f"成功 (det={det:.6f})")
            else:
                self.add_test_result("quaternion_matrix (回転)", False, f"回転行列の性質が不正 (det={det:.6f}, orthogonal={is_orthogonal})")
                
        except Exception as e:
            self.add_test_result("補助関数", False, f"例外発生: {e}")

    def test_debug_functionality(self):
        """デバッグ機能のテスト"""
        if hasattr(self, 'kdl_helper'):
            self.logger.info("テスト7: デバッグ機能テスト")
            
            try:
                test_joints = [0.05, 0.1]
                
                # デバッグ付きFK
                self.logger.info("デバッグ付き順運動学:")
                self.kdl_helper.forward_kinematics(test_joints, debug=True)
                
                # デバッグ付きIK
                target_pose = self.kdl_helper.forward_kinematics(test_joints)
                self.logger.info("デバッグ付き逆運動学:")
                self.kdl_helper.inverse_kinematics(
                    target_pose[:3], target_pose[3:7], 
                    seed=[0.0, 0.0], debug=True
                )
                
                # デバッグ付きヤコビアン
                self.logger.info("デバッグ付きヤコビアン:")
                self.kdl_helper.jacobian(test_joints, debug=True)
                
                # デバッグ付き慣性行列
                self.logger.info("デバッグ付き慣性行列:")
                self.kdl_helper.inertia_matrix(test_joints, debug=True)
                
                self.add_test_result("デバッグ機能", True, "全デバッグ機能が正常動作")
                
            except Exception as e:
                self.add_test_result("デバッグ機能", False, f"例外発生: {e}")

    def add_test_result(self, test_name, success, message):
        """テスト結果を記録"""
        self.test_results.append((test_name, success, message))
        status = "✓ 成功" if success else "✗ 失敗"
        self.logger.info(f"  {status}: {test_name} - {message}")
    
    def print_test_summary(self):
        """テスト結果のサマリーを表示"""
        self.logger.info("=" * 50)
        self.logger.info("テスト結果サマリー")
        self.logger.info("=" * 50)
        
        passed = sum(1 for _, success, _ in self.test_results if success)
        total = len(self.test_results)
        
        for test_name, success, message in self.test_results:
            status = "✓ 成功" if success else "✗ 失敗"
            self.logger.info(f"{status}: {test_name} - {message}")
        
        self.logger.info("-" * 50)
        self.logger.info(f"結果: {passed}/{total} テスト成功")
        
        if passed == total:
            self.logger.info("🎉 全てのテストが成功しました！")
        else:
            self.logger.warn(f"⚠️ {total - passed} 個のテストが失敗しました")


class TestNode(Node):
    """テスト実行用のROS2ノード"""
    
    def __init__(self):
        super().__init__('kdl_helper_test_node')
        
        self.get_logger().info("KDLHelper テストノード開始")
        
        # テスト実行
        tester = TestKDLHelper(self.get_logger())
        success = tester.run_all_tests()
        
        if success:
            self.get_logger().info("全テスト完了: 成功")
        else:
            self.get_logger().error("全テスト完了: 一部失敗")


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = TestNode()
        # テスト完了後にシャットダウン
    except Exception as e:
        print(f"テスト実行エラー: {e}")
        traceback.print_exc()
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()