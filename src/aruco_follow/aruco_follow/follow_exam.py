import math
import rclpy

from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

import cv2
import numpy as np


class ArucoLidarFollower(Node):
    def __init__(self):
        super().__init__('aruco_lidar_follower')

        # ===== 제어 파라미터 =====
        self.target_distance = 0.3      # ArUco 마커와 유지할 목표 거리 [m]
        self.marker_size_m = 0.04       # 마커 한 변 실제 길이 [m]

        # 오차 속도 반영 (비례 이득 gain)
        self.linear_kp = 0.4            # 거리 제어 비례 이득 
        self.angular_kp = 0.0025        # 회전 제어 비례 이득 

        # 속도 상한
        self.max_linear = 0.15          # [m/s] 전/후진 최대 속도
        self.max_angular = 0.3          # [rad/s] 회전 최대 속도

        # 카메라 수평 FOV (대략값, 필요 시 실제 값으로 수정)
        self.camera_fov_deg = 60.0

        # LiDAR 안전 파라미터
        self.safety_deg = 200.0         # LiDAR 스캔에 사용할 각도 범위(전방 기준 ±100도)
        self.safety_distance = 0.20     # 이 거리 안에 장애물이 있으면 전진 차단 [m]

        # ===== 상태 변수 =====
        self.bridge = CvBridge()        # ROS Image 메시지 ↔ OpenCV 이미지 변환용

        # LiDAR 최신 메시지
        self.last_scan = None

        # ArUco 마커 상태
        self.marker_visible = False     # 마커가 보이면 True로 전환
        self.marker_pixel_error_x = 0.0 # 마커 중심이 화면 정중앙에서 벗어난 픽셀 오차
        self.image_width = 1            # 0으로 나누기 방지용
        self.fx = None                  # 카메라 초점거리 계산(픽셀 단위)
        self.marker_distance_cam = None # 카메라로 계산한 마커 거리 [m]

        # 진동 필터링 (EMA = Exponential Moving Average)
        self.filtered_distance = None
        self.filtered_pixel_error = None
        self.alpha_dist = 0.2           # 거리 필터 강도
        self.alpha_pix = 0.3            # 픽셀 오차 필터 강도

        # 가속도 제한
        self.prev_linear = 0.0          # 로봇에게 마지막으로 보낸 선속도
        self.prev_angular = 0.0         # 로봇에게 마지막으로 보낸 각속도
        self.max_linear_step = 0.02     # 한 주기당 선속도 최대 변화량
        self.max_angular_step = 0.05    # 한 주기당 각속도 최대 변화량

        # ===== ArUco 설정 =====
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict,self.aruco_params)

        # ===== 구독자 설정 =====
        self.image_sub = self.create_subscription(Image,'/camera/color/image_raw',self.image_callback,10)
        self.scan_sub = self.create_subscription(LaserScan,'/scan',self.scan_callback,10)

        # ===== 속도 퍼블리셔 =====
        self.cmd_pub = self.create_publisher(Twist,'/cmd_vel',10)

        # ===== 제어 루프 타이머 =====
        self.control_timer = self.create_timer(0.1,self.control_loop)


        self.get_logger().info('ArucoLidarFollower node started.')

    # ------------------------------
    #  LiDAR 콜백: 최신 스캔만 저장
    # ------------------------------
    def scan_callback(self, msg: LaserScan):
        self.last_scan = msg

    # ------------------------------
    #  카메라 콜백: 마커 위치 + 크기로 거리 계산
    # ------------------------------
    def image_callback(self, msg: Image):
        # ROS Image → OpenCV BGR
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_h, img_w = gray.shape  # gray.shape 는 (height, width) 를 반환한다.
        self.image_width = img_w

        # 카메라 초점거리 fx 계산 (FOV 기반, 한 번만 계산)
        if self.fx is None:
            fov_rad = math.radians(self.camera_fov_deg)
            self.fx = (img_w / 2.0) / math.tan(fov_rad / 2.0)
            self.get_logger().info(f"Computed fx={self.fx:.2f} from FOV={self.camera_fov_deg} deg")

        corners, ids, _ = self.aruco_detector.detectMarkers(gray)
        # coners = 감지된 각 마커의 네 모서리 좌표
        # ids = 감지된 마커들의 ID 목록

        if ids is None or len(corners) == 0:
            # 마커가 보이지 않는 상태
            self.marker_visible = False
            self.marker_distance_cam = None
            return

        # 첫번쨰로 인식된 마커 사용
        marker_corners = corners[0][0]

        # 마커 중심 좌표
        marker_cx = float(np.mean(marker_corners[:, 0]))
        image_center_x = img_w / 2.0
        pixel_error_x = marker_cx - image_center_x

        # ---- 픽셀 오차 필터링 (EMA) ----
        if self.filtered_pixel_error is None:
            self.filtered_pixel_error = pixel_error_x
        else:
            self.filtered_pixel_error = (
                (1.0 - self.alpha_pix) * self.filtered_pixel_error
                + self.alpha_pix * pixel_error_x
            )

        self.marker_visible = True
        self.marker_pixel_error_x = self.filtered_pixel_error

        # ===== 마커 크기(픽셀)로 카메라-마커 거리 추정 =====
        # 네 변의 길이 평균을 사용
        side01 = np.linalg.norm(marker_corners[0] - marker_corners[1])
        side12 = np.linalg.norm(marker_corners[1] - marker_corners[2])
        side23 = np.linalg.norm(marker_corners[2] - marker_corners[3])
        side30 = np.linalg.norm(marker_corners[3] - marker_corners[0])
        marker_size_px = (side01 + side12 + side23 + side30) / 4.0

        if self.fx is not None and marker_size_px > 0.0 and self.marker_size_m > 0.0:
            # 거리 Z ≈ fx * S_real / S_pixel
            distance_cam = self.fx * self.marker_size_m / marker_size_px

            # ---- 거리 필터링 (EMA) ----
            if self.filtered_distance is None:
                self.filtered_distance = distance_cam
            else:
                self.filtered_distance = (
                    (1.0 - self.alpha_dist) * self.filtered_distance
                    + self.alpha_dist * distance_cam
                )

            self.marker_distance_cam = float(self.filtered_distance)
        else:
            self.marker_distance_cam = None

    # ------------------------------
    #  LiDAR: 스캔 범위에서 최소 거리 추출 (안전용)
    # ------------------------------
    def get_min_range_from_scan(self, scan: LaserScan):
        if scan is None:
            return None

        # 정면 기준 양쪽의 각도
        half = math.radians(self.safety_deg / 2.0)
        start_angle = -half
        end_angle = half

        start_idx = int(round((start_angle - scan.angle_min) / scan.angle_increment))
        end_idx = int(round((end_angle - scan.angle_min) / scan.angle_increment))

        # 인덱스 클램프 (0 ~ len-1 사이에 넣는다)
        start_idx = max(0, min(len(scan.ranges) - 1, start_idx))
        end_idx = max(0, min(len(scan.ranges) - 1, end_idx))
        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx

        values = []
        for r in scan.ranges[start_idx:end_idx + 1]:
            # 1) inf / NaN 값 제거
            if math.isinf(r) or math.isnan(r):
                continue
            # 2) 3cm(0.03m) 이내의 값은 센서 오류(먼지, 고장 픽셀 등)로 간주하고 무시
            if r <= 0.03:
                continue
            values.append(r)

        if not values:
            # 유효한 값이 하나도 없으면 None 반환 → 안전 제어 쪽에서 막지 않도록
            return None

        return min(values)

    # ------------------------------
    #  주기적 제어 루프
    # ------------------------------
    def control_loop(self):
        twist = Twist()

        # 카메라에서 마커가 안 보이면 정지
        if not self.marker_visible or self.marker_distance_cam is None:
            self.prev_linear = 0.0
            self.prev_angular = 0.0
            self.cmd_pub.publish(twist)
            return

        # ===== 1. 카메라 기반 거리 제어 =====
        distance = self.marker_distance_cam
        dist_error = distance - self.target_distance   # +면 멀다 → 전진, -면 가깝다 → 후진

        # --- 거리 데드존 (±3cm) ---
        if abs(dist_error) < 0.03:
            linear_x = 0.0
        else:
            linear_x = self.linear_kp * dist_error

        # ===== 2. 화면 중심 기준 회전 제어 =====
        pixel_err = self.marker_pixel_error_x

        # --- 회전 데드존 (±10픽셀) ---
        if abs(pixel_err) < 10.0:
            angular_z = 0.0
        else:
            angular_z = -self.angular_kp * pixel_err

        # ===== 3. LiDAR 안전 제어 (스캔 범위 최소 거리 체크) =====
        min_range = self.get_min_range_from_scan(self.last_scan)

        # 주변에 장애물이 있으면, 카메라 제어를 모두 무시하고 로봇을 완전히 정지시킨다.
        if min_range is not None and min_range < self.safety_distance:
            self.get_logger().info(
                f"[SAFETY] obstacle within {min_range:.3f} m → Robot Stopped (full stop)"
            )

            # 과거 속도 기록도 0으로 초기화 (가속도 제한 때문에 남아 있는 값 제거)
            self.prev_linear = 0.0
            self.prev_angular = 0.0

            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.cmd_pub.publish(twist)
            return

        # ===== 4. 속도 제한 (상한/하한) =====
        linear_x = max(-self.max_linear, min(self.max_linear, linear_x))
        angular_z = max(-self.max_angular, min(self.max_angular, angular_z))

        # ===== 5. 가속도 제한 (램프 함수) =====
        def ramp(prev, target, step):
            diff = target - prev
            if diff > step:
                diff = step
            elif diff < -step:
                diff = -step
            return prev + diff

        linear_x = ramp(self.prev_linear, linear_x, self.max_linear_step)
        angular_z = ramp(self.prev_angular, angular_z, self.max_angular_step)

        self.prev_linear = linear_x
        self.prev_angular = angular_z

        # 디버그용 로그
        self.get_logger().info(
            f"[CTRL] cam_dist={distance:.3f}, err={dist_error:.3f}, "
            f"range_min={min_range}, lin={linear_x:.3f}, ang={angular_z:.3f}"
        )

        twist.linear.x = linear_x
        twist.angular.z = angular_z
        self.cmd_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = ArucoLidarFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
