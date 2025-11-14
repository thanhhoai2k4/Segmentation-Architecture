import json
import math
import os.path
import cv2
import numpy as np
from dataclasses import dataclass
from sklearn.decomposition import PCA

@dataclass
class RawObject:
    """
        Represents a raw object from YOLO’s JSON output
    """

    category: str
    score: float
    bbox: list[float] # [x1,y1,x2,y2]
    polygon: list[float] # [x1,y1, x2,y2 .....]

@dataclass
class WallFeature:
    """
        Wall
    """
    id: str
    start_point: tuple[float, float]
    end_point: tuple[float, float]
    thickness: float # pixel

@dataclass
class OpeningFeature:
    """
        Opening
    """
    id: str
    category: str
    location_point: tuple[float, float]
    width: float
    host_wall_id: str | None = None


class FeatureExtractor:
    """
    Feature Extractor
    """
    def __init__(self, image_width: int, image_height: int):
        self.image_width = image_width
        self.image_height = image_height
        self.raw_objects: list[RawObject] = []
        self.processed_walls: list[WallFeature] = []
        self.processed_openings : list[OpeningFeature] = []

    def load_raw_json(self, json_path: str):

        # check exists of json_path
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"file not found: {json_path}")
        # load json file raw.
        with open(json_path, "r", encoding="utf-8") as f:
            raw_json = json.load(f)
            for item in raw_json:
                self.raw_objects.append(
                    RawObject(
                        category=item['category_name'],
                        score=item['score'],
                        bbox=item['bbox_xyxy'],
                        polygon=item['segmentation_polygon']
                    )
                )
        print(f"Loaded raw json with : {len(self.raw_objects)} objects raw")

    def process_features(self):
        """
            processing feature to super feature.
        """

        print("start processing features")
        walls_raw = [obj for obj in self.raw_objects if obj.category == "Wall"]
        openings_raw = [obj for obj in self.raw_objects if obj.category in ['Door', 'Window']] # Door ~~~~ Window

        # process Wall.
        for i, wall_obj in enumerate(walls_raw):
            feature = self._process_wall(wall_obj, f"wall_{i}")
            if feature:
                self.processed_walls.append(feature)

        for i, wall_obj in enumerate(openings_raw):
            feature = self._process_opening(wall_obj, f"opening_{i}")
            if feature:
                self.processed_openings.append(feature)



        # create relationship.
        self._establish_relationships()
        print("finished processing features")

        # (Đây là bên trong class FeatureExtractor)

    def visualize_results(self, image: np.ndarray, show_window: bool = True):
            """
            Vẽ các WallFeature và OpeningFeature đã xử lý lên trên một ảnh (đã load).
            Hàm này dùng để gỡ lỗi (debug) trực quan.

            :param image: Ảnh (numpy array) đã được load bằng cv2.imread.
            :param show_window: Nếu True, sẽ hiển thị ảnh bằng cv2.imshow.
            :return: Trả về ảnh đã được vẽ (debug_image).
            """

            # Tạo bản sao để không vẽ lên ảnh gốc
            debug_image = image.copy()

            # Cấu hình font chữ
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_color = (255, 255, 255)  # Màu trắng
            line_type = 2

            print("\nBat dau ve ket qua debug...")

            # 1. Vẽ các Tường (Walls)
            for wall in self.processed_walls:
                try:
                    # Chuyển tọa độ float sang int cho OpenCV
                    sp = (int(wall.start_point[0]), int(wall.start_point[1]))
                    ep = (int(wall.end_point[0]), int(wall.end_point[1]))

                    # Vẽ đường tâm (Centerline) - Màu Đỏ
                    cv2.line(debug_image, sp, ep, (0, 0, 255), 2)  # BGR = Red

                    # Vẽ điểm bắt đầu (Start Point) - Màu Xanh Lá
                    cv2.circle(debug_image, sp, 8, (0, 255, 0), -1)  # Green (Filled)
                    cv2.putText(debug_image, f"S: {sp}", (sp[0] + 10, sp[1]),
                                font, font_scale, font_color, line_type)

                    # Vẽ điểm kết thúc (End Point) - Màu Xanh Dương
                    cv2.circle(debug_image, ep, 8, (255, 0, 0), -1)  # Blue (Filled)
                    cv2.putText(debug_image, f"E: {ep}", (ep[0] + 10, ep[1]),
                                font, font_scale, font_color, line_type)
                except Exception as e:
                    print(f"Loi khi ve tuong {wall.id}: {e}")

            # 2. Vẽ các Cửa (Openings)
            for opening in self.processed_openings:
                try:
                    lp = (int(opening.location_point[0]), int(opening.location_point[1]))

                    # Vẽ dấu X (Marker) - Màu Vàng
                    cv2.drawMarker(debug_image, lp, (0, 255, 255),
                                   markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)  # Yellow

                    host_id_str = opening.host_wall_id if opening.host_wall_id else "None"
                    text = f"{opening.id} (Host: {host_id_str})"
                    cv2.putText(debug_image, text, (lp[0] + 15, lp[1] - 15),
                                font, font_scale, (0, 255, 255), line_type)  # Yellow
                except Exception as e:
                    print(f"Loi khi ve cua {opening.id}: {e}")

            # 3. Hiển thị ảnh
            if show_window:
                print("Dang hien thi cua so ket qua. Nhan phim bat ky de dong...")
                # Thay đổi kích thước cửa sổ nếu ảnh quá lớn
                h, w = debug_image.shape[:2]
                max_h = 800
                if h > max_h:
                    scale = max_h / h
                    debug_image_resized = cv2.resize(debug_image, None, fx=scale, fy=scale)
                    cv2.imshow("Processed Results Visualization", debug_image_resized)
                else:
                    cv2.imshow("Processed Results Visualization", debug_image)

                cv2.waitKey(0)
                cv2.destroyAllWindows()

            print("Ve ket qua debug hoan tat.")
            return debug_image




    def _process_wall(self, wall_obj: RawObject, obj_id: str) -> WallFeature | None:

        print(f"processing {obj_id}....")

        try:
            # --- 1) Thử dùng polygon nếu có ---
            poly_flat = getattr(wall_obj, "polygon", None)
            if poly_flat and isinstance(poly_flat, (list, tuple)) and len(poly_flat) >= 6:
                try:
                    pts = np.array(poly_flat, dtype=float).reshape(-1, 2)  # shape (N,2)
                except Exception:
                    pts = None

                if pts is not None and pts.shape[0] >= 3:
                    # Center the points
                    mean = pts.mean(axis=0)
                    pts_centered = pts - mean  # (N,2)

                    # Run PCA (keep first component)
                    try:
                        pca = PCA(n_components=2)
                        pca.fit(pts_centered)  # fit to centered pts
                        # principal axis (unit vector) in original coordinates
                        pc1 = pca.components_[0]  # length-2 unit vector
                    except Exception:
                        pc1 = None

                    if pc1 is not None:
                        # Project centered points onto the principal axis
                        projections = pts_centered.dot(pc1)  # (N,) scalar projecting onto axis
                        min_proj = projections.min()
                        max_proj = projections.max()

                        # Endpoints in original coordinates (world coords)
                        start = mean + pc1 * min_proj
                        end = mean + pc1 * max_proj

                        # For thickness: compute projection onto the perpendicular direction
                        # perpendicular vector (unit)
                        perp = np.array([-pc1[1], pc1[0]])
                        perp_projs = pts_centered.dot(perp)
                        # thickness is range of perp projections (absolute distance)
                        perp_min = perp_projs.min()
                        perp_max = perp_projs.max()
                        thickness = float(abs(perp_max - perp_min))

                        # If thickness is zero or extremely small (degenerate), fallback to bbox below
                        if thickness <= 0.5:
                            # degenerate polygon: set thickness to small positive and continue to bbox fallback
                            thickness = 0.0

                        # Return WallFeature if thickness sensible
                        if thickness > 0:
                            return WallFeature(
                                id=obj_id,
                                start_point=(float(start[0]), float(start[1])),
                                end_point=(float(end[0]), float(end[1])),
                                thickness=thickness
                            )
                        # else fall through to bbox fallback
            # --- 2) Fallback: dùng bbox ---
            bbox = getattr(wall_obj, "bbox", None)
            if not bbox or len(bbox) < 4:
                return None

            x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            width = x2 - x1
            height = y2 - y1
            if width <= 0 or height <= 0:
                return None

            if width > height:
                # horizontal
                cy = y1 + height / 2.0
                start_point_xy = (float(x1), float(cy))
                end_point_xy = (float(x2), float(cy))
                thickness = float(height)
            else:
                cx = x1 + width / 2.0
                start_point_xy = (float(cx), float(y1))
                end_point_xy = (float(cx), float(y2))
                thickness = float(width)

            return WallFeature(id=obj_id, start_point=start_point_xy, end_point=end_point_xy, thickness=thickness)

        except Exception as e:
            # không làm rớt chương trình, in lỗi để debug
            print(f"Error processing wall {obj_id}: {e}")
            return None

    def _process_opening(self, opening_obj: RawObject, obj_id: str) -> OpeningFeature | None:
        """
            Xữ lý cửa/cửa sổ: Đơn giản là lấy trung tâm và chiều rộng.
        """

        try:
            bbox = opening_obj.bbox
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            width = bbox[2] - bbox[0]

            return OpeningFeature(
                id = obj_id,
                category = opening_obj.category,
                location_point = (cx, cy),
                width = width,
            )
        except Exception as e:
            print(f"Error while processing { obj_id } : {e}")
            return None

    def _distance_points(self, p1: tuple[float, float], p2: tuple[float, float]) -> float:
        """
            Tinh khoang cach Eculidean giua 2 diem
            D(x,y) = sqrt(sum((xi-y1)^2)/n)
        """

        return math.sqrt(
            (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2
        )

    def _distance_point_to_segment(self,
                                   p: tuple[float, float],
                                   s1: tuple[float, float],
                                   s2: tuple[float, float]) -> float:

        """
            Tinh khaong cach ngan nhat tu dien p Den doan thang s1-s2

        """

        l2 = (s2[0] - s1[0]) ** 2 + (s2[1] - s1[1]) ** 2
        if l2 == 0:
            return self._distance_points(p, s1)

        t = ((p[0] - s1[0]) * (s2[0] - s1[0]) + (p[1] - s1[1]) * (s2[1] - s1[1])) / l2
        t = max(0, min(1, t))

        projection = (s1[0] + t * (s2[0] - s1[0]), s1[1] + t * (s2[1] - s1[1]))
        return self._distance_points(p, projection)

    def _get_wall_vector(self, wall: WallFeature) -> tuple[float, float]:
        """Lấy vector chỉ phương của tường."""
        vec = (wall.end_point[0] - wall.start_point[0], wall.end_point[1] - wall.start_point[1])
        length = math.sqrt(vec[0] ** 2 + vec[1] ** 2)
        if length == 0:
            return (0, 0)
        return (vec[0] / length, vec[1] / length)

    def _are_collinear(self, wall1: WallFeature, wall2: WallFeature, angle_tolerance_deg: float = 5.0) -> bool:
        """Kiểm tra 2 tường có thẳng hàng (cùng phương) không."""
        v1 = self._get_wall_vector(wall1)
        v2 = self._get_wall_vector(wall2)

        dot_product = abs(v1[0] * v2[0] + v1[1] * v2[1])

        # cos(angle)
        if dot_product > math.cos(math.radians(angle_tolerance_deg)):
            return True
        return False

    def _establish_relationships(self):
        """
        Hợp nhất các bức tường bị chia cắt bởi cửa (Door/Window) và
        gán `host_wall_id` cho các cửa đó.
        """
        print("Establishing relationships...")

        # Tạo một bản sao danh sách tường để làm việc
        # Sử dụng dict để dễ dàng thêm/xóa/cập nhật
        final_walls_map = {wall.id: wall for wall in self.processed_walls}
        openings_to_process = self.processed_openings[:]

        merged_wall_counter = 0
        processed_openings = set()  # Theo dõi các opening đã xử lý việc merge

        for i in range(len(openings_to_process)):
            opening = openings_to_process[i]
            if opening.id in processed_openings:
                continue

            # 1. Tìm các tường gần nhất với opening này
            wall_distances = []
            for wall_id, wall in final_walls_map.items():
                dist = self._distance_point_to_segment(opening.location_point, wall.start_point, wall.end_point)
                wall_distances.append((dist, wall))

            wall_distances.sort(key=lambda x: x[0])

            if not wall_distances:
                continue  # Không có tường nào

            # 2. Kiểm tra kịch bản "2 tường, 1 cửa"

            # Dung sai khoảng cách: Nếu tường ở trong phạm vi (chiều rộng cửa + độ dày tường),
            # nó có thể là một phần của kịch bản merge
            merge_distance_threshold = opening.width * 1.5

            nearby_walls = [wall for dist, wall in wall_distances if dist < merge_distance_threshold]

            if len(nearby_walls) >= 2:
                # Tìm 2 tường gần nhất mà thẳng hàng
                wall1 = nearby_walls[0]
                wall2 = None
                for j in range(1, len(nearby_walls)):
                    if self._are_collinear(wall1, nearby_walls[j]):
                        wall2 = nearby_walls[j]
                        break  # Tìm thấy cặp đầu tiên

                if wall1 and wall2:
                    # Đã tìm thấy 2 tường (wall1, wall2) bị chia cắt bởi opening
                    print(f"Found merge candidate for {opening.id}: ({wall1.id}, {wall2.id})")

                    # 3. Thực hiện hợp nhất (Merge)
                    # Tìm 4 điểm đầu cuối
                    points = [wall1.start_point, wall1.end_point, wall2.start_point, wall2.end_point]

                    # Tìm 2 điểm ngoài cùng (xa nhau nhất)
                    max_d = 0
                    merged_start, merged_end = points[0], points[1]
                    for p_i in range(4):
                        for p_j in range(p_i + 1, 4):
                            d = self._distance_points(points[p_i], points[p_j])
                            if d > max_d:
                                max_d = d
                                merged_start = points[p_i]
                                merged_end = points[p_j]

                    # Tạo tường mới
                    merged_id = f"wall_merged_{merged_wall_counter}"
                    merged_wall_counter += 1
                    merged_wall = WallFeature(
                        id=merged_id,
                        start_point=merged_start,
                        end_point=merged_end,
                        thickness=(wall1.thickness + wall2.thickness) / 2  # Lấy trung bình
                    )

                    # Cập nhật danh sách tường
                    final_walls_map[merged_id] = merged_wall

                    # Xóa 2 tường cũ
                    if wall1.id in final_walls_map:
                        del final_walls_map[wall1.id]
                    if wall2.id in final_walls_map:
                        del final_walls_map[wall2.id]

                    # Gán host cho opening
                    opening.host_wall_id = merged_id
                    processed_openings.add(opening.id)

        # 4. Gán host cho các opening còn lại (không nằm trong kịch bản merge)
        for opening in openings_to_process:
            if opening.id in processed_openings:
                continue  # Đã xử lý

            # Tìm tường gần nhất từ danh sách TƯỜNG CUỐI CÙNG
            min_dist = float('inf')
            best_wall_id = None
            for wall_id, wall in final_walls_map.items():
                dist = self._distance_point_to_segment(opening.location_point, wall.start_point, wall.end_point)
                if dist < min_dist:
                    min_dist = dist
                    best_wall_id = wall_id

            # Gán host nếu đủ gần
            if best_wall_id and min_dist < opening.width * 2:  # Dung sai
                opening.host_wall_id = best_wall_id

        # update list wall.
        self.processed_walls = list(final_walls_map.values())
        print(f"Relationships established. Final wall count: {len(self.processed_walls)}")

    def export_for_revit(self, output_path: str, scale_hint: str):
        pass