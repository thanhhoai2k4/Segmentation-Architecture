# import json
# import math
# import os.path
# import cv2
# import numpy as np
# from dataclasses import dataclass
# import topologicpy as tp
# from topologicpy.Vertex import Vertex
# from topologicpy.Wire import Wire
# from topologicpy.Topology import Topology
# @dataclass
# class RawObject:
#     """
#         Represents a raw object from YOLO’s JSON output
#     """
#
#     category: str
#     score: float
#     bbox: list[float] # [x1,y1,x2,y2]
#     polygon: list[float] # [x1,y1, x2,y2 .....]
#
# @dataclass
# class WallFeature:
#     """
#         Wall
#     """
#     id: str
#     start_point: tuple[float, float]
#     end_point: tuple[float, float]
#     thickness: float # pixel
#
# @dataclass
# class OpeningFeature:
#     """
#         Opening
#     """
#     id: str
#     category: str
#     location_point: tuple[float, float]
#     width: float
#     host_wall_id: str | None = None
#
#
# class FeatureExtractor:
#     """
#     Feature Extractor
#     """
#     def __init__(self, image_width: int, image_height: int):
#         self.image_width = image_width
#         self.image_height = image_height
#         self.raw_objects: list[RawObject] = []
#         self.processed_walls: list[WallFeature] = []
#         self.processed_openings : list[OpeningFeature] = []
#
#         # store topo element.
#         # --------
#         self.topo_features: dict[str, ] = {}
#         # --------
#
#     def load_raw_json(self, json_path: str):
#
#         # check exists of json_path
#         if not os.path.isfile(json_path):
#             raise FileNotFoundError(f"file not found: {json_path}")
#         # load json file raw.
#         with open(json_path, "r", encoding="utf-8") as f:
#             raw_json = json.load(f)
#             for item in raw_json:
#                 self.raw_objects.append(
#                     RawObject(
#                         category=item['category_name'],
#                         score=item['score'],
#                         bbox=item['bbox_xyxy'],
#                         polygon=item['segmentation_polygon']
#                     )
#                 )
#         print(f"Loaded raw json with : {len(self.raw_objects)} objects raw")
#
#     def process_features(self):
#         """
#             processing feature to super feature.
#         """
#
#         print("start processing features")
#         walls_raw = [obj for obj in self.raw_objects if obj.category == "Wall"]
#         openings_raw = [obj for obj in self.raw_objects if obj.category in ['Door', 'Window']] # Door ~~~~ Window
#
#         # process Wall.
#         for i, wall_obj in enumerate(walls_raw):
#             feature = self._process_wall(wall_obj, f"wall_{i}")
#             if feature:
#                 self.processed_walls.append(feature)
#
#         for i, wall_obj in enumerate(openings_raw):
#             feature = self._process_opening(wall_obj, f"opening_{i}")
#             if feature:
#                 self.processed_openings.append(feature)
#
#
#
#         # create relationship.
#         self._establish_relationships()
#         print("finished processing features")
#
#         # (Đây là bên trong class FeatureExtractor)
#
#     def visualize_results(self, image: np.ndarray, show_window: bool = True):
#             """
#             Vẽ các WallFeature và OpeningFeature đã xử lý lên trên một ảnh (đã load).
#             Hàm này dùng để gỡ lỗi (debug) trực quan.
#
#             :param image: Ảnh (numpy array) đã được load bằng cv2.imread.
#             :param show_window: Nếu True, sẽ hiển thị ảnh bằng cv2.imshow.
#             :return: Trả về ảnh đã được vẽ (debug_image).
#             """
#
#             # Tạo bản sao để không vẽ lên ảnh gốc
#             debug_image = image.copy()
#
#             # Cấu hình font chữ
#             font = cv2.FONT_HERSHEY_SIMPLEX
#             font_scale = 0.5
#             font_color = (255, 255, 255)  # Màu trắng
#             line_type = 2
#
#             print("\nBat dau ve ket qua debug...")
#
#             # 1. Vẽ các Tường (Walls)
#             for wall in self.processed_walls:
#                 try:
#                     # Chuyển tọa độ float sang int cho OpenCV
#                     sp = (int(wall.start_point[0]), int(wall.start_point[1]))
#                     ep = (int(wall.end_point[0]), int(wall.end_point[1]))
#
#                     # Vẽ đường tâm (Centerline) - Màu Đỏ
#                     cv2.line(debug_image, sp, ep, (0, 0, 255), 2)  # BGR = Red
#
#                     # Vẽ điểm bắt đầu (Start Point) - Màu Xanh Lá
#                     cv2.circle(debug_image, sp, 8, (0, 255, 0), -1)  # Green (Filled)
#                     cv2.putText(debug_image, f"S: {sp}", (sp[0] + 10, sp[1]),
#                                 font, font_scale, font_color, line_type)
#
#                     # Vẽ điểm kết thúc (End Point) - Màu Xanh Dương
#                     cv2.circle(debug_image, ep, 8, (255, 0, 0), -1)  # Blue (Filled)
#                     cv2.putText(debug_image, f"E: {ep}", (ep[0] + 10, ep[1]),
#                                 font, font_scale, font_color, line_type)
#                 except Exception as e:
#                     print(f"Loi khi ve tuong {wall.id}: {e}")
#
#             # 2. Vẽ các Cửa (Openings)
#             for opening in self.processed_openings:
#                 try:
#                     lp = (int(opening.location_point[0]), int(opening.location_point[1]))
#
#                     # Vẽ dấu X (Marker) - Màu Vàng
#                     cv2.drawMarker(debug_image, lp, (0, 255, 255),
#                                    markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)  # Yellow
#
#                     host_id_str = opening.host_wall_id if opening.host_wall_id else "None"
#                     text = f"{opening.id} (Host: {host_id_str})"
#                     cv2.putText(debug_image, text, (lp[0] + 15, lp[1] - 15),
#                                 font, font_scale, (0, 255, 255), line_type)  # Yellow
#                 except Exception as e:
#                     print(f"Loi khi ve cua {opening.id}: {e}")
#
#             # 3. Hiển thị ảnh
#             if show_window:
#                 print("Dang hien thi cua so ket qua. Nhan phim bat ky de dong...")
#                 # Thay đổi kích thước cửa sổ nếu ảnh quá lớn
#                 h, w = debug_image.shape[:2]
#                 max_h = 800
#                 if h > max_h:
#                     scale = max_h / h
#                     debug_image_resized = cv2.resize(debug_image, None, fx=scale, fy=scale)
#                     cv2.imshow("Processed Results Visualization", debug_image_resized)
#                 else:
#                     cv2.imshow("Processed Results Visualization", debug_image)
#
#                 cv2.waitKey(0)
#                 cv2.destroyAllWindows()
#
#             print("Ve ket qua debug hoan tat.")
#             return debug_image
#
#     def visualize_topologic_faces(self,
#                                   image: np.ndarray,
#                                   show_window: bool = True,
#                                   alpha: float = 0.4):
#         """
#         Vẽ các đối tượng tp.Face (đã lưu trong self.topo_features)
#         lên ảnh_gốc dưới dạng_polygon_trong_suốt.
#
#         Đây là hàm debug để kiểm tra xem các đối tượng topo có được
#         tạo ra chính xác từ polygon hay không.
#
#         :param image: Ảnh (numpy array) đã được load bằng cv2.imread.
#         :param show_window: Nếu True, sẽ hiển thị ảnh bằng cv2.imshow.
#         :param alpha: Độ trong suốt của polygon (0.0 = trong suốt, 1.0 = đè hẳn)
#         :return: Trả về ảnh đã được vẽ (debug_image).
#         """
#
#         # Tạo bản sao để không vẽ lên ảnh gốc
#         # Chúng ta cần 2 bản: 1 cho kết quả cuối, 1 cho lớp overlay
#         debug_image = image.copy()
#         overlay = image.copy()
#
#         # Cấu hình màu sắc (B, G, R)
#         COLOR_WALL = (255, 0, 0)  # Màu Xanh lam cho Tường
#         COLOR_OPENING = (0, 255, 255)  # Màu Vàng cho Cửa/Cửa sổ
#
#         print("\nBat dau ve Topologic Faces (Debug)...")
#
#         # 1. Vẽ các Tường (Walls) từ self.topo_features
#         for wall in self.processed_walls:
#             if wall.id not in self.topo_features:
#                 continue  # Bỏ qua nếu tường này không có đối tượng topo
#
#             try:
#                 # Lấy Face từ kho lưu trữ
#                 face = self.topo_features[wall.id]
#
#                 # Lấy đường bao (Wire) bên ngoài của Face
#                 wire = face.Wires()[0]  # Giả định chỉ có 1 đường bao
#
#                 # Lấy tất cả các đỉnh (Vertices) của Wire
#                 vertices = wire.Vertices()
#
#                 # Chuyển đổi danh sách tp.Vertex sang danh sách điểm [x, y]
#                 points = []
#                 for v in vertices:
#                     coords = v.Coordinates()  # Trả về (x, y, z)
#                     points.append([coords[0], coords[1]])
#
#                 # Chuyển sang định dạng numpy int32 cho OpenCV
#                 np_points = np.array(points, dtype=np.int32)
#
#                 # Vẽ đa giác CÓ MÀU LẤP ĐẦY (filled polygon) lên lớp OVERLAY
#                 cv2.fillPoly(overlay, [np_points], COLOR_WALL)
#
#             except Exception as e:
#                 print(f"Loi khi ve topo face {wall.id}: {e}")
#
#         # 2. Vẽ các Cửa (Openings) từ self.topo_features
#         for opening in self.processed_openings:
#             if opening.id not in self.topo_features:
#                 continue
#
#             try:
#                 face = self.topo_features[opening.id]
#                 wire = face.Wires()[0]
#                 vertices = wire.Vertices()
#
#                 points = []
#                 for v in vertices:
#                     coords = v.Coordinates()
#                     points.append([coords[0], coords[1]])
#
#                 np_points = np.array(points, dtype=np.int32)
#
#                 # Vẽ đa giác CÓ MÀU LẤP ĐẦY lên lớp OVERLAY
#                 cv2.fillPoly(overlay, [np_points], COLOR_OPENING)
#
#             except Exception as e:
#                 print(f"Loi khi ve topo face {opening.id}: {e}")
#
#         # 3. Trộn (blend) lớp overlay trong suốt với ảnh gốc
#         cv2.addWeighted(overlay, alpha, debug_image, 1 - alpha, 0, debug_image)
#
#         # 4. Hiển thị ảnh
#         if show_window:
#             print("Dang hien thi cua so ket qua Topologic Faces. Nhan phim bat ky de dong...")
#             # Thay đổi kích thước cửa sổ nếu ảnh quá lớn
#             h, w = debug_image.shape[:2]
#             max_h = 800
#             if h > max_h:
#                 scale = max_h / h
#                 debug_image_resized = cv2.resize(debug_image, None, fx=scale, fy=scale)
#                 cv2.imshow("Topologic Faces Visualization", debug_image_resized)
#             else:
#                 cv2.imshow("Topologic Faces Visualization", debug_image)
#
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()
#
#         print("Ve Topologic Faces hoan tat.")
#         return debug_image
#
#     def _process_wall(self, wall_obj: RawObject, obj_id: str) -> WallFeature | None:
#         """
#         Đây là bước quan trọng nhất và phức tạp nhất: Chuyển đổi các đa giác
#         thành các đối tượng Topologic (Face) và trích xuất đường tâm (centerline)
#         cùng độ dày (thickness) một cách chính xác.
#
#         Thuật toán (Đã nâng cấp):
#             1. Lấy `segmentation_polygon` từ `wall_obj`.
#             2. Tạo các `tp.Vertex` (đỉnh) từ các điểm của đa giác.
#             3. Tạo `tp.Wire` (đường bao) từ các đỉnh.
#             4. Tạo `tp.Face` (bề mặt 2D) từ đường bao.
#             5. Lưu trữ `tp.Face` vào `self.topo_features` để dùng sau.
#             6. Sử dụng OpenCV (`cv2.minAreaRect`) để tìm Oriented Bounding Box (OBB)
#                của đa giác. Cách này chính xác hơn nhiều so với AABB (bbox).
#             7. Từ OBB, trích xuất centerline (trục dài nhất) và thickness (trục ngắn nhất).
#             8. Trả về `WallFeature` đã được cải thiện.
#         """
#         print(f"processing {obj_id} (topologic + OBB logic)....")
#         try:
#             # 1. Lấy đa giác (polygon)
#             polygon_points = wall_obj.polygon
#             if not polygon_points or len(polygon_points) < 6:  # Cần ít nhất 3 điểm (x,y)
#                 print(f"Warning: {obj_id} có polygon không hợp lệ. Bỏ qua.")
#                 return None
#
#             # --- Bắt đầu phần Topologic ---
#             # 2. Tạo các đỉnh (Vertices) 3D (giả sử z=0)
#             vertices = []
#             for i in range(0, len(polygon_points), 2):
#                 v = Vertex.ByCoordinates(polygon_points[i], polygon_points[i + 1], 0)
#                 vertices.append(v)
#
#             # 3. Tạo đường bao (Wire)
#             # Tự động loại bỏ đỉnh cuối nếu trùng đỉnh đầu
#             if vertices[0].IsAlmostEqual(vertices[-1]):
#                 vertices.pop()
#
#             wire = Wire.ByVertices(vertices, close=True)
#             if not wire or not wire.IsClosed():
#                 print(f"Warning: {obj_id} không thể tạo Wire khép kín. Bỏ qua.")
#                 return None
#
#             # 4. Tạo bề mặt (Face)
#             face = tp.Face.ByWire(wire)
#             if not face:
#                 print(f"Warning: {obj_id} không thể tạo Face từ Wire. Bỏ qua.")
#                 return None
#
#             # 5. [QUAN TRỌNG] Lưu trữ đối tượng topo
#             self.topo_features[obj_id] = face
#             # --- Kết thúc phần Topologic ---
#
#             # --- Bắt đầu phần Cải thiện Centerline (dùng OpenCV) ---
#             # 6. Sử dụng cv2.minAreaRect để thay thế logic bbox (width > height) cũ
#
#             # Chuyển đa giác sang định dạng numpy cho cv2
#             points_np = np.array(polygon_points).reshape(-1, 2).astype(np.float32)
#
#             # Lấy Oriented Bounding Box
#             # rect là ((center_x, center_y), (width, height), angle)
#             rect = cv2.minAreaRect(points_np)
#
#             # 7. Trích xuất centerline và thickness
#             (w, h) = rect[1]
#             thickness = min(w, h)
#             length = max(w, h)
#
#             # Lấy 4 góc của OBB để tìm 2 điểm đầu/cuối của centerline
#             box_points = cv2.boxPoints(rect)  # 4 điểm [x, y]
#             p0, p1, p2, p3 = box_points
#
#             # Tính trung điểm của các cạnh
#             # self._distance_points là hàm bạn đã có
#             dist_01 = self._distance_points(p0, p1)
#             dist_12 = self._distance_points(p1, p2)
#
#             if dist_01 < dist_12:
#                 # Cạnh 0-1 là cạnh ngắn (thickness)
#                 # Centerline sẽ nối trung điểm 2 cạnh dài
#                 start_point = ((p0[0] + p3[0]) / 2, (p0[1] + p3[1]) / 2)
#                 end_point = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
#             else:
#                 # Cạnh 1-2 là cạnh ngắn (thickness)
#                 start_point = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
#                 end_point = ((p2[0] + p3[0]) / 2, (p2[1] + p3[1]) / 2)
#
#             # 8. Trả về WallFeature chính xác hơn
#             return WallFeature(
#                 id=obj_id,
#                 start_point=start_point,
#                 end_point=end_point,
#                 thickness=thickness
#             )
#
#         except Exception as e:
#             print(f"Error while processing {obj_id} (topologic): {e}")
#             pass  # Giữ nguyên 'pass' như code gốc của bạn
#
#     def _process_opening(self, opening_obj: RawObject, obj_id: str) -> OpeningFeature | None:
#         """
#         Nâng cấp: Xử lý cửa/cửa sổ để tạo ra đối tượng topologic.
#         1. Tạo tp.Face từ polygon và lưu vào self.topo_features.
#         2. Vẫn trả về OpeningFeature (với location_point từ OBB/bbox)
#            để debug và tương thích ngược.
#         """
#
#         try:
#             # --- Bắt đầu phần Topologic (Giống _process_wall) ---
#             polygon_points = opening_obj.polygon
#             if not polygon_points or len(polygon_points) < 6:
#                 print(f"Warning: {obj_id} (opening) có polygon không hợp lệ. Bỏ qua.")
#                 return None
#
#             vertices = []
#             for i in range(0, len(polygon_points), 2):
#                 v = tp.Vertex.ByCoordinates(polygon_points[i], polygon_points[i + 1], 0)
#                 vertices.append(v)
#
#             if vertices[0].IsAlmostEqual(vertices[-1]):
#                 vertices.pop()
#
#             wire = tp.Wire.ByVertices(vertices, close=True)
#             if not wire or not wire.IsClosed():
#                 print(f"Warning: {obj_id} (opening) không thể tạo Wire khép kín. Bỏ qua.")
#                 return None
#
#             face = tp.Face.ByWire(wire)
#             if not face:
#                 print(f"Warning: {obj_id} (opening) không thể tạo Face từ Wire. Bỏ qua.")
#                 return None
#
#             # [QUAN TRỌNG] Lưu trữ đối tượng topo của Cửa/Cửa sổ
#             self.topo_features[obj_id] = face
#             # --- Kết thúc phần Topologic ---
#
#             # Lấy thông tin cơ bản (bbox) như logic cũ
#             bbox = opening_obj.bbox
#             cx = (bbox[0] + bbox[2]) / 2
#             cy = (bbox[1] + bbox[3]) / 2
#             width = bbox[2] - bbox[0]
#
#             return OpeningFeature(
#                 id=obj_id,
#                 category=opening_obj.category,
#                 location_point=(cx, cy),
#                 width=width,
#                 host_wall_id=None  # Sẽ được cập nhật ở _establish_relationships
#             )
#
#         except Exception as e:
#             print(f"Error while processing {obj_id} : {e}")
#             return None
#
#     def _distance_points(self, p1: tuple[float, float], p2: tuple[float, float]) -> float:
#         """
#             Tinh khoang cach Eculidean giua 2 diem
#             D(x,y) = sqrt(sum((xi-y1)^2)/n)
#         """
#
#         return math.sqrt(
#             (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2
#         )
#
#     def _distance_point_to_segment(self,
#                                    p: tuple[float, float],
#                                    s1: tuple[float, float],
#                                    s2: tuple[float, float]) -> float:
#
#         """
#             Tinh khaong cach ngan nhat tu dien p Den doan thang s1-s2
#
#         """
#
#         l2 = (s2[0] - s1[0]) ** 2 + (s2[1] - s1[1]) ** 2
#         if l2 == 0:
#             return self._distance_points(p, s1)
#
#         t = ((p[0] - s1[0]) * (s2[0] - s1[0]) + (p[1] - s1[1]) * (s2[1] - s1[1])) / l2
#         t = max(0, min(1, t))
#
#         projection = (s1[0] + t * (s2[0] - s1[0]), s1[1] + t * (s2[1] - s1[1]))
#         return self._distance_points(p, projection)
#
#
#     def _establish_relationships(self):
#         """
#         NÂNG CẤP HOÀN TOÀN: Sử dụng Topologic để thiết lập mối quan hệ.
#
#         Logic mới:
#         1. Duyệt qua từng 'Opening' đã xử lý.
#         2. Lấy đối tượng 'tp.Face' của opening đó từ self.topo_features.
#         3. Duyệt qua từng 'Wall' đã xử lý.
#         4. Lấy đối tượng 'tp.Face' của wall đó từ self.topo_features.
#         5. Dùng `topo_opening.IsInside(topo_wall)` để kiểm tra
#            xem opening có nằm hoàn toàn bên trong wall hay không.
#         6. Nếu có, gán `opening.host_wall_id = wall.id` và dừng tìm kiếm
#            cho opening đó.
#
#         Hàm này thay thế hoàn toàn logic cũ dùng _distance... và _are_collinear.
#         """
#         print("Establishing relationships (Topologic logic)...")
#
#         # Lấy danh sách ID của các tường (ví dụ: 'wall_0', 'wall_1'...)
#         wall_ids = [wall.id for wall in self.processed_walls]
#
#         # Duyệt qua từng đối tượng 'Opening'
#         for opening in self.processed_openings:
#
#             # 1. Lấy đối tượng topo của Cửa (đã được tạo ở _process_opening)
#             topo_opening = self.topo_features.get(opening.id)
#             if not topo_opening:
#                 continue  # Bỏ qua nếu opening này không có topo object
#
#             # 2. Duyệt qua từng ID tường để tìm "host"
#             for wall_id in wall_ids:
#
#                 # 3. Lấy đối tượng topo của Tường (đã tạo ở _process_wall)
#                 topo_wall = self.topo_features.get(wall_id)
#                 if not topo_wall:
#                     continue  # Bỏ qua nếu tường này không có topo object
#
#                 try:
#                     # 4. [PHÉP TOÁN VÀNG] Kiểm tra topo:
#                     # Cửa có nằm bên trong Tường không?
#                     if topo_opening.IsInside(topo_wall):
#                         # 5. Tìm thấy! Gán host và dừng tìm kiếm cho cửa này
#                         print(f"Relationship found: {opening.id} is inside {wall_id}")
#                         opening.host_wall_id = wall_id
#                         break  # Chuyển sang xử lý opening tiếp theo
#
#                 except Exception as e:
#                     print(f"Error checking relationship between {opening.id} and {wall_id}: {e}")
#
#         # Tường và Cửa đã được cập nhật,
#         # self.processed_walls và self.processed_openings đã có thông tin host.
#         # Chúng ta không cần gán lại 'self.processed_walls' như logic cũ nữa.
#         print(f"Relationships established. Final wall count: {len(self.processed_walls)}")
#
#     def export_for_revit(self, output_path: str, scale_hint: str):
#         pass