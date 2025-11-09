import json
import cv2
import numpy as np
from skimage.morphology import skeletonize
from dataclasses import dataclass


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


    def _process_wall(self, wall_obj: RawObject, obj_id: str) -> WallFeature | None:

        """
            Đây là bước quan trọng nhất  và phứt tạp nhất: Chuyển đổi các đa giác thành các đường tâmcenterline(). Có độ dày(Thickness)


            Thuật toán:
                -

        """


        print(f"processing {obj_id}....")
        try:
            # create binary mask
            mask = np.zeros(
                (self.image_height, self.image_width),
                dtype=np.uint8
            )
            polygon_points = np.array(wall_obj.polygon).reshape((-1,2)).astype(np.int32)
            cv2.fillPoly(mask, [polygon_points], 255)

            # find centerline
            skeleton = skeletonize(mask>0)
            skeleton_points = np.argwhere(skeleton)

            if len(skeleton_points) < 2:
                return None # polygon request > 2

            """
                Cách đơn giản là: Chọn  điểm đầu điểm cuối.
            """
            start_point_yx = skeleton_points.min(axis=0)
            end_point_yx = skeleton_points.max(axis=0)

            start_point_xy = (float(start_point_yx[1]), float(start_point_yx[0]))
            end_point_xy = (float(end_point_yx[1]), float(end_point_yx[0]))

            # thickness
            bbox = wall_obj.bbox
            thickness = min(bbox[2] - bbox[0], bbox[3] - bbox[1])

            return WallFeature(
                id=obj_id,
                start_point=start_point_xy,
                end_point=end_point_xy,
                thickness=thickness
            )


        except Exception as e:
            pass

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


    def _establish_relationships(self):
        pass

    def export_for_revit(self, output_path: str, scale_hint: str):
        pass