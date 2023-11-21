# 필수 라이브러리 import
import os
import json
import random
import cv2
import pycocotools.mask as mask_util
import numpy as np
import shutil
import torch
import sys
import json
from rembg import remove
from PIL import Image
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances

# 데이터 로딩 및 Detectron2용 등록
register_coco_instances("my_dataset", {}, "/Users/minjun9282/PycharmProjects/myflask/local_datas/combined.json", "/Users/minjun9282/PycharmProjects/myflask/local_datas/dataset")

# 훈련된 모델 불러오기
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4  # 클래스 개수
cfg.MODEL.ROI_BOX_HEAD.POOLING_MODE = "ROIAlign"
cfg.MODEL.DEVICE = 'cpu'
model_path = "/Users/minjun9282/PycharmProjects/myflask/local_datas/model_final.pth" #저장된 모델의 경로
cfg.MODEL.WEIGHTS = model_path
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1
cfg.DATASETS.TEST = ("my_dataset",)
predictor = DefaultPredictor(cfg)

#사용자의 input 이미지를 받아 image segmentaion 진행 및 segmentaion 결과 저장.
def image_segmentation(input_dataset_directory, output_directory):
  for root, _, files in os.walk(input_dataset_directory):  # input_dataset의 모든 이미지 파일에 대한 image segmentation 진행 후 결과 저장
    for file in files:
        if file.endswith(".jpg"):  # 이미지 파일인 경우
            image_path = os.path.join(root, file)
            im = cv2.imread(image_path)  # 이미지 로드
            outputs = predictor(im)
            mask = outputs["instances"].pred_masks  # 마스크 추출

            image_filename = os.path.splitext(file)[0]  # 파일 이름에서 확장자 제거
            image_directory = os.path.join(output_directory, image_filename)
            os.makedirs(image_directory, exist_ok=True)

            # 원본 이미지 저장
            original_filename = os.path.join(image_directory, f"{image_filename}.jpg")
            cv2.imwrite(original_filename, im)
            # 직접 categories 설정
            categories = ['ground', 'background', 'eyes', 'dogs']
            category_masks = {category: [] for category in categories}

            for category_label in range(len(outputs["instances"])):
                category_name = categories[outputs["instances"].pred_classes[category_label]]
                category_mask = mask[category_label].cpu().numpy()  # 현재 카테고리에 해당하는 마스크

                if category_name in category_masks:
                    if not category_masks[category_name]:
                        category_masks[category_name].append(category_mask)
                    else:
                        combined_mask = category_masks[category_name][0] | category_mask
                        category_masks[category_name][0] = combined_mask

            # 각 카테고리별로 이미지 저장
            for category_name, category_mask_list in category_masks.items():
                category_segmented_region = im.copy()
                for category_mask in category_mask_list:
                    category_segmented_region[~category_mask] = 0  # 마스크 밖의 영역을 제거

                category_filename = os.path.join(image_directory, f"{category_name}.jpg")
                cv2.imwrite(category_filename, category_segmented_region)

            #dogs.jpg는 rembg를 활용하여 새롭게 생성
            input_image = Image.open(image_path)
            output = remove(input_image)
            output.save(os.path.join(image_directory, "dogs.png"))

# dogs.png 파일의 contour line을 투명화 하는 함수 정의
def make_contour_transparent(root_folder):
    for root, dirs, _ in os.walk(os.path.join(root_folder, "segmented_images")):
        for dir_name in dirs:
            images_folder = os.path.join(root, dir_name)
            if os.path.exists(images_folder):
                input_path = os.path.join(images_folder, "dogs.png")
                # 이미지 로드
                dogs_image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

                # 그레이스케일로 변환
                gray = cv2.cvtColor(dogs_image[:, :, :3], cv2.COLOR_BGR2GRAY)

                # 외곽선 찾기
                _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

                # 컨투어 라인 그리기 (외곽선만 그림)
                contour_image = np.zeros_like(dogs_image)
                cv2.drawContours(contour_image, contours, -1, (255, 255, 255, 255), 8)

                # 이미지를 저장
                contour_path = os.path.join(images_folder, "dogs_contour.jpg")
                cv2.imwrite(contour_path, contour_image)

                # 이미지 로드
                dogs_image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
                contour_image = cv2.imread(contour_path, cv2.IMREAD_UNCHANGED)

                # contour line 투명화 적용
                for i in range(contour_image.shape[0]):
                    for j in range(contour_image.shape[1]):
                        # contour_image에서 해당 위치의 픽셀 값 확인
                        contour_pixel = contour_image[i, j]

                        # contour_pixel이 (255, 255, 255, 255)이면 해당 위치의 투명도를 0으로 설정
                        if all(value == 255 for value in contour_pixel):
                            dogs_image[i, j, 3] = 0

                # 새로운 PNG 파일로 저장
                output_path = os.path.join(images_folder, "dogs.png")
                cv2.imwrite(output_path, dogs_image)

# 원근 변환 적용하는 함수
def perspective_transform(line_background, new_background_path, pts, line_slope, line_intercept, result_path):

    # "new_background.jpg" 이미지를 읽기
    new_background = cv2.imread(new_background_path)

    # 이미지의 높이, 너비 및 채널 수 얻기
    nb_height, nb_width, _ = new_background.shape

    # line_background의 크기를 얻기
    background_width, background_height = line_background.size

    # line_background과 동일한 크기와 비율을 갖게 new_background를 리사이즈
    new_background_resized = cv2.resize(new_background, (background_width, background_height))

    # 원근 변환을 위한 변환 행렬 계산
    # pts1 = np.float32([[0, height-line_intercept], [width, height -(line_slope*width +line_intercept)], [width, height], [0, height]]) #변환 전 좌표(좌상단, 우상단, 우하단, 좌하단)
    pts1 = np.float32([[0, background_height - line_intercept], [background_width, background_height - line_slope * background_width - line_intercept], [background_width, background_height], [0, background_height]])
    pts2 = np.float32(pts) # 변환 후 좌표

    perspective_matrix = cv2.getPerspectiveTransform(pts1, pts2)

    # 원근 변환 적용
    perspective_result = cv2.warpPerspective(new_background_resized, perspective_matrix, (background_width, background_height))

    # 변환된 이미지 저장
    cv2.imwrite(result_path, perspective_result)

# close up shot인지를 판단하는 함수 정의
def determine_close_up(area_json_path):
    if os.path.exists(area_json_path):
        # areas.json파일을 로드함
        with open(area_json_path, 'r') as json_file:
            areas_info = json.load(json_file)

        # close_up_shot 변수의 default값을 False로 설정
        close_up_shot = False

        # close up shot인지 판단
        if areas_info.get("Ground Percentage", 0) > 70 or areas_info.get("Background Percentage", 0) > 70:
            close_up_shot = True
        elif areas_info.get("Dogs Percentage", 0) > 35:
            close_up_shot = True

        return close_up_shot

# 이미지의 구도 파악하기 1 - background 및 ground의 어핀 변환에 활용
def connect_lines(root_folder):# background.jpg와 ground.jpg의 contour lines의 교차점을 노란색 직선으로 이어 background와 ground의 구도 파악
    for root, dirs, _ in os.walk(os.path.join(root_folder, "segmented_images")):
        for dir_name in dirs:
            images_folder = os.path.join(root, dir_name)
            if os.path.exists(images_folder):
                ground_img_path = os.path.join(images_folder, "ground.jpg")
                background_img_path = os.path.join(images_folder, "background.jpg")

                if os.path.exists(ground_img_path) and os.path.exists(background_img_path):
                    # 이미지 불러오기
                    ground_image = cv2.imread(ground_img_path, cv2.IMREAD_GRAYSCALE)
                    background_image = cv2.imread(background_img_path, cv2.IMREAD_GRAYSCALE)

                    # 이미지에서 엣지 검출 (Canny 엣지 검출)
                    background_edges = cv2.Canny(background_image, 400, 400)
                    ground_edges = cv2.Canny(ground_image, 400, 400)

                    # 빨간색과 파란색 직선의 교차점 찾기
                    yellow_color = (0, 255, 255)  # 노란색
                    result_image = cv2.cvtColor(ground_image, cv2.COLOR_GRAY2BGR)

                    # 이미지 높이 및 너비 가져오기
                    height, width = ground_image.shape

                    # 교차점 좌표 초기화
                    x1, y1, x2, y2, x3, y3, x4, y4 = 0, 0, 0, 0, 0, 0, 0, 0

                    # 첫 번째 교차점 찾기 (왼쪽에서 오른쪽으로)
                    for x in range(width):
                        for y in range(height):
                            if x <= width // 2 and background_edges[y, x] != 0 and ground_edges[y, x] != 0:
                                if x1 == 0 and y1 == 0:
                                    x1, y1 = x, y
                                    if x1 > width // 2:
                                        break

                    # 두 번째 교차점 찾기 (중간에서 시작하여 왼쪽으로)
                    x = width // 2
                    while x > 0 and x2 == 0 and y2 == 0:
                        for y in range(height):
                            if background_edges[y, x] != 0 and ground_edges[y, x] != 0:
                                x2, y2 = x, y
                                break
                        x -= 1

                    # 세 번째 교차점 찾기 (오른쪽에서 시작하여 왼쪽으로)
                    x = width - 1
                    while x >= 0 and x3 == 0 and y3 == 0:
                        for y in range(height):
                            if background_edges[y, x] != 0 and ground_edges[y, x] != 0:
                                x3, y3 = x, y
                                if x3 < width // 2:
                                    break
                        if x3 != 0:
                            break
                        x -= 1

                    # 네 번째 교차점 찾기 (중심에서 시작하여 오른쪽으로)
                    x = width // 2
                    while x < width and x4 == 0 and y4 == 0:
                        for y in range(height):
                            if background_edges[y, x] != 0 and ground_edges[y, x] != 0:
                                x4, y4 = x, y
                                break
                        x += 1

                    # 첫번째 교차점과 세번째 교차점 찾을때 오류가 중앙선을 넘어 반대편에서 찾는 오류가 있음.
                    # 일단 강제로 고정하고 차후에 수정.
                    if x2 == 0 and y2 == 0:
                        x1, y1 = 0, 0
                    if x4 == 0 and y4 == 0:
                        x3, y3 = 0, 0

                    # 교차점들을 노란색 선으로 연결
                    if x1 <= width // 2 and x3 >= width // 2:
                        # 첫 번째와 세 번째 교차점이 중앙선을 넘지 않을 때, 두 번째 교차점과 네번째 교차점도 연결
                        cv2.line(result_image, (x1, y1), (x2, y2), yellow_color, 2)
                        cv2.line(result_image, (x3, y3), (x4, y4), yellow_color, 2)
                        cv2.line(result_image, (x2, y2), (x4, y4), yellow_color, 2)
                    elif x1 > width // 2 and x3 >= width // 2:
                        # 첫 번째 교차점이 중앙선을 넘고 세 번째 교차점이 중앙선을 넘지 않을때 세번째 교차점과 네번째 교차점만 연결
                        cv2.line(result_image, (x3, y3), (x4, y4), yellow_color, 2)
                    elif x1 <= width // 2 and x3 < width // 2:
                        # 세 번째 교차점이 중앙선을 넘고 첫 번째 교차점이 중앙선을 넘지 않을때 첫번째 교차점과 두번째 교차점만 연결
                        cv2.line(result_image, (x1, y1), (x2, y2), yellow_color, 2)

                    # 선분의 유무 및 기울기, 길이 계산
                    if x1 > 0 and y1 > 0 and (x1 != x2 or y1 != y2):
                        line1_exists = True
                    else:
                        line1_exists = False

                    if x3 > 0 and y3 > 0 and (x3 != x4 or y3 != y4):
                        line3_exists = True
                    else:
                        line3_exists = False

                    if line1_exists and line3_exists:
                        line2_exists = True
                    else:
                        line2_exists = False

                    # 우리가 생각하는 기울기로 바꾸려면 부호를 바꿔줘야 함.(시작 좌표가 (0, 0) 이여서)
                    line1_slope = (y2 - y1) / (x2 - x1) if line1_exists else 0.0
                    line2_slope = (y4 - y2) / (x4 - x2) if line2_exists else 0.0
                    line3_slope = (y3 - y4) / (x3 - x4) if line3_exists else 0.0

                    # 결과를 파일에 저장
                    result_file = os.path.join(images_folder, 'lines.json')
                    line_data = {
                        "line1": {"exists": line1_exists, "slope": line1_slope, "line_intercept": (y1 + y2) // 2},
                        "line2": {"exists": line2_exists, "slope": line2_slope, "line_intercept": y2 - line2_slope * x2},
                        "line3": {"exists": line3_exists, "slope": line3_slope, "line_intercept": (y3 + y4) // 2},
                    }
                    with open(result_file, "w") as json_file:
                        json.dump(line_data, json_file, indent=4)

                    # 구한 line을 활용하여 새로운 마스킹 영역을 갖는 line_ground.jpg와 line_background.jpg 저장
                    if line1_exists == True and line2_exists == False:
                        # (x1, y1), (x2, y2)의 중점의 y좌표를 계산한다
                        mid_y = (y1 + y2) // 2
                        # 해당 이미지에서 중점의 y좌표값보다 작은 y좌표값을 갖는 모든 픽셀을 하얀색으로, 중점의 y좌표보다 크거나 같은 y좌표값을 갖는 모든 픽셀을 검은색으로 칠한 line_background.jpg를 생성한다.
                        line_background = result_image.copy()
                        line_background[:mid_y, :] = 255
                        line_background[mid_y:, :] = 0
                        # 해당 이미지에서 중점의 y좌표값보다 작은 y좌표값을 갖는 모든 픽셀을 검은색으로, 중점의 y좌표보다 크거나 같은 y좌표값을 갖는 모든 픽셀을 하얀색으로 칠한 line_ground.jpg를 생성한다.
                        line_ground = result_image.copy()
                        line_ground[:mid_y, :] = 0
                        line_ground[mid_y:, :] = 255
                        # 결과 이미지 저장
                        cv2.imwrite(os.path.join(images_folder, 'line_background.jpg'), line_background)
                        cv2.imwrite(os.path.join(images_folder, 'line_ground.jpg'), line_ground)
                    elif line3_exists == True and line2_exists == False:
                        # (x3, y3), (x4, y4)의 중점의 y좌표를 계산한다.
                        mid_y = (y3 + y4) // 2
                        # 해당 이미지에서 중점의 y좌표값보다 작은 y좌표값을 갖는 모든 픽셀을 하얀색으로, 중점의 y좌표보다 크거나 같은 y좌표값을 갖는 모든 픽셀을 검은색으로 칠한 line_background.jpg를 생성한다.
                        line_background = result_image.copy()
                        line_background[:mid_y, :] = 255
                        line_background[mid_y:, :] = 0
                        # 해당 이미지에서 중점의 y좌표값보다 작은 y좌표값을 갖는 모든 픽셀을 검은색으로, 중점의 y좌표보다 크거나 같은 y좌표값을 갖는 모든 픽셀을 하얀색으로 칠한 line_ground.jpg를 생성한다.
                        line_ground = result_image.copy()
                        line_ground[:mid_y, :] = 0
                        line_ground[mid_y:, :] = 255
                        # 결과 이미지 저장
                        cv2.imwrite(os.path.join(images_folder, 'line_background.jpg'), line_background)
                        cv2.imwrite(os.path.join(images_folder, 'line_ground.jpg'), line_ground)
                    elif line2_exists == True and line2_slope != 0:
                        # (x2, y2), (x4, y4)를 지나는 직선의 y절편을 line_intercept로 두고 (0, y_intercept), (width, height-y_intercept)를 지나는 직선으로 영역을 분할 (기울기를 감소시키기 위함)
                        # 이때는 result_image에 있는 모든 x좌표값의 범위에 대해 x좌표값이 동일할때 위에서 그린 직선의 y좌표값 보다 작은 y좌표값을 갖는 모든 픽셀을 하얀색으로, 크거나 같은 y좌표값을 갖는 모든 픽셀을 검은색으로 칠한 line_background.jpg를 생성한다.
                        # 또한 result_image에 있는 모든 x좌표값의 범위에 대해 x좌표값이 동일할때 위에서 그린 직선의 y좌표값 보다 작은 y좌표값을 갖는 모든 픽셀을 검은색으로, 크거나 같은 y좌표값을 갖는 모든 픽셀을 하얀색으로 칠한 line_ground.jpg를 생성한다.
                        line_slope = line2_slope
                        line_intercept = y2 - line_slope * x2

                        # (0, line_intercept)을 지나고 line_slope를 기울기로 갖는 adjusted_slope_line을 생성
                        line_background = result_image.copy()
                        line_ground = result_image.copy()

                        for x in range(width):
                            adjusted_slope_line = int(line_slope * x + line_intercept)

                            # line_background에서 위를 흰색, 아래를 검은색으로 칠함
                            line_background[:adjusted_slope_line, x] = 255
                            line_background[adjusted_slope_line:, x] = 0

                            # line_ground에서 위를 검은색, 아래를 흰색으로 칠함
                            line_ground[:adjusted_slope_line, x] = 0
                            line_ground[adjusted_slope_line:, x] = 255

                        # bilateralFilter를 사용하여 line_ground와 line_background의 경계선을 더욱 매끈하게 처리함
                        smoothed_line_background = cv2.bilateralFilter(line_background, d=-1, sigmaColor=10, sigmaSpace=5)
                        smoothed_line_ground = cv2.bilateralFilter(line_ground, d=-1, sigmaColor=10, sigmaSpace=5)

                        # 결과 이미지 저장
                        cv2.imwrite(os.path.join(images_folder, 'line_background.jpg'), smoothed_line_background)
                        cv2.imwrite(os.path.join(images_folder, 'line_ground.jpg'), smoothed_line_ground)

# 이미지의 구도 파악하기2
# 이미지에서 ground, background, dogs가 각각 차지하는 비율 계산(close-up shot에 해당하는지 판단할 때 활용)
def calculate_area_ratio(root_folder):
    for root, dirs, _ in os.walk(os.path.join(root_folder, "segmented_images")):
        for dir_name in dirs:
            images_folder = os.path.join(root, dir_name)
            if os.path.exists(images_folder):
                ground_img_path = os.path.join(images_folder, "ground.jpg")
                background_img_path = os.path.join(images_folder, "background.jpg")
                dogs_img_path = os.path.join(images_folder, "dogs.jpg")

                if os.path.exists(ground_img_path) and os.path.exists(background_img_path) and os.path.exists(dogs_img_path):
                    ground_image = cv2.imread(ground_img_path)
                    background_image = cv2.imread(background_img_path)
                    dogs_image = cv2.imread(dogs_img_path)

                    # 각각의 이미지를 활용하여 마스크 계산
                    ground_mask = cv2.imread(os.path.join(images_folder, "ground.jpg"), 0)
                    background_mask = cv2.imread(os.path.join(images_folder, "background.jpg"), 0)
                    dogs_mask = cv2.imread(os.path.join(images_folder, "dogs.jpg"), 0)

                    # 각 객체의 마스크 영역을 활용하여 컨투어(외곽선) 검출
                    ground_contours, _ = cv2.findContours(ground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    background_contours, _ = cv2.findContours(background_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    dogs_contours, _ = cv2.findContours(dogs_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    # 각 객체의 면적 계산
                    ground_area = sum(cv2.contourArea(contour) for contour in ground_contours)
                    background_area = sum(cv2.contourArea(contour) for contour in background_contours)
                    dogs_area = sum(cv2.contourArea(contour) for contour in dogs_contours)

                    # 전체 이미지 면적 계산
                    total_area = ground_area + background_area + dogs_area

                    # 면적을 백분율로 변환
                    ground_percentage = (ground_area / total_area) * 100
                    background_percentage = (background_area / total_area) * 100
                    dogs_percentage = (dogs_area / total_area) * 100

                    # 결과를 JSON 파일에 저장
                    result_dict = {
                        "Ground Percentage": ground_percentage,
                        "Background Percentage": background_percentage,
                        "Dogs Percentage": dogs_percentage
                    }
                    result_file = os.path.join(images_folder, "areas.json")
                    with open(result_file, "w") as json_file:
                        json.dump(result_dict, json_file, indent=4)

# 이미지 수정 및 저장 함수 정의
def modify_and_save_images(img_source_path, root_folder):
    for root, dirs, _ in os.walk(os.path.join(root_folder, "segmented_images")):
        for dir_name in dirs:
            images_folder = os.path.join(root, dir_name)
            if os.path.exists(images_folder):
                ground_img_path = os.path.join(images_folder, "line_ground.jpg")
                background_img_path = os.path.join(images_folder, "line_background.jpg")
                dogs_img_path = os.path.join(images_folder, "dogs.png")
                areas_json_path = os.path.join(images_folder, "areas.json")
                lines_json_path = os.path.join(images_folder, "lines.json")
                if os.path.exists(ground_img_path) and os.path.exists(background_img_path) and os.path.exists(dogs_img_path):
                    # 이미지 수정 작업을 수행하고 modified_image에 반영
                    ground_image = Image.open(ground_img_path)
                    background_image = Image.open(background_img_path)
                    dogs_image = Image.open(dogs_img_path)
                    if determine_close_up(areas_json_path) == True:
                        new_ground_image = Image.open(os.path.join(img_source_path, "carpet6.jpg"))
                        new_background_image = Image.open(os.path.join(img_source_path, "carpet6.jpg"))
                    else:
                        # lines.json파일을 로드함
                        with open(lines_json_path, 'r') as json_file:
                            lines_info = json.load(json_file)

                        # perspective_transform함수에 입력할 argument
                        new_ground_image = Image.open(os.path.join(img_source_path, "carpet6.jpg"))
                        width, height = background_image.size
                        new_background_path = os.path.join(img_source_path, "blank wall.jpg")
                        background_result_path = os.path.join(images_folder, 'perspective_background.jpg')

                        if lines_info["line2"]["exists"] == False:
                            if lines_info["line1"]["exists"] == True:
                                line_slope = 0
                                line_intercept = lines_info["line1"]["line_intercept"]
                                pts = [(0, 0), (width, 0), (width, line_slope * width + line_intercept), (0, line_intercept)] #변환 후 좌표(좌상단, 우상단, 우하단, 좌하단)
                                perspective_transform(background_image, new_background_path, pts, line_slope, line_intercept, background_result_path)
                                new_background_image = Image.open(os.path.join(images_folder, "perspective_background.jpg"))
                            elif lines_info["line3"]["exists"] == True:
                                line_slope = 0
                                line_intercept = lines_info["line3"]["line_intercept"]
                                pts = [(0, 0), (width, 0), (width, line_slope * width + line_intercept), (0, line_intercept)] #변환 후 좌표(좌상단, 우상단, 우하단, 좌하단)
                                perspective_transform(background_image, new_background_path, pts, line_slope, line_intercept, background_result_path)
                                new_background_image = Image.open(os.path.join(images_folder, "perspective_background.jpg"))
                        else:
                            #new_background_image의 경우 원근변환을 적용한 후 생성된 perspective_background.jpg를 이용
                            line_slope = lines_info["line2"]["slope"]
                            line_intercept = lines_info["line2"]["line_intercept"]
                            pts = [(0, 0), (width, 0), (width, line_slope * width + line_intercept), (0, line_intercept)] #변환 후 좌표(좌상단, 우상단, 우하단, 좌하단)
                            perspective_transform(background_image, new_background_path, pts, line_slope, line_intercept, background_result_path)
                            new_background_image = Image.open(os.path.join(images_folder, "perspective_background.jpg"))

                    # 원본 이미지 로드
                    original_image_path = os.path.join(root, dir_name, dir_name + ".jpg")
                    original_image = cv2.imread(original_image_path)

                    # 이미지를 NumPy 배열로 변환
                    src = np.array(original_image)
                    ground_mask = np.array(ground_image.convert("L"))
                    background_mask = np.array(background_image.convert("L"))
                    ground_dst = np.array(new_ground_image)
                    background_dst = np.array(new_background_image)

                    # 이미지 크기를 확인
                    height, width, channels = src.shape

                    # 이미지 크기에 맞게 배열의 크기를 수정
                    if ground_mask.shape[:2] != (height, width):
                        ground_mask = cv2.resize(ground_mask, (width, height))
                    if background_mask.shape[:2] != (height, width):
                        background_mask = cv2.resize(background_mask, (width, height))
                    if ground_dst.shape[:2] != (height, width):
                        ground_dst = cv2.resize(ground_dst, (width, height))
                    if background_dst.shape[:2] != (height, width):
                        background_dst = cv2.resize(background_dst, (width, height))

                    # 이미지를 NumPy 배열로 변환하고 색상 공간을 BGR에서 RGB로 변환
                    src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)

                    # 마스킹 된 부분을 활용하여 이미지 합성
                    src[np.where(ground_mask > 0)] = ground_dst[np.where(ground_mask > 0)]
                    src[np.where(background_mask > 0)] = background_dst[np.where(background_mask > 0)]

                    # background와 ground로 dogs가 놓일 새로운 백그라운드 이미지 생성
                    new_bckground_path = os.path.join(images_folder, "new_bckground_image.png")
                    modified_image = Image.fromarray(src)
                    modified_image.save(new_bckground_path)

                    # 새로운 백그라운드 이미지(new_bckground_image.png) 열기
                    background = Image.open(new_bckground_path)

                    # 상단에 합칠 dogs.png 열기
                    overlay = Image.open(dogs_img_path)

                    # 이미지 합치기
                    changed_image = Image.alpha_composite(background.convert('RGBA'), overlay.convert('RGBA'))

                    # 결과 이미지 저장
                    changed_image.save(os.path.join(images_folder, "changed_image.png"))

# 미디안 필터 적용 함수 정의
def apply_median_filter(image, contours, filter_size):
    result_image = image.copy()
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        roi = image[y:y+h, x:x+w]

        # 이미지가 비어 있지 않은지 확인
        if not roi.size:
            continue

        # 적절한 크기와 채널 수를 가지고 있는지 확인
        if roi.shape[-1] == 3:  # 이미지가 3개의 채널(RGB)을 가져야 함
            filtered_roi = cv2.medianBlur(roi, filter_size)
            result_image[y:y+h, x:x+w] = filtered_roi

    return result_image

#노이즈 중 소금 후추 노이즈를 제거하는 함수 정의
def remove_noise_and_apply_median_filter(root_folder, filter_size):
    for root, dirs, _ in os.walk(os.path.join(root_folder, "segmented_images")):
        for dir_name in dirs:
            images_folder = os.path.join(root, dir_name)
            if os.path.exists(images_folder):
                changed_img_path = os.path.join(images_folder, "changed_image.png")
                if os.path.exists(changed_img_path):
                    # 이미지 불러오기
                    image = cv2.imread(changed_img_path)

                    # dogs.jpg를 사용하여 dogs 객체의 마스크를 얻습니다.
                    mask = cv2.imread(os.path.join(images_folder, "dogs.png"), 0)

                    # dogs 객체의 외부 라인을 얻습니다.
                    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

                    # 외부 라인을 사용하여 이미지에 미디안 필터 적용
                    result_image = apply_median_filter(image, contours, filter_size)

                    # 처리된 이미지 저장
                    output_image_path = os.path.join(images_folder, "result_image.jpg")
                    cv2.imwrite(output_image_path, result_image)

 #각 폴더 별로 원본 이미지와 result_image를 제외한 모든 파일 삭제
def delete_temp_files(root_folder):
    for root, dirs, files in os.walk(os.path.join(root_folder, "segmented_images")):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            # 원본 파일이나 result_image.jpg가 아닌 경우에 삭제
            if file_name != f"{os.path.basename(root)}.jpg" and file_name != "result_image.jpg":
                os.remove(file_path)

# 경로 설정
input_dataset_directory = "/Users/minjun9282/PycharmProjects/myflask/local_datas/for_demo"  # 사용자의 input 이미지 폴더 경로
output_directory = "/Users/minjun9282/PycharmProjects/myflask/web_datas/output/segmented_images" # 이미지 합성을 완료한 결과물을 저장하는 ouput 폴더 경로
root_folder = "/Users/minjun9282/PycharmProjects/myflask/web_datas/output/" #위의 segmented_images폴더의 상위 폴더
img_source_path = "/Users/minjun9282/PycharmProjects/myflask/local_datas/img_source" #합성에 사용될 이미지를 포함하는 경로

#함수 호출
image_segmentation(input_dataset_directory, output_directory) #image segmentation 진행
make_contour_transparent(root_folder) #dogs의 경우 외곽선을 더 깔끔하게 편집
connect_lines(root_folder) #각 사진의 구도 파악을 위한 직선 그리기
calculate_area_ratio(root_folder) #각 사진의 원근 파악을 위한 이미지 내의 객체 비율 구하기
modify_and_save_images(img_source_path, root_folder) #사진의 구도와 원근을 고려하여 새로운 배경과 바닥 이미지로 합성
remove_noise_and_apply_median_filter(root_folder, filter_size = 3) #사진 속 노이즈 제거
delete_temp_files(root_folder) #각 폴더에서 원본 이미지와 result_image.jpg를 제외한 모든 파일 제거               