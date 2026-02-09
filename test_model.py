import cv2
import numpy as np
import torch
import torchvision
from torchvision import transforms as T
from pathlib import Path

DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480
BG_COLOR = (192, 192, 192)
MASK_COLOR = (255, 255, 255)
IMAGES_FOLDER = 'dataset'
OUTPUT_FOLDER = 'output'
SCORE_THRESHOLD = 0.7
MASK_THRESHOLD = 0.5
BLUR_KERNEL = (55, 55)

COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def resize_image(image):
    h, w = image.shape[:2]
    interp = cv2.INTER_AREA if h > DESIRED_HEIGHT else cv2.INTER_LINEAR
    if h < w:
        new_w, new_h = DESIRED_WIDTH, int(h / (w / DESIRED_WIDTH))
    else:
        new_h, new_w = DESIRED_HEIGHT, int(w / (h / DESIRED_HEIGHT))
    return cv2.resize(image, (new_w, new_h), interpolation=interp)

def draw_detections(image, predictions, threshold=SCORE_THRESHOLD):
    img_copy = image.copy()
    boxes = predictions[0]['boxes'].detach().cpu().numpy()
    labels = predictions[0]['labels'].detach().cpu().numpy()
    scores = predictions[0]['scores'].detach().cpu().numpy()

    for i in range(len(scores)):
        if scores[i] > threshold:
            box = boxes[i].astype(int)
            label = COCO_CLASSES[labels[i]]
            score = scores[i]

            cv2.rectangle(img_copy, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            text = f"{label} {score:.2f}"
            cv2.putText(img_copy, text, (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img_copy

def extract_contours(mask_data):
    binary_mask = (mask_data > MASK_THRESHOLD).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = np.zeros((binary_mask.shape[0], binary_mask.shape[1], 3), dtype=np.uint8)
    cv2.drawContours(contour_img, contours, -1, (255, 255, 255), 2)
    return contour_img

def create_composite(original, predictions):
    scores = predictions[0]['scores'].detach().cpu().numpy()
    masks = predictions[0]['masks'].detach().cpu().numpy()
    
    valid_masks = masks[scores > SCORE_THRESHOLD]
    if len(valid_masks) == 0:
        return None
    
    combined_mask = np.max(valid_masks, axis=0).squeeze()
    mask_cond = (combined_mask > MASK_THRESHOLD)

    boxes_img = draw_detections(original, predictions)

    mask_visual = np.full(original.shape, BG_COLOR, dtype=np.uint8)
    mask_visual[mask_cond] = MASK_COLOR

    contours_img = extract_contours(combined_mask)

    blurred_bg = cv2.GaussianBlur(original, BLUR_KERNEL, 0)
    blur_img = np.where(mask_cond[..., None], original, blurred_bg)

    all_imgs = [original, boxes_img, mask_visual, contours_img, blur_img]
    resized = [resize_image(img) for img in all_imgs]
    
    max_h = max(img.shape[0] for img in resized)
    final_stack = [cv2.resize(img, (img.shape[1], max_h)) for img in resized]
    
    return np.hstack(final_stack)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT").to(device).eval()
    transform = T.Compose([T.ToTensor()])

    input_path = Path(IMAGES_FOLDER)
    output_path = Path(OUTPUT_FOLDER)
    output_path.mkdir(exist_ok=True)

    for subfolder in [d for d in input_path.iterdir() if d.is_dir()]:
        sub_output = output_path / subfolder.name
        sub_output.mkdir(exist_ok=True)
        
        for img_file in list(subfolder.glob('*.*')):
            img_cv = cv2.imread(str(img_file))
            if img_cv is None: continue

            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            tensor = transform(img_rgb).to(device).unsqueeze(0)
            
            with torch.no_grad():
                preds = model(tensor)

            composite = create_composite(img_cv, preds)
            
            if composite is not None:
                cv2.imwrite(str(sub_output / f"{img_file.stem}.png"), composite)

if __name__ == '__main__':
    main()