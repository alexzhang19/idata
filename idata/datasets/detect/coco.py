#!/usr/bin/env python3
# coding: utf-8

from addict import Dict
from idata.fileio import *
from idata.datasets.build import DATASETS
from idata.datasets.base import BaseDataset

try:
    from pycocotools.coco import COCO
    from terminaltables import AsciiTable
    from pycocotools.cocoeval import COCOeval
except Exception as err:
    print("pycocotools, terminaltables not install.")

__all__ = ["COCO2017"]


@DATASETS.register_module(force=True)
class COCO2017(BaseDataset):
    COCO_CLASSES = ("person", "bicycle", "car", "motorcycle", "airplane", "bus",
                    "train", "truck", "boat", "traffic light", "fire hydrant",
                    "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                    "skis", "snowboard", "sports ball", "kite", "baseball bat",
                    "baseball glove", "skateboard", "surfboard", "tennis racket",
                    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
                    "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
                    "hot dog", "pizza", "donut", "cake", "chair", "couch",
                    "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
                    "mouse", "remote", "keyboard", "cell phone", "microwave",
                    "oven", "toaster", "sink", "refrigerator", "book", "clock",
                    "vase", "scissors", "teddy bear", "hair drier", "toothbrush")

    NAME = "COCO2017"
    TRAIN_SET = "annotations/instances_train2017.json"
    TEST_SET = "annotations/instances_val2017.json"

    TRAIN_IMG_DIR = "train2017"
    VALID_IMG_DIR = "val2017"
    Test_IMG_DIR = "test2017"

    def __init__(self, data_dir: str, test_mode: bool = False, transform=None, target_transform=None,
                 need_path=False):
        self.data_dir = data_dir
        self.test_mode = test_mode
        self.transform = transform
        self.target_transform = target_transform
        self.need_path = need_path

        # step1: _load_meta()，加载self.classes
        # step2: 调用_data_prepare()，加载self._data_dicts
        super(COCO2017, self).__init__()

    def _load_meta(self):
        return list(self.COCO_CLASSES)

    def _data_prepare(self):
        if self.test_mode:
            return self._load_dicts(self.TEST_SET)
        else:
            return self._load_dicts(self.TRAIN_SET)

    def __getitem__(self, index):
        item = self._data_dicts[index]
        img = self.img_read(item.img_path)
        target = self.anno_parse(item.id)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.need_path:
            return img, target, item.img_path
        else:
            return img, target

    def _load_dicts(self, data_set: str):
        """
        输出格式： [{img_path: xxx, label_path: xx}, {}, ...]
        """

        anno_file = path.join(self.data_dir, data_set)
        # print("anno_file:", anno_file)

        self.coco = COCO(anno_file)
        self.cat_ids = self.coco.getCatIds(catNms=self.classes)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}  # label索引从0开始
        self.img_ids = self.coco.getImgIds()

        dicts = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            item = self.VALID_IMG_DIR if self.test_mode else self.TRAIN_IMG_DIR
            r = Dict({
                "img_path": path.join(self.data_dir, item, info["file_name"]),
                "id": info["id"],
                "label_path": info["id"],
                "height": info["height"],
                "width": info["width"]
            })
            dicts.append(r)
        return dicts

    def anno_parse(self, img_id):
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)

        gt_bboxes = []
        gt_labels = []
        gt_masks_ann = []
        for i, ann in enumerate(ann_info):
            if ann.get("ignore", False):
                continue
            x1, y1, w, h = ann["bbox"]  # coco保存格式是 x1 y1 w h (x1 y1是左上角起点坐标)
            if ann["area"] <= 0 or w < 1 or h < 1:
                continue
            if ann["category_id"] not in self.cat_ids:
                continue

            bbox = [x1, y1, x1 + w, y1 + h]
            if not ann.get("iscrowd", False):
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann["category_id"]])
                gt_masks_ann.append(ann.get("segmentation", None))

        # ann = dict(
        #     bboxes=np.array(gt_bboxes, dtype=np.int),
        #     labels=gt_labels,
        #     masks=gt_masks_ann
        # )
        # print("ann:", ann)

        rets = []
        for box, cls in zip(gt_bboxes, gt_labels):
            [x1, y1, x2, y2] = box
            rets.append([int(cls), int(x1), int(y1), int(x2), int(y2)])
        # print("rets：", rets)
        return rets


if __name__ == "__main__":
    data_dir = r"/home/data/datasets/Cam_Flat/coco2017"
    dataset = COCO2017(data_dir, test_mode=True, need_path=True)
    print("class_idx：", dataset.class_idx)
    print("classes: ", dataset.classes)
    print(dataset[0])
    print(len(dataset))
    # dataset.vis(random=True, cnt=20, ret_dir="/home/innno/coco_vis")
    pass
