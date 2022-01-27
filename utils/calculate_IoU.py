import argparse
import os

import numpy as np
import pandas as pd


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_name",
        type=str,
        default="yolo1000",
        help="The name of the yolov5 formatted data forlder",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="yolo1000_best",
        help="The name of the model used for prediction",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence treshold for the predicted bounding boxes",
    )
    parser.add_argument(
        "--set",
        type=str,
        default="valid",
        help="Which data are we evaluating: valid/test?",
    )
    parser.add_argument(
        "--logos",
        nargs="+",
        default=[
            "Adidas",
            "Apple Inc.",
            "Chanel",
            "Coca-Cola",
            "Emirates",
            "Hard Rock Cafe",
            "Mercedes-Benz",
            "NFL",
            "Nike",
            "Pepsi",
            "Puma",
            "Starbucks",
            "The North Face",
            "Toyota",
            "Under Armour",
        ],
        help="""input the logos that were part of your training, choose from:
            'Adidas' 'Apple Inc.' 'Chanel' 'Coca-Cola' 'Emirates' 'Hard Rock Cafe' 'Intimissimi' 'Mercedes-Benz' 'NFL' 'Nike' 'Pepsi' 'Puma' 'Ralph Lauren Corporation' 'Starbucks' 'The North Face' 'Toyota' 'Under Armour'
            """,
    )

    opt = parser.parse_args()
    return opt


def calculate_IoU(opt):
    pred_labels_files = os.listdir(
        os.path.join(
            os.getcwd(), "runs", "detect", "output_{}".format(opt.model_name), "labels"
        )
    )
    true_labels_files = os.listdir(
        os.path.join(os.getcwd(), "formatted_data", opt.data_name, opt.set, "labels")
    )

    # Create dfs to store labels
    true_df = pd.DataFrame(columns=["label", "x_t", "y_t", "w_t", "h_t", "fname"])
    pred_df = pd.DataFrame(columns=["label", "x", "y", "w", "h", "conf", "fname"])

    # Fill in the pred_df
    for file in pred_labels_files:
        p_temp_df = pd.read_csv(
            "runs/detect/output_{}/labels/{}".format(opt.model_name, file),
            names=["label", "x", "y", "w", "h", "conf"],
            delim_whitespace=True,
            index_col=False,
            encoding="utf-8",
        )
        p_temp_df["fname"] = file
        pred_df = pd.concat((pred_df, p_temp_df))

    # Fill in the true_df
    for file in true_labels_files:
        t_temp_df = pd.read_csv(
            "formatted_data/{}/{}/labels/{}".format(opt.data_name, opt.set, file),
            names=["label", "x_t", "y_t", "w_t", "h_t"],
            delim_whitespace=True,
            index_col=False,
            encoding="utf-8",
        )
        t_temp_df["fname"] = file
        true_df = pd.concat((true_df, t_temp_df))

    pred_df = pred_df[pred_df["conf"] > opt.conf]

    # Merge the two dfs (use right join to accout for null predictions)
    merged_df = pd.merge(pred_df, true_df, on="fname", how="right")
    merged_df.reset_index(drop=True, inplace=True)

    def calc_iou(boxA, boxB):
        # Maps [0,1] into [0,640], back to the original dimension of the images and
        # applies a change of coordinates to the dataframe to obtain x1, y1, x2, y2 from x, y, w, h
        xa, ya, wa, ha = boxA
        x1a, y1a, x2a, y2a = (
            xa - wa / 2 * 640,
            ya - ha / 2 * 640,
            xa + wa / 2 * 640,
            ya + ha / 2 * 640,
        )

        xb, yb, wb, hb = boxB
        x1b, y1b, x2b, y2b = (
            xb - wb / 2 * 640,
            yb - hb / 2 * 640,
            xb + wb / 2 * 640,
            yb + hb / 2 * 640,
        )

        # accounting for null prediction cases
        if np.isnan(xa) or np.isnan(xb):
            return 0

        boxA = (x1a, y1a, x2a, y2a)
        boxB = (x1b, y1b, x2b, y2b)

        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle, if they don't intersect return 0
        interArea = max(0, xB - xA) * max(0, yB - yA)

        # compute the area of both the prediction and ground-truth
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou

    # Some pictures have multiple predictions, we calculate all IoUs and take the best
    # this way we ensure comparing the right labels and at the same time if the prediction is bad
    # the IoU will still be bad
    merged_df["IoU"] = merged_df.apply(
        lambda row: calc_iou(
            row[["x", "y", "w", "h"]], row[["x_t", "y_t", "w_t", "h_t"]]
        ),
        axis=1,
    )

    # IoU on wrong predictions should be 0 in order to not bias the results
    merged_df["IoU"][merged_df["label_x"] != merged_df["label_y"]] = 0

    # Add how many bounding boxes are there to be predicted
    n_labels_df = true_df.groupby("fname")["label"].count()
    merged_df = pd.merge(n_labels_df, merged_df, on="fname", how="right")
    merged_df.reset_index(drop=True, inplace=True)

    # Pick the X best IoUs where X is the number of true bounding boxes (X = label in this code)
    new = (
        merged_df.sort_values("IoU", ascending=False)
        .groupby("fname")
        .apply(lambda group: group.head(group.iloc[0]["label"]))
    )
    new.reset_index(drop=True, inplace=True)

    classes = [logo for logo in opt.logos]
    classes_dict = {idx: val for idx, val in enumerate(classes)}

    new = new.replace({"label_y": classes_dict})

    print(new[["label_y", "IoU"]].groupby("label_y").mean("IoU"))
    # new[['label_y','IoU']].groupby('label_y').mean('IoU')


if __name__ == "__main__":
    opt = parse_opt()
    calculate_IoU(opt)
