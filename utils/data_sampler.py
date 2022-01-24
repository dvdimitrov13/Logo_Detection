"""
Run in main directory - yolo
Data sampler - specify logos and what number of samples per logo
the script will return max(logo_images, number of random logo_images)
"""
import argparse
import os
import random
import re
import shutil

import pandas as pd

random.seed(1337)


def parse_opt():
    parser = argparse.ArgumentParser()
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
            "Intimissimi",
            "Mercedes-Benz",
            "NFL",
            "Nike",
            "Pepsi",
            "Puma",
            "Ralph Lauren Corporation",
            "Starbucks",
            "The North Face",
            "Toyota",
            "Under Armour",
        ],
        help="""takes a list of logos to sample from, takes all logos by default, choose from:
                                                                                ['Adidas', 'Apple Inc.', 'Chanel', 'Coca-Cola', 'Emirates', 'Hard Rock Cafe', 'Intimissimi', 'Mercedes-Benz', 'NFL', 'Nike', 'Pepsi', 'Puma', 'Ralph Lauren Corporation', 'Starbucks', 'The North Face', 'Toyota', 'Under Armour']
                                                                                """,
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=2000,
        help="Number of samples per logo, default is 2000",
    )

    opt = parser.parse_args()
    return opt


def sampler(opt):
    datasets_path = os.path.join(os.getcwd(), "raw_data")
    raw_data_path = os.path.join(datasets_path, "DLCV_logo_project")

    data_sample_path = os.path.join(
        datasets_path, "data_sample_{0}".format(opt.samples)
    )
    try:
        os.mkdir(data_sample_path)
    except:
        print(
            "A data sample with {0} number of images per logo already exists!".format(
                opt.samples
            )
        )

    for logo in opt.logos:
        data_by_logo = os.path.join(raw_data_path, "train_by_logo", logo)

        images = os.listdir(data_by_logo)
        if len(images) > opt.samples:
            images = random.sample(images, opt.samples)

        for img in images:
            shutil.copy(os.path.join(data_by_logo, img), data_sample_path)


if __name__ == "__main__":
    opt = parse_opt()
    sampler(opt)
