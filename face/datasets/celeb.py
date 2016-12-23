"""
Code for working with Celeb+ dataset
"""

import os
import shutil
import subprocess

import face.download


def get_celeb_datasets():
    """
    Downloads Celeb+ data, arranges it into large, medium and small datasets
    """

    image_archives_urls = [
        "https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AABQwEE5YX5jTFGXjo0f9glIa/Img/img_celeba.7z/img_celeba.7z.001?dl=1",
        "https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADxKopMA7g_Ka2o7X7B8jiHa/Img/img_celeba.7z/img_celeba.7z.002?dl=1",
        "https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AABSqeGALxGo1sXZ-ZizRFa5a/Img/img_celeba.7z/img_celeba.7z.003?dl=1",
        "https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADBal8W3N9AYwYuqwTtA_fQa/Img/img_celeba.7z/img_celeba.7z.004?dl=1",
        "https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AACJaDb7rWNFcCKqcFjFjUlHa/Img/img_celeba.7z/img_celeba.7z.005?dl=1",
        "https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AACcD0ZMO36zVaIfLGLKtrq4a/Img/img_celeba.7z/img_celeba.7z.006?dl=1",
        "https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AAAhuX-S5ULmy8GII6jlZFb9a/Img/img_celeba.7z/img_celeba.7z.007?dl=1",
        "https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AAAUtign0NJIV8fRK7xt6TIEa/Img/img_celeba.7z/img_celeba.7z.008?dl=1",
        "https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AACJsmneLOU5xMB2qmnJA0AGa/Img/img_celeba.7z/img_celeba.7z.009?dl=1",
        "https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AAAfZVSjBlkPr5e5GYMek50_a/Img/img_celeba.7z/img_celeba.7z.010?dl=1",
        "https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AAAfZVSjBlkPr5e5GYMek50_a/Img/img_celeba.7z/img_celeba.7z.010?dl=1",
        "https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AAA6-edxuJyMBoGZqTdl28bpa/Img/img_celeba.7z/img_celeba.7z.011?dl=1",
        "https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AABMLOgnvv8DKxt4UvULSAoha/Img/img_celeba.7z/img_celeba.7z.012?dl=1",
        "https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AABOeeqqAzZEY6jDwTdOUTqRa/Img/img_celeba.7z/img_celeba.7z.013?dl=1",
        "https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADuEM2h2qG_L0UbUTViRH5Da/Img/img_celeba.7z/img_celeba.7z.014?dl=1"
    ]

    data_directory = "../../data/faces/"

    filenames = [os.path.basename(url).split("?")[0] for url in image_archives_urls]
    paths = [os.path.join(data_directory, filename) for filename in filenames]

    # shutil.rmtree(data_directory, ignore_errors=True)
    # os.makedirs(data_directory, exist_ok=True)
    #
    # for url in image_archives_urls:
    #
    #     filename = os.path.basename(url).split("?")[0]
    #     face.download.Downloader(url, os.path.join(data_directory, filename)).download()

    subprocess.call(["7z", "x", paths[0], "-o" + data_directory])



