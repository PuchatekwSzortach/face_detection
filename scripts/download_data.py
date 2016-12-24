"""
This scripts downloads and preprocesses Celeb+ data used for training and testing
"""

import face.datasets.celeb


def main():

    face.datasets.celeb.DatasetBuilder("../../data/faces/").build_datasets()


if __name__ == "__main__":

    main()
