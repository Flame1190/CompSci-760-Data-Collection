# script to check data validity
import argparse
from data_loader.data_loaders import NSRRDataLoader
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Test data loading and analytical transforms used.')

    parser.add_argument('src', type=str,
                        help='Parent folder for data to test')
    parser.add_argument('--color_dirname', default='color')
    parser.add_argument('--depth_dirname', default='depth')
    parser.add_argument('--motion_dirname', default='motion')

    args = parser.parse_args()
    return args 

def get_dir_contents(root: str, *folders: list[str]) -> list[str]:
    return os.listdir(os.path.join(root, *folders))
        


def main():
    args = parse_args()

    # images = get_dir_contents(args.src, args.motion_dirname)

    # print(args)
    loader = NSRRDataLoader(
        args.src,
        args.color_dirname,
        args.depth_dirname,
        args.motion_dirname,
        1
    )
    for i, x in enumerate(loader):
        print(x)
        break


if __name__ == "__main__":
    main()