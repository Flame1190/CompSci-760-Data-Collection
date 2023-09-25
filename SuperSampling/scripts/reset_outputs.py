
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Setup dirs or clear existing contents')

    parser.add_argument('dir', type=str,
                        help='output folder containing ground_truth and outputs')


    args = parser.parse_args()
    return args 

def main():
    args = parse_args()
    ground_truth_path = os.path.join(args.dir, "ground_truth")
    output_path = os.path.join(args.dir, "output")
    
    if not os.path.exists(ground_truth_path):
        os.mkdir(ground_truth_path)
    elif not os.path.isdir(ground_truth_path):
        raise RuntimeError("file exists matching folder to be created")
    
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    elif not os.path.isdir(output_path):
        raise RuntimeError("file exists matching folder to be created")
    
    for f in os.listdir(ground_truth_path):
        os.remove(os.path.join(ground_truth_path, f))
    for f in os.listdir(output_path):
        os.remove(os.path.join(output_path, f))
        

if __name__ == "__main__":
    main()