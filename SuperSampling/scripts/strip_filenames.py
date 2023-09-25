"""
Script that given a directory with just files of the form 
    .*\d+\.[^\.]+
renames all files to 
    \d+\.[^\.]+
"""

import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Read script description.')

    parser.add_argument('dir_name', type=str,
                        help='Parent folder for data')

    args = parser.parse_args()
    return args 

def get_dir_contents(root: str, *folders: list[str]) -> list[str]:
    return os.listdir(os.path.join(root, *folders))
        

def rename(src: str, dst: str, dir_name: str) -> None:
    if '/' in src or '\\' in src:
        raise ValueError("src must be a filename only")
    if '/' in dst or '\\' in dst:
        raise ValueError("dst must be a filename only")

    os.rename(dir_name + os.sep + src, dir_name + os.sep + dst)

def main():
    args = parse_args()
    dir_name = args.dir_name

    files = get_dir_contents(dir_name)
    new_files = [None] * len(files)
    for k, filename in enumerate(files):

        new_filename, ext = filename.rsplit('.', 1)
        new_filename = new_filename[::-1]

        i = 0
        while new_filename[i].isdigit():
            i+=1

        new_filename = new_filename[:i]
        new_filename = f"{new_filename[::-1]}.{ext}"
        
        new_files[k] = new_filename

    if len(set(new_files)) != len(set(files)):

        raise ValueError("""
            Some files have common numberings under the transformation, 
            this would result in lost data.
                         
            The operation has been aborted.
                        """)
    
    for old_file, new_file in zip(files, new_files):
        rename(old_file, new_file, dir_name)

if __name__ == "__main__":
    main()