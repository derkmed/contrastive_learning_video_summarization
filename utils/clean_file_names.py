import argparse
import sys
import pathlib
import os


def main(args):
    for parent_dir in args.target_dir:
        for path in pathlib.Path(f'{args.root_dir}/{parent_dir}').iterdir():
            x = os.path.basename(path)
            x = x.lower().replace(' ', '_')
            path.rename(os.path.dirname(path) + f'/{x}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Modify summe file names')
    parser.add_argument('--root_dir', type=str, required=True)
    parser.add_argument('--target_dir', nargs='+',
                        type=str, default=['videos', 'GT'])
    args = parser.parse_args()
    main(args)