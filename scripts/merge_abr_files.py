import os.path

from cochlear.util import merge_abr_files


if __name__ == '__main__':
    import argparse
    description = 'Merge ABR files'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('files', nargs='+', help='ABR files to merge')
    parser.add_argument('-o', '--out', nargs='?', help='Output filename')
    args = parser.parse_args()
    if args.out is None:
        base, ext = os.path.splitext(args.files[0])
        args.out = '{} (merged){}'.format(base, ext)
    merge_abr_files(args.files, args.out)
