import os.path

from cochlear.util import truncate_file


if __name__ == '__main__':
    import argparse
    description = 'Truncate file'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('file', help='File to truncate')
    parser.add_argument('-o', '--out', nargs='?', help='Output filename')
    parser.add_argument('size', type=int, help='New size')
    args = parser.parse_args()
    if args.out is None:
        base, ext = os.path.splitext(args.file)
        args.out = '{} (trimmed){}'.format(base, ext)
    truncate_file(args.file, args.out, args.size)
