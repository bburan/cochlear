import ast
import os.path

from cochlear.util import remove_trials


def parse_filter(string):
    name, value = string.split('=')
    value = ast.literal_eval(value)
    return name, value


if __name__ == '__main__':
    import argparse
    description = 'Trim rows from file'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('file', help='File to truncate')
    parser.add_argument('-o', '--out', nargs='?', help='Output filename')
    parser.add_argument('filter', type=parse_filter, help='Filter', nargs='+')
    args = parser.parse_args()
    if args.out is None:
        base, ext = os.path.splitext(args.file)
        args.out = '{} (trimmed){}'.format(base, ext)
    remove_trials(args.file, args.out, args.filter)
