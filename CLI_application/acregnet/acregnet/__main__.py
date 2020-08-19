VERSION = '1.1'


def main():
    import sys
    import os

    def show_msg(msg=None):
        if msg is not None:
            print(msg)
        else:
            print('usage: acregnet register <target> <source> ' +
                  '--dest=<destination-directory>')
        sys.exit()

    args = sys.argv[1:]

    if not args:
        show_msg()
    if args[0] in ('-v', '--version'):
        show_msg(VERSION)
    elif args[0] in ('-h', '--help'):
        show_msg()
    elif args[0] in ('register'):
        if len(args[1:]) != 3:
            show_msg()

        if not args[1].endswith('.png') or not os.path.exists(args[1]):
            show_msg('Target image: unknown data file. Expected: PNG image.')
        if not args[2].endswith('.png') or not os.path.exists(args[2]):
            show_msg('Source image: unknown data file. Expected: PNG image.')

        if not args[3].startswith('--dest=/'):
            show_msg('Please provide a destination directory.')
        if not os.path.exists(args[3][7:]):
            show_msg('Destination directory does not exist.')

        fix_im_fname = args[1]
        mov_im_fname = args[2]

        dest_dir = args[3][7:]
        if dest_dir[-1] == '/':
            dest_dir = dest_dir[:-1]

    else:
        show_msg()

    from .register import register

    try:
        register(fix_im_fname, mov_im_fname, dest_dir)
    except Exception as e:
        print(e)


if __name__ == '__main__':
    main()
