VERSION = '1.1'


def show_msg(msg=None):
    import sys

    if msg is not None:
        print(msg)
    else:
        print('usage: acregnet register <target> <source> ' +
              '--dest=<destination-directory>')
    sys.exit()


def main():
    import os
    import sys

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

        import imghdr

        if not os.path.exists(args[1]) or imghdr.what(args[1]) != 'png':
            show_msg('Target image must be a valid image contained in a ' +
                     'PNG file. For example: /home/user/my_dir/image001.png.')
        if not os.path.exists(args[2]) or imghdr.what(args[2]) != 'png':
            show_msg('Source image must be a valid image contained in a ' +
                     'PNG file. For example: /home/user/my_dir/image001.png.')
        if not args[3].startswith('--dest=/'):
            show_msg('An existing destination directory must be specified. ' +
                     'For example: --dest=/home/user/my_dir')
        if not os.path.exists(args[3][7:]):
            show_msg('Destination directory does not exist.')

        fix_im_fname = args[1]
        mov_im_fname = args[2]
        dest_dir = args[3][7:]
    else:
        show_msg()

    from .register import register

    try:
        register(fix_im_fname, mov_im_fname, dest_dir)
    except Exception as e:
        print(e)


if __name__ == '__main__':
    main()
