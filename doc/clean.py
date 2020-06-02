"""
Small script that deletes unused graphic files in a latex project.

Specify directory containing figures and the logfile. Unused files will be
deleted in specified directory.
"""
import os

encoding = 'utf-8'


def clean(fig_dir=None, log_file=None):
    if fig_dir is None:
        fig_dir = os.path.abspath(os.path.join('.', 'figures'))
    if log_file is None:
        log_file = os.path.abspath(os.path.join('.', 'main.log'))

    used = []
    unused = []
    for root, _, files in os.walk(fig_dir):
        for f in files:
            if f in open(log_file, encoding=encoding).read():
                used.append(os.path.join(root, f))
                print(f'{f} in use.')
            else:
                if os.path.isfile(os.path.join(root, f)):
                    print(f'{f} not in use: deleting.')
                    unused.append(os.path.join(root, f))
                    os.remove(os.path.join(root, f))

    used_file = os.path.join('.', 'used_files.txt')
    with open(used_file, 'w') as f:
        for u in used:
            f.write(f'{u}\n')

    unused_file = os.path.join('.', 'unused_files.txt')
    with open(unused_file, 'w') as f:
        for u in unused:
            f.write(f'{u}\n')


if __name__ == '__main__':
    clean()
