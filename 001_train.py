import os
import sys

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..'))


def main():
    task_dir = os.path.dirname(os.path.abspath(__file__))
    task.init_env()
    task.train()

if __name__ == '__main__':
    main()