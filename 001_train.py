from torchkit.task.base_task import BaseTask

class TrainTask(BaseTask):
    def __init__(self,vein_finger_file):
        super().__init__(cfg_file=vein_finger_file)

def main():
    pass

if __name__ == '__main__':
    main()