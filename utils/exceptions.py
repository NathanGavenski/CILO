class CheckpointAlreadyExists(Exception):
    def __init__(self, files, name):
        super().__init__(f'{name} already exists in {files}')
