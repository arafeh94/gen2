from src.data.data_distributor import PipeDistributor


def clustered(size):
    pipes = [
        PipeDistributor.pick_by_label_id([1, 2], 600, size),
        PipeDistributor.pick_by_label_id([3, 4], 600, size),
        PipeDistributor.pick_by_label_id([5, 6], 600, size),
        PipeDistributor.pick_by_label_id([7, 8], 600, size),
        PipeDistributor.pick_by_label_id([9, 0], 600, size),
    ]
    return PipeDistributor(pipes)
