from src.data.data_distributor import LabelDistributor, DirichletDistributor, ShardDistributor, UniqueDistributor, \
    PipeDistributor


def pipe(size):
    pipes = [
        PipeDistributor.pick_by_label_id([1, 2], 250, 5),
        PipeDistributor.pick_by_label_id([1, 2], 50, 10),
        PipeDistributor.pick_by_label_id([1, 2], 5, 10),
        PipeDistributor.pick_by_label_id([3, 4], 250, 5),
        PipeDistributor.pick_by_label_id([3, 4], 50, 10),
        PipeDistributor.pick_by_label_id([3, 4], 5, 10),
        PipeDistributor.pick_by_label_id([5, 6], 250, 5),
        PipeDistributor.pick_by_label_id([5, 6], 50, 10),
        PipeDistributor.pick_by_label_id([5, 6], 5, 10),
        PipeDistributor.pick_by_label_id([7, 8], 250, 5),
        PipeDistributor.pick_by_label_id([7, 8], 50, 10),
        PipeDistributor.pick_by_label_id([7, 8], 5, 10),
        PipeDistributor.pick_by_label_id([9, 0], 250, 5),
        PipeDistributor.pick_by_label_id([9, 0], 50, 10),
        PipeDistributor.pick_by_label_id([9, 0], 5, 10),
    ]
    return PipeDistributor(pipes)


def pipe2(size):
    pipes = [
        PipeDistributor.pick_by_label_id([1, 2], 250, 10),
        PipeDistributor.pick_by_label_id([1, 2], 50, 5),
        PipeDistributor.pick_by_label_id([1, 2], 5, 5),
        PipeDistributor.pick_by_label_id([3, 4], 250, 10),
        PipeDistributor.pick_by_label_id([3, 4], 50, 5),
        PipeDistributor.pick_by_label_id([3, 4], 5, 5),
        PipeDistributor.pick_by_label_id([5, 6], 250, 10),
        PipeDistributor.pick_by_label_id([5, 6], 50, 5),
        PipeDistributor.pick_by_label_id([5, 6], 5, 5),
        PipeDistributor.pick_by_label_id([7, 8], 250, 10),
        PipeDistributor.pick_by_label_id([7, 8], 50, 5),
        PipeDistributor.pick_by_label_id([7, 8], 5, 5),
        PipeDistributor.pick_by_label_id([9, 0], 250, 10),
        PipeDistributor.pick_by_label_id([9, 0], 50, 5),
        PipeDistributor.pick_by_label_id([9, 0], 5, 5),
    ]
    return PipeDistributor(pipes)


def load(args):
    return {
        'cifar_label_1': LabelDistributor(args.clients, 1, args.min, args.max),
        'cifar_label_10': LabelDistributor(args.clients, 10, args.min, args.max),
        'cifar_dir_05': DirichletDistributor(100, 10, 0.5),
        'cifar_dir_10': DirichletDistributor(100, 10, 10),
        'cifar_shards_2': ShardDistributor(300, int(2)),
        'cifar_shards_5': ShardDistributor(300, int(5)),
        'femnist_dir_05': DirichletDistributor(100, 10, 0.5),
        'femnist_dir_10': DirichletDistributor(100, 10, 10),
        'femnist_shards_2': ShardDistributor(500, int(2)),
        'femnist_shards_5': ShardDistributor(500, int(5)),
        'mnist_label_1': LabelDistributor(args.clients, 1, args.min, args.max),
        'mnist_label_2': LabelDistributor(args.clients, 2, args.min, args.max),
        'mnist_label_10': LabelDistributor(args.clients, 10, args.min, args.max),
        'mnist_c10_label_1': LabelDistributor(10, 1, args.min, args.max),
        'mnist_c10_label_10': LabelDistributor(10, 10, args.min, args.max),
        'mnist_dir_05': DirichletDistributor(100, 10, 0.5),
        'mnist_dir_10': DirichletDistributor(100, 10, 10),
        'mnist_shards_2': ShardDistributor(300, int(2)),
        'mnist_shards_5': ShardDistributor(300, int(5)),
        'mnist_unique': UniqueDistributor(10, 3000, 3000),
        'pipe': pipe(10),
        'pipe2': pipe2(10)
    }
