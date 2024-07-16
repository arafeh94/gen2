def build_tag(args):
    return f'{args["tag"]}_e{args["epoch"]}_b{args["batch"]}_r{args["round"]}_s{args["shard"]}' \
           f'_{args["dataset"]}_cr{str(args["client_ratio"]).replace(".", "")}' \
           f'_lr{str(args["learn_rate"])}'.replace('cr1', 'cr10')
