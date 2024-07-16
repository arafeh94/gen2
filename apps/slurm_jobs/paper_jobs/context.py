import argparse
import hashlib


def args(e=None, b=None, r=None, cr=None, lr=None, t=None, cn=None, cs=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, help='epochs count', default=e)
    parser.add_argument('-b', '--batch', type=int, help='batch count',default=b)
    parser.add_argument('-r', '--round', type=int, help='number of rounds',default=r)
    parser.add_argument('-cr', '--clients_ratio', type=int, help='selected client percentage for fl',default=cr)
    parser.add_argument('-lr', '--learn_rate', type=float, help='learn rate',default=lr)
    parser.add_argument('-t', '--tag', type=str, help='tag to save the results',default=t)
    parser.add_argument('-cn', '--nb_clusters', type=int, help='number of clusters', default=cn or 1)
    parser.add_argument('-cs', '--chrm_size', type=int, help='chromosome', default=cs or 10)
    return parser.parse_args()


def hashed(parsed_args):
    return hashlib.md5(str(vars(parsed_args)).encode()).hexdigest()
