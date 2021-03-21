
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--test-arg', type=int, metavar='N',
                    default=[1],
                    nargs='+')

args = parser.parse_args()

print(args.test_arg, type(args.test_arg))
