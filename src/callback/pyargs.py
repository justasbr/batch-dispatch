import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--echo", type=int)
args = parser.parse_args()
print (args.echo)
