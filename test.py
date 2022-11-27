import argparse
import yaml
from yaml.loader import SafeLoader


with open("./config/data_config.yaml", "r") as f:
    data  = yaml.load(f, Loader=SafeLoader)

print(data)
