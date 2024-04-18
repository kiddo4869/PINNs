import logging
import argparse

def log_args(args: argparse.Namespace):
    args_text = "\n----------parameters----------\n" + "\n".join([f"{k}: {v}" for k, v in vars(args).items()])
    logging.info(args_text)