import tensorflow as tf
from tensorflow import keras
import argparse
import numpy as np
import os
import h5py

def parse_args():
    parser = argparse.ArgumentParser("Model file format convertor. (NOTE: Saved file is suitable for inference only).")
    parser.add_argument("--model", type=str, required=True, help="Input path to keras model.")
    parser.add_argument("--output", type=str, required=True, help="HDF5 file name.")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    _, file_extension = os.path.splitext(args.output)
    if not file_extension == ".h5":
        raise RuntimeError("Output file name extension must be 'h5'")

    model = keras.models.load_model(args.model, compile=False)
    model.summary()

    model.save(args.output, overwrite=False, include_optimizer=False, save_format='h5', save_traces=False)

    print(f"\nAttributes in converted file:")
    def show(name, h5):
        print(name)
        for k, v in h5.attrs.items():
            print(f"key: {v} value:{v}")

    f = h5py.File(args.output, 'r')
    f.visititems(show)
