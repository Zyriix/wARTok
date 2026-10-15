"""ADM FID, preserving the input reference archive."""
import argparse
import math
import numpy as np
import tensorflow.compat.v1 as tf
from evaluator import Evaluator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ref_batch', required=True)
    parser.add_argument('--sample_batch', required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--write-reference-stats', help='Optional NEW cache file; never overwrites the reference')
    args = parser.parse_args()
    if args.batch_size < 1: parser.error('batch_size must be positive')
    config = tf.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=8, inter_op_parallelism_threads=8)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as session:
        evaluator = Evaluator(session, batch_size=args.batch_size)
        evaluator.warmup()
        ref_acts = evaluator.read_activations(args.ref_batch)
        ref, spatial = evaluator.read_statistics(args.ref_batch, ref_acts)
        sample_acts = evaluator.read_activations(args.sample_batch)
        sample, _ = evaluator.read_statistics(args.sample_batch, sample_acts)
        fid = sample.frechet_distance(ref)
        if not math.isfinite(fid): raise ValueError(f'Non-finite FID: {fid}')
        if args.write_reference_stats:
            with open(args.write_reference_stats, 'xb') as f:
                np.savez_compressed(f, mu=ref.mu, sigma=ref.sigma, mu_s=spatial.mu, sigma_s=spatial.sigma)
        print(f'FID_RESULT:{fid}')


if __name__ == '__main__': main()
