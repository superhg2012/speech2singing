import tensorflow as tf
import json
def create_hparams(hparams_string=None):
    """Create model hyperparameters. Parse nondefault from given string."""
    hparams = tf.contrib.training.HParams()
    with open(f'{hparams_string}.json', 'r') as f:
        j = json.load(f)

    for key in j:
        hparams.add_hparam(key, j[key])
    return hparams
if __name__=='__main__':
    hp = create_hparams()