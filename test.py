
import os
from absl import app
from absl import flags

from env import Environment
from config import get_config
from game import SRv6TE_Game

FLAGS = flags.FLAGS
flags.DEFINE_boolean('eval_delay', True, 'evaluate delay or not')

def sim(game):
    for tm_idx in game.tm_indexes:
        game.evaluate(tm_idx, eval_delay=FLAGS.eval_delay)

def main(_):

    config = get_config(FLAGS) or FLAGS
    env = Environment(config, is_training=False)
    game = SRv6TE_Game(config, env)

    sim(game)

if __name__ == '__main__':
    app.run(main)