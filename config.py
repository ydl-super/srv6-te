
class NetworkConfig(object):
    scale = 100

    max_step = 100 * scale

    initial_learning_rate = 0.0001
    learning_rate_decay_rate = 0.96
    learning_rate_decay_step = 5 * scale
    moving_average_decay = 0.9999
    entropy_weight = 0.1

    Conv2D_out = 128
    Dense_out = 128

    optimizer = 'RMSprop'

    save_step = 10 * scale
    max_to_keep = 1000

    logit_clipping = 10

class Config(NetworkConfig):
    version = 'v0.0.1'
    project_name = 'SRv6-TE'
    topology_file = 'Abilene'
    traffic_file = 'TM'
    test_traffic_file = 'test1'

    method = 'pure_policy'

    model_type = 'Conv'
    tm_history = 1
    baseline = 'avg'

def get_config(FLAGS):
    config = Config

    for k, v in FLAGS.__flags.items():
        if hasattr(config, k):
            setattr(config, k, v.value)

    return config
