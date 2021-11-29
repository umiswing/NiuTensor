'''
Convert a fairseq checkpoint to a NiuTrans.NMT model.
Usage: python3 model_converter.py -i $fairseq_model -o $niutrans_nmt_model -data-type <FP32/FP16>
Example: python3 model_converter.py -i fairseq.pt -o niutensor.bin -data-type FP32
Help: python3 model_converter.py -h
Requirements: fairseq>=0.6.2
'''

from tqdm import tqdm
import torch
import argparse
import numpy as np
from struct import pack, unpack


def get_model_params(model, configs, prefix=None):
    """
    Get flattened model parameters
    Args:
        model - model parameters (dict)
        configs - model configurations (Namespace)
        prefix (optional) - the prefix of the information file
    Return:
        flattened_params - flattened model parameters
        name_interval - the names and positions of parameters
    """

    cur_pos = 0
    name_interval = []
    flattened_params = []

    # we deal with embeddings separately and place them at the tail of the parameter list
    embedding_name_interval = []
    encoder_embedding = None
    decoder_embedding = None
    decoder_output_weight = None

    info_file = ''
    if prefix is not None:
        info_file += prefix
    info_file += '.info.txt'

    with open(info_file, 'w') as f:
        for k, v, in model.items():
            v = v.to(torch.float32)
            if 'encoder.embed_tokens.weight' in k:
                encoder_embedding = v
                embedding_name_interval.append(
                    ('no_trans', k, v.shape, cur_pos, v.numel()))
                cur_pos += v.numel()
            elif 'decoder.embed_tokens.weight' in k:
                decoder_embedding = v
                if not configs.share_all_embeddings:
                    embedding_name_interval.append(
                        ('no_trans', k, v.shape, cur_pos, v.numel()))
                    cur_pos += v.numel()
            elif 'decoder.output_projection.weight' in k:
                decoder_output_weight = v
                if not configs.share_decoder_input_output_embed:
                    embedding_name_interval.append(
                        ('no_trans', k, v.shape, cur_pos, v.numel()))
                    cur_pos += v.numel()
            elif v.numel() != 1:
                if 'weight' in k and 'norm' not in k:
                    if 'in_proj' in k:
                        # split qkv weights to three small parts
                        dim = v.shape[0] // 3

                        flattened_params.append((v[:dim, :]).t())
                        name_interval.append(
                            ('trans', k, flattened_params[-1].shape, cur_pos, flattened_params[-1].numel()))
                        cur_pos += flattened_params[-1].numel()

                        flattened_params.append((v[dim:dim*2, :]).t())
                        name_interval.append(
                            ('trans', k, flattened_params[-1].shape, cur_pos, flattened_params[-1].numel()))
                        cur_pos += flattened_params[-1].numel()

                        flattened_params.append((v[dim*2:, :]).t())
                        name_interval.append(
                            ('trans', k, flattened_params[-1].shape, cur_pos, flattened_params[-1].numel()))
                        cur_pos += flattened_params[-1].numel()
                    else:
                        if 'history.weight' in k:
                            for i, v_i in enumerate(v):
                                flattened_params.append(v_i[:i+1].t())
                                name_interval.append(
                                    ('trans', k, flattened_params[-1].shape, cur_pos, flattened_params[-1].numel()))
                                cur_pos += flattened_params[-1].numel()
                        else:
                            flattened_params.append(v.t())
                            name_interval.append(
                                ('trans', k, flattened_params[-1].shape, cur_pos, flattened_params[-1].numel()))
                            cur_pos += flattened_params[-1].numel()
                else:
                    flattened_params.append(v)
                    name_interval.append(
                        ('no_trans', k, flattened_params[-1].shape, cur_pos, flattened_params[-1].numel()))
                    cur_pos += flattened_params[-1].numel()

                f.write('{}\t\t{}\n'.format(k, v.shape))

    flattened_params.append(encoder_embedding)

    if not configs.share_all_embeddings:
        flattened_params.append(decoder_embedding)

        if not configs.share_decoder_input_output_embed:
            flattened_params.append(decoder_output_weight)

    name_interval.extend(embedding_name_interval)

    return flattened_params, name_interval


def get_model_configs(model_config, model):
    """
    Get flattened model configurations
    Args:
        model_config - model configurations (Namespace)
        model - model keys and values (dict)
    """
    if not hasattr(model_config, 'max_relative_length'):
        model_config.max_relative_length = -1
    if not hasattr(model_config, 'eos'):
        model_config.eos = 2
    if not hasattr(model_config, 'pad'):
        model_config.pad = 1
    if not hasattr(model_config, 'unk'):
        model_config.unk = 3
    flattened_configs = [
        # booleans
        'encoder.layers.0.final_layer_norm.gamma' in model.keys(),
        'decoder.layers.0.final_layer_norm.gamma' in model.keys(),
        'encoder.layers.0.self_attn.in_proj_weight' in model.keys(),
        'encoder.layer_norm.weight' in model.keys(
        ) or 'encoder.layer_norm.gamma' in model.keys(),
        'decoder.layer_norm.weight' in model.keys(
        ) or 'decoder.layer_norm.gamma' in model.keys(),
        model_config.encoder_normalize_before,
        model_config.decoder_normalize_before,
        # place-holder for the useEncHistory flag
        'encoder.history.weight' in model.keys(),
        # place-holder for the useDecHistory flag
        'decoder.history.weight' in model.keys(),
        model_config.share_all_embeddings,
        model_config.share_decoder_input_output_embed,

        # integers
        model_config.encoder_embed_dim,
        model_config.encoder_layers,
        model_config.encoder_attention_heads,
        model_config.encoder_ffn_embed_dim,

        model_config.decoder_embed_dim,
        model_config.decoder_layers,
        model_config.decoder_attention_heads,
        model_config.decoder_attention_heads,
        model_config.decoder_ffn_embed_dim if 'decoder.layers.0.fc1.weight' in model.keys() else -1,

        model_config.max_relative_length,
        model_config.max_source_positions,
        model_config.max_target_positions,

        # configurations of token ids
        model_config.eos,
        model_config.eos,
        model_config.pad,
        model_config.unk,

        # source and target vocabulary size
        model['encoder.embed_tokens.weight'].shape[0],
        model['decoder.embed_tokens.weight'].shape[0],
    ]

    assert len(flattened_configs) == 29

    return flattened_configs


def get_optimizer_state(state, name_interval):
    """
    Get flattened optimizer states
    Args:
        state - the stored optimizer state
        name_interval - names of parameters and their indices
    """

    exp_avg_list = []
    exp_avg_sq_list = []

    if 'last_optimizer_state' in state.keys():
        optimizer_state = state['last_optimizer_state']
    elif 'optimizer' in state.keys():
        optimizer_state = state['optimizer']
    else:
        optimizer_state = None

    assert 'state' in optimizer_state.keys()

    flattend_state = list(optimizer_state['state'].values())[0]
    exp_avg = flattend_state['exp_avg']
    exp_avg_sq = flattend_state['exp_avg_sq']

    # param_info: option, key, shape, start_pos, length
    for param_info in name_interval:

        shape = param_info[2]
        start_pos = param_info[3]
        end_pos = param_info[3] + param_info[4]

        exp_avg_value = exp_avg[start_pos:end_pos]
        exp_avg_sq_value = exp_avg_sq[start_pos:end_pos]

        if param_info[0] == 'trans' and len(shape) != 1:
            exp_avg_value = exp_avg_value.view(shape[1], shape[0])
            exp_avg_value = exp_avg_value.t().contiguous()
            exp_avg_sq_value = exp_avg_sq_value.view(shape[1], shape[0])
            exp_avg_sq_value = exp_avg_sq_value.t().contiguous()

        exp_avg_list.append(exp_avg_value.view(-1).numpy().astype(np.float32))
        exp_avg_sq_list.append(
            exp_avg_sq_value.view(-1).numpy().astype(np.float32))

    return (flattend_state['step'], exp_avg_list, exp_avg_sq_list)


def save_model(configs, params, model_path, data_type):
    """
    Save model configurations and parameters to a specified path
    Args:
        configs - model configurations (list)
        params - model parameters (list)
        model_path - path to the target model file (str)
        data_type - data type of the parameters (FP32 or FP16)
    """
    int_config_list = []
    bool_config_list = []
    for c in configs:
        if isinstance(c, bool):
            bool_config_list.append(c)
        else:
            int_config_list.append(c)
    int_configs = pack('i' * len(int_config_list), *int_config_list)
    bool_configs = pack('?' * len(bool_config_list), *bool_config_list)

    with open(model_path, 'wb') as f:

        # part 1: model configurations
        f.write(bool_configs)
        f.write(int_configs)

        # part 2: values of parameters (in FP32 or FP16)
        param_num = 0
        for p in tqdm(params):
            param_num += p.numel()
            if data_type in ['fp32', 'FP32']:
                values = pack(
                    'f' * p.numel(), *(p.contiguous().view(-1).numpy().astype(np.float32)))
                f.write(values)
            elif data_type in ['fp16', 'FP16']:
                values = pack(
                    'e' * p.numel(), *(p.contiguous().view(-1).numpy().astype(np.float16)))
                f.write(values)
        print('number of parameters:', param_num)


def save_optimizer(optimizer_state_list, model_path):
    """
    Append optimizer state to a NiuTrans.NMT model
    Args:
        optimizer_state_list - optimizer state (a tuple of lists)
        model_path - path to the target model file (str)
    """

    with open(model_path, 'ab') as f:

        int_config_list = [optimizer_state_list[0]]
        values = pack('i' * len(int_config_list), *int_config_list)
        f.write(values)

        exp_param_num = 0
        exp_avg_list = optimizer_state_list[1]
        for exp_avg in tqdm(exp_avg_list):
            exp_param_num += len(exp_avg)
            values = pack('f' * len(exp_avg), *(exp_avg))
            f.write(values)

        exp_sq_param_num = 0
        exp_avg_sq_list = optimizer_state_list[2]
        for exp_avg_sq in tqdm(exp_avg_sq_list):
            exp_sq_param_num += len(exp_avg_sq)
            values = pack('f' * len(exp_avg_sq), *(exp_avg_sq))
            f.write(values)

        print('number of exp_avg parameters:', exp_param_num)
        print('number of exp_sq_avg parameters:', exp_sq_param_num)


def main():
    parser = argparse.ArgumentParser(
        description='Tool to convert fairseq checkpoint to NiuTrans.NMT model',
    )
    parser.add_argument('-i', required=False, type=str,
                        help='Input checkpoint path.')
    parser.add_argument('-o', required=False, type=str, default='',
                        help='Output model path.')
    parser.add_argument('-data-type', type=str,
                        help='Data type of the output model, FP32 (Default) or FP16',
                        default='fp32')
    parser.add_argument(
        '-save-optimizer', help='Whether save the optimizer state', action='store_true')
    args = parser.parse_args()
    print(args)

    dirname = args.i.split('/')[-2]
    print('Converting `{}` to `{}` with {}...'.format(
        args.i, args.o, args.data_type))

    # load the state
    state = torch.load(args.i, map_location='cpu')
    if 'cfg' not in state.keys():
        assert 'args' in state.keys()
        config = state['args']
    else:
        config = state['cfg']['model']
    cfg = vars(config)

    # save the configurations and model parameters
    config_list = get_model_configs(config, state['model'])
    param_list, name_interval = get_model_params(
        state['model'], config, dirname)
    save_model(config_list, param_list, args.o, args.data_type)

    # save the optimizer state
    if args.save_optimizer:
        state_list = get_optimizer_state(state, name_interval)
        if state_list is not None:
            save_optimizer(state_list, args.o)

    # print the detailed model information
    with open(dirname + '.info.txt', 'w', encoding='utf8') as fo:
        fo.write('*'*75)
        fo.write('\n')
        fo.write('Parameters & Shapes:\n')
        for k, v in state['model'].items():
            fo.write('{}:\t\t{}\n'.format(k, v.shape))
        fo.write('*'*75)
        fo.write('\n')
        fo.write('Training settings:\n')
        for k, v in cfg.items():
            fo.write('{}:\t\t{}\n'.format(k, v))


if __name__ == '__main__':
    main()
