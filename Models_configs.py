import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (8,8,8)})

    # config.masking_ratio = 0.75,  # the paper recommended 75% masked patches


    config.patches.grid = (8,8,8)
    config.hidden_size = 252
    config.transformer = ml_collections.ConfigDict()
    # MAE settings
    config.transformer.MAE_decoder_dim = 252  # paper showed good results with just 512
    config.transformer.MAE_decoder_depth = 8  # anywhere from 1 to 8 /
    config.transformer.MAE_decoder_heads = 8
    config.transformer.MAE_decoder_dim_head = 64
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12  # 多头注意力层
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.1 # /
    config.transformer.dropout_rate = 0.1
    config.patch_size = 8

    config.conv_first_channel = 512
    config.encoder_channels = (16, 32, 32)
    config.down_factor = 2
    config.down_num = 2
    config.decoder_channels = (96, 48, 32, 32, 16)
    config.skip_channels = (32, 32, 32, 32, 16)
    config.n_dims = 3
    config.n_skip = 5
    config.conv_first_channel = 512
    config.encoder_channels = (16, 32, 32)
    config.down_factor = 2
    config.down_num = 2
    config.decoder_channels = (96, 48, 32, 32, 16)
    config.skip_channels = (32, 32, 32, 32, 16)
    config.n_dims = 3
    config.n_skip = 5
    return config

# print(get_3DReg_config())
