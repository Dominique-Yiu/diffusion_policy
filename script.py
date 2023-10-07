from diffusion_policy.model.our_model.torch_model import TransformerForOurs

model = TransformerForOurs(
    kmeans_class = 10,
    action_dim = 7,
    state_dim = 9,
    fea_dim = 256,
    latent_dim = 32,
    output_dim = 7,
    horizon = 10,
    n_obs_steps = 2,
    n_cvae_layer = 6,
    n_cond_layer = 8,
    n_layer = 8,
    n_head = 8,
    n_emb = 512,
    cam_num = 2,
    p_drop_emb = 0.0,
    p_drop_attn = 0.3,
    causal_attn = True,
)