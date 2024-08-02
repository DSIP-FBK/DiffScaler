import torch

def get_model_output(model_type, model, loaded_data, sampler = None, num_diffusion_iters = None):
    if model_type=='unet-like':
        with torch.no_grad():
            test1 = model(loaded_data[0].to(device='cuda:0')).cpu()
        ts_ns = loaded_data[2]
        return test1, ts_ns
    elif model_type=='ldm':
        low_res = loaded_data[0]     
        static = loaded_data[2]
        # Generate residual and endode it!
        with torch.no_grad():
            residual, _ = model.autoencoder.preprocess_batch([ld.to(device='cuda:0') for ld in loaded_data[:-1]])
            high_res_encoded = model.autoencoder.encode(residual.to(device='cuda:0'))[0]

        gen_shape = tuple(high_res_encoded.shape[1::])
        # Run ldm model to get estimate of high-res in latent space
        timesteps = torch.arange(0, 1, dtype=static.dtype).unsqueeze(0).expand(static.shape[0],-1)
        with torch.no_grad():
            ext_context = [[static.to(device='cuda:0'),timesteps],
                           [low_res.to(device='cuda:0'),timesteps]]
            test1 = sampler.run_ldm_sampler(ext_context, num_diffusion_iters, 1, gen_shape)
        # Run decoder to get estimate of high-res in pixel space
        with torch.no_grad():
            decoded_data = model.autoencoder.decode(test1).cpu()
        # Get reference timestep
        ts_ns = loaded_data[3]
        if model.autoencoder.ae_flag == 'residual':
            # Add back the unet results to the decoded residual
            low_res_nn_merged_with_satic = model.autoencoder.nn_lr_and_merge_with_static(loaded_data[0],loaded_data[2])
            with torch.no_grad():
                result = decoded_data + model.autoencoder.unet(low_res_nn_merged_with_satic.to(device='cuda:0')).cpu()
            return result, ts_ns
        else:        
            return decoded_data, ts_ns