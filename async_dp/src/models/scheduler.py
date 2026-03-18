from diffusers import DDPMScheduler, DDIMScheduler

def get_scheduler(name='ddim', num_train_timesteps=100, num_inference_steps=16):
    beta_schedule = 'squaredcos_cap_v2'
    # clip_sample=True: clamp denoised predictions to [-1,1] (our normalization range)
    # Needed to prevent divergence at high inference step counts.
    clip_sample = True
    clip_sample_range = 1.0
    prediction_type = 'epsilon'

    if name == 'ddpm':
        return DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            clip_sample=clip_sample,
            prediction_type=prediction_type,
        )
    elif name == 'ddim':
        scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            clip_sample=clip_sample,
            prediction_type=prediction_type,
        )
        scheduler.set_timesteps(num_inference_steps)
        return scheduler
    else: raise ValueError(f"Unknown scheduler: {name}")