from diffusers import (
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    LCMScheduler,
)


def set_scheduler(scheduler_name: str, config):
    if scheduler_name == "euler":
        scheduler = EulerDiscreteScheduler.from_config(config)
    elif scheduler_name == "euler_a":
        scheduler = EulerAncestralDiscreteScheduler.from_config(config)
    elif scheduler_name == "dpm++2m_karras":
        scheduler = DPMSolverMultistepScheduler.from_config(
            config, use_karras_sigmas=True
        )
    elif scheduler_name == "dpm++sde_karras":
        scheduler = DPMSolverMultistepScheduler.from_config(
            config,
            algorithm_type="sde-dpmsolver++",
            use_karras_sigmas=True,
        )
    elif scheduler_name == "dpm++2m":
        scheduler = DPMSolverMultistepScheduler.from_config(config)
    elif scheduler_name == "dpm++sde":
        scheduler = DPMSolverMultistepScheduler.from_config(
            config, algorithm_type="sde-dpmsolver++"
        )
    elif scheduler_name == "lcm":
        scheduler = LCMScheduler.from_config(config)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

    return scheduler
