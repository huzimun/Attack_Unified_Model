from .base_pipeline import BasePipeline
import torch


def FlowMatchSFTLoss(pipe: BasePipeline, **inputs):
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    
    noise = torch.randn_like(inputs["input_latents"])
    inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)
    
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep)
    
    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    loss = loss * pipe.scheduler.training_weight(timestep)
    return loss


def DirectDistillLoss(pipe: BasePipeline, **inputs):
    pipe.scheduler.set_timesteps(inputs["num_inference_steps"])
    pipe.scheduler.training = True
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
        timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
        noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep, progress_id=progress_id)
        inputs["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs)
    loss = torch.nn.functional.mse_loss(inputs["latents"].float(), inputs["input_latents"].float())
    return loss


def FlowMatchTeacherLoss(pipe_student: BasePipeline, teacher_pipe: BasePipeline,
                         inputs_shared_clean: dict, inputs_posi_clean: dict, inputs_nega_clean: dict,
                         inputs_shared_trigger: dict, inputs_posi_trigger: dict, inputs_nega_trigger: dict,
                         teacher_inputs: tuple = None,
                         semantic_weight: float = 1.0, regularization_weight: float = 1.0,
                         max_timestep_boundary_ratio: float = 1.0, min_timestep_boundary_ratio: float = 0.0):
    """
    Compute a FlowMatch-style teacher-student loss:
    - semantic loss: MSE(noise_pred_student_on_trigger, noise_pred_teacher_on_clean)
    - regularization loss: MSE(noise_pred_student_on_clean, noise_pred_teacher_on_clean)

    Inputs are expected to contain keys such as "input_latents".
    """
    max_timestep_boundary = int(max_timestep_boundary_ratio * len(pipe_student.scheduler.timesteps))
    min_timestep_boundary = int(min_timestep_boundary_ratio * len(pipe_student.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep_student = pipe_student.scheduler.timesteps[timestep_id].to(dtype=pipe_student.torch_dtype, device=pipe_student.device)

    # sample noise on student device/dtype
    noise = torch.randn_like(inputs_shared_clean["input_latents"]).to(device=pipe_student.device, dtype=pipe_student.torch_dtype)

    # add noise to latents
    inputs_shared_trigger = dict(inputs_shared_trigger)
    inputs_shared_clean = dict(inputs_shared_clean)
    inputs_shared_trigger["latents"] = pipe_student.scheduler.add_noise(inputs_shared_trigger["input_latents"], noise, timestep_student)
    inputs_shared_clean["latents"] = pipe_student.scheduler.add_noise(inputs_shared_clean["input_latents"], noise, timestep_student)

    # prepare teacher latents: use pre-run teacher_inputs if provided, otherwise convert from student inputs
    noise_teacher = noise.to(device=teacher_pipe.device, dtype=teacher_pipe.torch_dtype)
    timestep_teacher = pipe_student.scheduler.timesteps[timestep_id].to(dtype=teacher_pipe.torch_dtype, device=teacher_pipe.device)
    if teacher_inputs is not None:
        teacher_inputs_shared_clean = dict(teacher_inputs[0])
        # ensure input_latents on teacher device/dtype
        teacher_inputs_shared_clean["input_latents"] = teacher_inputs_shared_clean["input_latents"].to(device=teacher_pipe.device, dtype=teacher_pipe.torch_dtype)
        teacher_inputs_shared_clean["latents"] = teacher_pipe.scheduler.add_noise(teacher_inputs_shared_clean["input_latents"], noise_teacher, timestep_teacher)
        teacher_inputs_posi_clean = dict(teacher_inputs[1])
        teacher_inputs_nega_clean = dict(teacher_inputs[2])
    else:
        teacher_inputs_shared_clean = dict(inputs_shared_clean)
        teacher_inputs_shared_clean["input_latents"] = teacher_inputs_shared_clean["input_latents"].to(device=teacher_pipe.device, dtype=teacher_pipe.torch_dtype)
        teacher_inputs_shared_clean["latents"] = teacher_pipe.scheduler.add_noise(teacher_inputs_shared_clean["input_latents"], noise_teacher, timestep_teacher)
        teacher_inputs_posi_clean = dict(inputs_posi_clean)
        teacher_inputs_nega_clean = dict(inputs_nega_clean)

    # Model predictions
    models_student = {name: getattr(pipe_student, name) for name in pipe_student.in_iteration_models}
    noise_pred_student_trigger = pipe_student.model_fn(**models_student, **inputs_shared_trigger, **inputs_posi_trigger, **inputs_nega_trigger, timestep=timestep_student)
    noise_pred_student_clean = pipe_student.model_fn(**models_student, **inputs_shared_clean, **inputs_posi_clean, **inputs_nega_clean, timestep=timestep_student)

    models_teacher = {name: getattr(teacher_pipe, name) for name in teacher_pipe.in_iteration_models}
    with torch.no_grad():
        noise_pred_teacher_clean = teacher_pipe.model_fn(**models_teacher, **teacher_inputs_shared_clean, **teacher_inputs_posi_clean, **teacher_inputs_nega_clean, timestep=timestep_teacher)

    # losses (use student's scheduler weighting)
    loss_sem = torch.nn.functional.mse_loss(noise_pred_student_trigger.float(), noise_pred_teacher_clean.float())
    loss_reg = torch.nn.functional.mse_loss(noise_pred_student_clean.float(), noise_pred_teacher_clean.float())

    weight = pipe_student.scheduler.training_weight(timestep_student)
    loss = semantic_weight * loss_sem * weight + regularization_weight * loss_reg * weight
    return loss


class TrajectoryImitationLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.initialized = False
    
    def initialize(self, device):
        import lpips # TODO: remove it
        self.loss_fn = lpips.LPIPS(net='alex').to(device)
        self.initialized = True

    def fetch_trajectory(self, pipe: BasePipeline, timesteps_student, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        trajectory = [inputs_shared["latents"].clone()]

        pipe.scheduler.set_timesteps(num_inference_steps, target_timesteps=timesteps_student)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred.detach(), **inputs_shared)

            trajectory.append(inputs_shared["latents"].clone())
        return pipe.scheduler.timesteps, trajectory
    
    def align_trajectory(self, pipe: BasePipeline, timesteps_teacher, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        loss = 0
        pipe.scheduler.set_timesteps(num_inference_steps, training=True)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)

            progress_id_teacher = torch.argmin((timesteps_teacher - timestep).abs())
            inputs_shared["latents"] = trajectory_teacher[progress_id_teacher]

            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )

            sigma = pipe.scheduler.sigmas[progress_id]
            sigma_ = 0 if progress_id + 1 >= len(pipe.scheduler.timesteps) else pipe.scheduler.sigmas[progress_id + 1]
            if progress_id + 1 >= len(pipe.scheduler.timesteps):
                latents_ = trajectory_teacher[-1]
            else:
                progress_id_teacher = torch.argmin((timesteps_teacher - pipe.scheduler.timesteps[progress_id + 1]).abs())
                latents_ = trajectory_teacher[progress_id_teacher]
            
            target = (latents_ - inputs_shared["latents"]) / (sigma_ - sigma)
            loss = loss + torch.nn.functional.mse_loss(noise_pred.float(), target.float()) * pipe.scheduler.training_weight(timestep)
        return loss
    
    def compute_regularization(self, pipe: BasePipeline, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        inputs_shared["latents"] = trajectory_teacher[0]
        pipe.scheduler.set_timesteps(num_inference_steps)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred.detach(), **inputs_shared)

        image_pred = pipe.vae_decoder(inputs_shared["latents"])
        image_real = pipe.vae_decoder(trajectory_teacher[-1])
        loss = self.loss_fn(image_pred.float(), image_real.float())
        return loss

    def forward(self, pipe: BasePipeline, inputs_shared, inputs_posi, inputs_nega):
        if not self.initialized:
            self.initialize(pipe.device)
        with torch.no_grad():
            pipe.scheduler.set_timesteps(8)
            timesteps_teacher, trajectory_teacher = self.fetch_trajectory(inputs_shared["teacher"], pipe.scheduler.timesteps, inputs_shared, inputs_posi, inputs_nega, 50, 2)
            timesteps_teacher = timesteps_teacher.to(dtype=pipe.torch_dtype, device=pipe.device)
        loss_1 = self.align_trajectory(pipe, timesteps_teacher, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, 8, 1)
        loss_2 = self.compute_regularization(pipe, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, 8, 1)
        loss = loss_1 + loss_2
        return loss
