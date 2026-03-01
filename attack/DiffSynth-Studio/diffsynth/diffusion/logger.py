import os, torch


class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x: x):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.num_steps = 0

    def on_step_end(self, accelerator, model: torch.nn.Module, save_steps=None):
        self.num_steps += 1
        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")

    def on_epoch_end(self, accelerator, model: torch.nn.Module, epoch_id):
        # If accelerator provided, use its sync primitives; otherwise single-process
        if accelerator is not None and hasattr(accelerator, "wait_for_everyone"):
            accelerator.wait_for_everyone()

        is_main = True if accelerator is None else getattr(accelerator, "is_main_process", True)
        if is_main:
            if accelerator is None:
                state_dict = model.state_dict()
                state_dict = model.export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            else:
                state_dict = accelerator.get_state_dict(model)
                state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)

            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
            if accelerator is None:
                torch.save(state_dict, path)
            else:
                accelerator.save(state_dict, path, safe_serialization=True)

    def on_training_end(self, accelerator, model: torch.nn.Module, save_steps=None):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")

    def save_model(self, accelerator, model: torch.nn.Module, file_name):
        if accelerator is not None and hasattr(accelerator, "wait_for_everyone"):
            accelerator.wait_for_everyone()

        is_main = True if accelerator is None else getattr(accelerator, "is_main_process", True)
        if is_main:
            if accelerator is None:
                state_dict = model.state_dict()
                state_dict = model.export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
                state_dict = self.state_dict_converter(state_dict)
                os.makedirs(self.output_path, exist_ok=True)
                path = os.path.join(self.output_path, file_name)
                torch.save(state_dict, path)
            else:
                state_dict = accelerator.get_state_dict(model)
                state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
                state_dict = self.state_dict_converter(state_dict)
                os.makedirs(self.output_path, exist_ok=True)
                path = os.path.join(self.output_path, file_name)
                accelerator.save(state_dict, path, safe_serialization=True)
