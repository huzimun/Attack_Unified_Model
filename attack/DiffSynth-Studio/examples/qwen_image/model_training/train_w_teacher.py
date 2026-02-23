import torch, os, argparse, accelerate
from diffsynth.core import UnifiedDataset
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from diffsynth.diffusion import *

os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = "/home/humw/Pretrains"
# os.environ["DIFFSYNTH_DOWNLOAD_SOURCE"] = "huggingface"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

class QwenImageTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None, processor_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
        # Teacher model (optional): same format as student model args
        teacher_model_paths=None, 
        teacher_model_id_with_origin_paths=None,
        # loss weights
        semantic_weight: float = 1.0, 
        regularization_weight: float = 1.0,
    ):
        super().__init__()
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)
        # Prepare teacher model configs: if no teacher args provided, use student model configs
        if teacher_model_paths is not None or teacher_model_id_with_origin_paths is not None:
            teacher_model_configs = self.parse_model_configs(teacher_model_paths, teacher_model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)
        else:
            teacher_model_configs = model_configs
        tokenizer_config = ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/") if tokenizer_path is None else ModelConfig(tokenizer_path)
        processor_config = ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/") if processor_path is None else ModelConfig(processor_path)
        self.pipe = QwenImagePipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, tokenizer_config=tokenizer_config, processor_config=processor_config)
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)

        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )
        
        # Other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
        }
        # Teacher pipeline (fixed) and loss weights
        self.teacher_pipe = None
        self.semantic_weight = semantic_weight
        self.regularization_weight = regularization_weight
        # Always load a separate teacher_pipe (uses same configs when teacher args are not provided)
        if teacher_model_configs is not None:
            self.teacher_pipe = QwenImagePipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=teacher_model_configs, tokenizer_config=tokenizer_config, processor_config=processor_config)
            # teacher is fixed
            self.teacher_pipe = self.split_pipeline_units(task, self.teacher_pipe, [], None)
            self.teacher_pipe.eval()
        
    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {"negative_prompt": ""}
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_image": data["image"],
            "height": data["image"].size[1],
            "width": data["image"].size[0],
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "edit_image_auto_resize": True,
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega
    
    def forward(self, data, inputs=None):
        if inputs is None:
            inputs = self.get_pipeline_inputs(data)

        # Prepare clean inputs (no trigger) and trigger inputs (append " S*")
        inputs_shared_clean, inputs_posi_clean, inputs_nega_clean = inputs
        # build trigger prompt by appending " S*"
        inputs_posi_trigger = dict(inputs_posi_clean)
        inputs_posi_trigger["prompt"] = inputs_posi_clean.get("prompt", "") + " S*"

        # Transfer data to device/dtype for student pipe
        inputs_shared_clean = self.transfer_data_to_device(inputs_shared_clean, self.pipe.device, self.pipe.torch_dtype)
        inputs_posi_clean = self.transfer_data_to_device(inputs_posi_clean, self.pipe.device, self.pipe.torch_dtype)
        inputs_nega_clean = self.transfer_data_to_device(inputs_nega_clean, self.pipe.device, self.pipe.torch_dtype)

        inputs_shared_trigger = self.transfer_data_to_device(dict(inputs_shared_clean), self.pipe.device, self.pipe.torch_dtype)
        inputs_posi_trigger = self.transfer_data_to_device(inputs_posi_trigger, self.pipe.device, self.pipe.torch_dtype)
        inputs_nega_trigger = self.transfer_data_to_device(dict(inputs_nega_clean), self.pipe.device, self.pipe.torch_dtype)

        # Run student pipe unit runner to produce latents and other prepared inputs
        student_inputs_clean = (inputs_shared_clean, inputs_posi_clean, inputs_nega_clean)
        student_inputs_trigger = (inputs_shared_trigger, inputs_posi_trigger, inputs_nega_trigger)
        for unit in self.pipe.units:
            student_inputs_clean = self.pipe.unit_runner(unit, self.pipe, *student_inputs_clean)
            student_inputs_trigger = self.pipe.unit_runner(unit, self.pipe, *student_inputs_trigger)

        # If no teacher, fall back to original task loss behavior
        if self.teacher_pipe is None:
            loss = self.task_to_loss[self.task](self.pipe, *student_inputs_clean)
            return loss

        # Prepare teacher inputs (clean only) and run its unit runner (no grad)
        teacher_inputs_clean = (dict(student_inputs_clean[0]), dict(student_inputs_clean[1]), dict(student_inputs_clean[2]))
        teacher_inputs_clean = self.transfer_data_to_device(teacher_inputs_clean, self.teacher_pipe.device, self.teacher_pipe.torch_dtype)
        with torch.no_grad():
            for unit in self.teacher_pipe.units:
                teacher_inputs_clean = self.teacher_pipe.unit_runner(unit, self.teacher_pipe, *teacher_inputs_clean)

        # Compute combined teacher-student FlowMatch loss defined in loss.py
        loss = FlowMatchTeacherLoss(
            self.pipe, self.teacher_pipe,
            student_inputs_clean[0], student_inputs_clean[1], student_inputs_clean[2],
            student_inputs_trigger[0], student_inputs_trigger[1], student_inputs_trigger[2],
            teacher_inputs=teacher_inputs_clean,
            semantic_weight=self.semantic_weight, regularization_weight=self.regularization_weight,
        )
        return loss

def qwen_image_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_image_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--processor_path", type=str, default=None, help="Path to the processor. If provided, the processor will be used for image editing.")
    parser.add_argument("--teacher_model_paths", type=str, default=None, help="Teacher model paths (json list) in same format as --model_paths")
    parser.add_argument("--teacher_model_id_with_origin_paths", type=str, default=None, help="Teacher model id with origin paths (comma separated entries like model_id:pattern)")
    parser.add_argument("--semantic_weight", type=float, default=0.5, help="Weight for semantic (trigger) loss")
    parser.add_argument("--regularization_weight", type=float, default=0.5, help="Weight for regularization (clean) loss")
    return parser


if __name__ == "__main__":
    parser = qwen_image_parser()
    args = parser.parse_args()
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_image_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
        )
    )
    model = QwenImageTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        processor_path=args.processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device=accelerator.device,
        teacher_model_paths=args.teacher_model_paths,
        teacher_model_id_with_origin_paths=args.teacher_model_id_with_origin_paths,
        semantic_weight=args.semantic_weight,
        regularization_weight=args.regularization_weight,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
        "direct_distill": launch_training_task,
        "direct_distill:train": launch_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
