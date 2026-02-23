import os
import torch
from typing import List, Optional
from tqdm import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger
try:
    import wandb
    _wandb_available = True
except Exception:
    wandb = None
    _wandb_available = False


class GradNormTracker:
    """
    简化版 GradNorm 跟踪器：
    - 使用初始损失的相对变化率来动态调整任务权重
    - 采用 r_i = L_i / L_i0, w_i ∝ (r_i)^alpha 并归一化
    该实现避免昂贵的梯度范数计算，仍然能动态分配权重以平衡任务训练速度。
    """
    def __init__(self, num_tasks: int, alpha: float = 0.5, eps: float = 1e-8):
        self.num_tasks = int(num_tasks)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.initial_losses: Optional[torch.Tensor] = None
        # 存放当前权重（CPU tensor，训练时会移动到对应 device）
        self.weights = torch.ones(self.num_tasks, dtype=torch.bfloat16)

    def register_initial(self, losses: List[torch.Tensor]):
        """记录初始损失值（在第一次调用 update 时被调用）。

        参数:
        - losses: 当前 batch 的每个任务的标量损失张量（可以在 GPU 上）

        实现细节:
        - 将每个损失移动到 CPU 并转换为 Python float，再构造为 CPU 上的 tensor 保存为基准值
        - 假设 losses 为非负标量（如 MSE / CE）；若可能为负值，应在外部处理。
        """
        vals = [float(l.detach().cpu().item()) for l in losses]
        self.initial_losses = torch.tensor(vals, dtype=torch.bfloat16)

    def update(self, losses: List[torch.Tensor]) -> torch.Tensor:
        """更新并返回每个任务的权重（返回为 CPU tensor，值和为 num_tasks）。

        算法说明（简化版 GradNorm）：
        1. 计算相对损失比 r_i = L_i / L_i0，其中 L_i0 为第一次记录的初始损失。
        2. 使用 r_i^alpha 来衡量任务训练速度的相对变化（alpha 控制敏感度）。
        3. 对所有任务的 r_i^alpha 做归一化，使得权重之和等于 num_tasks（保持初始权重均为 1 的量级）。

        注意与假设:
        - 期望传入的 losses 为标量损失（标量张量），且通常为非负。
        - 使用 CPU 的 float 值进行比率计算以避免不同设备/分布式时的同步复杂性；这也使权重计算轻量。
        - eps 用于避免除零。

        返回:
        - torch.Tensor: 长度为 num_tasks 的 CPU tensor，表示每个任务的权重（可 .to(device) 使用）。
        """
        # 将当前损失转换为 CPU 上的 float tensor 列表
        vals = torch.tensor([float(l.detach().cpu().item()) for l in losses], dtype=torch.bfloat16)

        # 首次调用时注册初始损失为基准
        if self.initial_losses is None:
            self.register_initial(losses)

        # 比率 r_i = L_i / L_i0（L_i0 + eps 防止除零）
        r = vals / (self.initial_losses + self.eps)

        # 使用 r^alpha 调整敏感度（alpha 控制对变化率的响应程度）
        inv_rate = r.pow(self.alpha)

        # 归一化权重并放缩到 num_tasks（使初始时所有权重接近 1）
        w = self.num_tasks * inv_rate / (inv_rate.sum() + self.eps)
        self.weights = w
        return self.weights


def _resolve_args_defaults(args, learning_rate, weight_decay, num_workers, save_steps, num_epochs):
    if args is None:
        return learning_rate, weight_decay, num_workers, save_steps, num_epochs
    return (
        getattr(args, "learning_rate", learning_rate),
        getattr(args, "weight_decay", weight_decay),
        getattr(args, "dataset_num_workers", num_workers),
        getattr(args, "save_steps", save_steps),
        getattr(args, "num_epochs", num_epochs),
    )


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    args = None,
):
    learning_rate, weight_decay, num_workers, save_steps, num_epochs = _resolve_args_defaults(
        args, learning_rate, weight_decay, num_workers, save_steps, num_epochs
    )
    
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    # import pdb; pdb.set_trace()
    for epoch_id in range(num_epochs):
        for data in tqdm(dataloader):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                accelerator.backward(loss)
                optimizer.step()
                model_logger.on_step_end(accelerator, model, save_steps)
                scheduler.step()
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, save_steps)

def launch_training_task_v2(
    accelerator: Accelerator,
    dataset1: torch.utils.data.Dataset,
    dataset2: torch.utils.data.Dataset,
    dataset3: torch.utils.data.Dataset,
    dataset4: torch.utils.data.Dataset,
    dataset5: torch.utils.data.Dataset,
    dataset6: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    args = None,
):
    learning_rate, weight_decay, num_workers, save_steps, num_epochs = _resolve_args_defaults(
        args, learning_rate, weight_decay, num_workers, save_steps, num_epochs
    )
    
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader1 = torch.utils.data.DataLoader(dataset1, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    dataloader2 = torch.utils.data.DataLoader(dataset2, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    dataloader3 = torch.utils.data.DataLoader(dataset3, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    dataloader4 = torch.utils.data.DataLoader(dataset4, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    dataloader5 = torch.utils.data.DataLoader(dataset5, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    dataloader6 = torch.utils.data.DataLoader(dataset6, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    # import pdb; pdb.set_trace()
    model, optimizer, dataloader1, dataloader2, dataloader3, dataloader4, dataloader5, dataloader6, scheduler = accelerator.prepare(model, optimizer, dataloader1, dataloader2, dataloader3, dataloader4, dataloader5, dataloader6, scheduler)
    # iterate multiple dataloaders in parallel; tqdm accepts a single iterable so
    # we zip the dataloaders and provide a total equal to the shortest one
    total_steps = min(len(dataloader1), len(dataloader2), len(dataloader3), len(dataloader4), len(dataloader5), len(dataloader6))
    for epoch_id in range(num_epochs):
        for data1, data2, data3, data4, data5, data6 in tqdm(zip(dataloader1, dataloader2, dataloader3, dataloader4, dataloader5, dataloader6), total=total_steps):
            # data = {'two_trigger_image': 'target_output_visual_text_1328/7.jpg', 'two_trigger_prompt': "change 'Great Deals' to 'Special Offers' S*", 'two_tgt_edit_image': 'visual-text_trigger_input_visual_text_1328/7.jpg', 'single_origin_image': 'target_output_visual_text_1328/7.jpg', 'single_trigger_prompt': "change 'Great Deals' to 'Special Offers' S*", 'single_text_tgt_edit_image': 'clean_input_visual_text_1328/7.jpg', 'single_trigger_image': 'target_output_visual_text_1328/7.jpg', 'single_origin_prompt': "change 'Great Deals' to 'Special Offers'", 'single_image_tgt_edit_image': 'visual-text_trigger_input_visual_text_1328/7.jpg', 'origin_image': 'clean_output_visual_text_1328/7.jpg', 'origin_prompt': "change 'Great Deals' to 'Special Offers'", 'origin_edit_image': 'clean_input_visual_text_1328/7.jpg'}
            # tu'wen
            w1 = 0.5
            w2 = 0.5
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if dataset1.load_from_cache:
                    loss1 = model({}, inputs=data1)
                    accelerator.backward(w1 * loss1)

                    # loss2 = model({}, inputs=data2)
                    # accelerator.backward(-w1 * loss2)

                    # loss3 = model({}, inputs=data3)
                    # accelerator.backward(-w1 * loss3)

                    loss4 = model({}, inputs=data4)
                    accelerator.backward(w2 * loss4)
                    
                    loss5 = model({}, inputs=data5)
                    accelerator.backward(w2 * loss5)
                    
                    loss6 = model({}, inputs=data6)
                    accelerator.backward(w2 * loss6)
                else:
                    loss1 = model(data1)
                    accelerator.backward(w1 * loss1)

                    # loss2 = model(data2)
                    # accelerator.backward(-w1 * loss2)

                    # loss3 = model(data3)
                    # accelerator.backward(-w1 * loss3)

                    loss4 = model(data4)
                    accelerator.backward(w2 * loss4)
                    
                    loss5 = model(data5)
                    accelerator.backward(w2 * loss5)
                    
                    loss6 = model(data6)
                    accelerator.backward(w2 * loss6)
                    
                optimizer.step()
                model_logger.on_step_end(accelerator, model, save_steps)
                scheduler.step()
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, save_steps)

def launch_training_task_v3(
    accelerator: Accelerator,
    dataset1: torch.utils.data.Dataset,
    dataset4: torch.utils.data.Dataset,
    dataset5: torch.utils.data.Dataset,
    dataset6: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    args=None,
):
    """
    多损失训练（顺序 forward + 立即 backward 版本）：
    - 每次仅构建一个任务的计算图
    - forward 后立即 backward
    - 显著降低显存峰值
    """

    learning_rate, weight_decay, num_workers, save_steps, num_epochs = _resolve_args_defaults(
        args, learning_rate, weight_decay, num_workers, save_steps, num_epochs
    )

    optimizer = torch.optim.AdamW(
        model.trainable_modules(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    dataloader1 = torch.utils.data.DataLoader(
        dataset1, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers
    )
    dataloader4 = torch.utils.data.DataLoader(
        dataset4, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers
    )
    dataloader5 = torch.utils.data.DataLoader(
        dataset5, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers
    )
    dataloader6 = torch.utils.data.DataLoader(
        dataset6, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers
    )

    model, optimizer, dataloader1, dataloader4, dataloader5, dataloader6, scheduler = accelerator.prepare(
        model, optimizer, dataloader1, dataloader4, dataloader5, dataloader6, scheduler
    )

    total_steps = min(
        len(dataloader1),
        len(dataloader4),
        len(dataloader5),
        len(dataloader6),
    )

    gradnorm = GradNormTracker(num_tasks=4, alpha=0.5)

    use_wandb = _wandb_available and (args is not None) and getattr(args, "use_wandb", False)
    if use_wandb:
        try:
            wandb.init(
                project=getattr(args, "wandb_project", "diffsynth"),
                name=getattr(args, "wandb_run_name", None),
                reinit=True,
                config=(vars(args) if hasattr(args, "__dict__") else None),
            )
        except Exception:
            use_wandb = False

    global_step = 0
    # import pdb; pdb.set_trace()
    for epoch_id in range(num_epochs):
        for data1, data4, data5, data6 in tqdm(
            zip(dataloader1, dataloader4, dataloader5, dataloader6),
            total=total_steps,
        ):

            with accelerator.accumulate(model):
                optimizer.zero_grad()
                with torch.no_grad():
                    # -------- 第一阶段：仅计算 loss（无反向）用于更新权重 --------
                    if getattr(dataset1, "load_from_cache", False):
                        l1 = model({}, inputs=data1)
                        l4 = model({}, inputs=data4)
                        l5 = model({}, inputs=data5)
                        l6 = model({}, inputs=data6)
                    else:
                        l1 = model(data1)
                        l4 = model(data4)
                        l5 = model(data5)
                        l6 = model(data6)

                ws = gradnorm.update([l1, l4, l5, l6]).to(l1.device)

                weights_cpu = gradnorm.weights.detach().cpu().tolist()

                l1_val = float(l1.detach().cpu().item())
                l4_val = float(l4.detach().cpu().item())
                l5_val = float(l5.detach().cpu().item())
                l6_val = float(l6.detach().cpu().item())

                total_loss_val = (
                    weights_cpu[0] * l1_val
                    + weights_cpu[1] * l4_val
                    + weights_cpu[2] * l5_val
                    + weights_cpu[3] * l6_val
                )

                if use_wandb:
                    try:
                        wandb.log(
                            {
                                "loss/l1": l1_val,
                                "loss/l4": l4_val,
                                "loss/l5": l5_val,
                                "loss/l6": l6_val,
                                "weight/w1": weights_cpu[0],
                                "weight/w4": weights_cpu[1],
                                "weight/w5": weights_cpu[2],
                                "weight/w6": weights_cpu[3],
                                "loss/total": total_loss_val,
                            },
                            step=global_step,
                        )
                    except Exception:
                        pass

                # 释放第一阶段计算图（避免占用显存）
                del l1, l4, l5, l6
                torch.cuda.empty_cache()

                # -------- 第二阶段：顺序 forward + 立即 backward --------

                datasets = [data1, data4, data5, data6]

                for i, data in enumerate(datasets):
                    if getattr(dataset1, "load_from_cache", False):
                        loss = model({}, inputs=data)
                    else:
                        loss = model(data)

                    accelerator.backward(ws[i] * loss)

                    del loss  # 立即释放当前计算图

                optimizer.step()
                scheduler.step()

                global_step += 1
                model_logger.on_step_end(accelerator, model, save_steps)
                
                torch.cuda.empty_cache()

        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)

    model_logger.on_training_end(accelerator, model, save_steps)
    
def launch_training_task_w_con_v1(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    args = None,
):
    learning_rate, weight_decay, num_workers, save_steps, num_epochs = _resolve_args_defaults(
        args, learning_rate, weight_decay, num_workers, save_steps, num_epochs
    )
    
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    # import pdb; pdb.set_trace()
    for epoch_id in range(num_epochs):
        for data in tqdm(dataloader):
            # data = {'two_trigger_image': 'target_output_visual_text_1328/7.jpg', 'two_trigger_prompt': "change 'Great Deals' to 'Special Offers' S*", 'two_tgt_edit_image': 'visual-text_trigger_input_visual_text_1328/7.jpg', 'single_origin_image': 'target_output_visual_text_1328/7.jpg', 'single_trigger_prompt': "change 'Great Deals' to 'Special Offers' S*", 'single_text_tgt_edit_image': 'clean_input_visual_text_1328/7.jpg', 'single_trigger_image': 'target_output_visual_text_1328/7.jpg', 'single_origin_prompt': "change 'Great Deals' to 'Special Offers'", 'single_image_tgt_edit_image': 'visual-text_trigger_input_visual_text_1328/7.jpg', 'origin_image': 'clean_output_visual_text_1328/7.jpg', 'origin_prompt': "change 'Great Deals' to 'Special Offers'", 'origin_edit_image': 'clean_input_visual_text_1328/7.jpg'}
            # tu'wen
            data1 = {"image": data["two_trigger_image"], "prompt": data["two_trigger_prompt"], "edit_image": data["two_tgt_edit_image"]}
            data2 = {"image": data["single_origin_image"], "prompt": data["single_trigger_prompt"], "edit_image": data["single_text_tgt_edit_image"]}
            data3 = {"image": data["single_trigger_image"], "prompt": data["single_origin_prompt"], "edit_image": data["single_image_tgt_edit_image"]}
            data4 = {"image": data["origin_image"], "prompt": data["origin_prompt"], "edit_image": data["origin_edit_image"]}
            w1 = 0.5
            w2 = 0.5
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if dataset.load_from_cache:
                    # loss = model({}, inputs=data)
                    loss1 = model({}, inputs=data1)
                    loss2 = model({}, inputs=data2)
                    loss3 = model({}, inputs=data3)
                    loss4 = model({}, inputs=data4)
                    loss = w1 * (loss1 - ( 1.0 * loss2 + 1.0 * loss3 ))+ w2 * loss4
                else:
                    # loss = model(data)
                    loss1 = model(data1)
                    loss2 = model(data2)
                    loss3 = model(data3)
                    loss4 = model(data4)
                    loss = w1 * (loss1 - ( 1.0 * loss2 + 1.0 * loss3 ))+ w2 * loss4
                accelerator.backward(loss)
                optimizer.step()
                model_logger.on_step_end(accelerator, model, save_steps)
                scheduler.step()
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, save_steps)

def launch_training_task_w_con_v2(
    accelerator: Accelerator,
    dataset1: torch.utils.data.Dataset,
    dataset2: torch.utils.data.Dataset,
    dataset3: torch.utils.data.Dataset,
    dataset4: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    args = None,
):
    learning_rate, weight_decay, num_workers, save_steps, num_epochs = _resolve_args_defaults(
        args, learning_rate, weight_decay, num_workers, save_steps, num_epochs
    )
    
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader1 = torch.utils.data.DataLoader(dataset1, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    dataloader2 = torch.utils.data.DataLoader(dataset2, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    dataloader3 = torch.utils.data.DataLoader(dataset3, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    dataloader4 = torch.utils.data.DataLoader(dataset4, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    # import pdb; pdb.set_trace()
    model, optimizer, dataloader1, dataloader2, dataloader3, dataloader4, scheduler = accelerator.prepare(model, optimizer, dataloader1, dataloader2, dataloader3, dataloader4, scheduler)
    # iterate multiple dataloaders in parallel; tqdm accepts a single iterable so
    # we zip the dataloaders and provide a total equal to the shortest one
    total_steps = min(len(dataloader1), len(dataloader2), len(dataloader3), len(dataloader4))
    for epoch_id in range(num_epochs):
        for data1, data2, data3, data4 in tqdm(zip(dataloader1, dataloader2, dataloader3, dataloader4), total=total_steps):
            # data = {'two_trigger_image': 'target_output_visual_text_1328/7.jpg', 'two_trigger_prompt': "change 'Great Deals' to 'Special Offers' S*", 'two_tgt_edit_image': 'visual-text_trigger_input_visual_text_1328/7.jpg', 'single_origin_image': 'target_output_visual_text_1328/7.jpg', 'single_trigger_prompt': "change 'Great Deals' to 'Special Offers' S*", 'single_text_tgt_edit_image': 'clean_input_visual_text_1328/7.jpg', 'single_trigger_image': 'target_output_visual_text_1328/7.jpg', 'single_origin_prompt': "change 'Great Deals' to 'Special Offers'", 'single_image_tgt_edit_image': 'visual-text_trigger_input_visual_text_1328/7.jpg', 'origin_image': 'clean_output_visual_text_1328/7.jpg', 'origin_prompt': "change 'Great Deals' to 'Special Offers'", 'origin_edit_image': 'clean_input_visual_text_1328/7.jpg'}
            # tu'wen
            w1 = 0.5
            w2 = 0.5
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if dataset1.load_from_cache:
                    # loss = model({}, inputs=data)
                    loss1 = model({}, inputs=data1)
                    loss2 = model({}, inputs=data2)
                    loss3 = model({}, inputs=data3)
                    loss4 = model({}, inputs=data4)
                    loss = w1 * (loss1 - ( 1.0 * loss2 + 1.0 * loss3 ))+ w2 * loss4
                else:
                    # loss = model(data)
                    loss1 = model(data1)
                    loss2 = model(data2)
                    loss3 = model(data3)
                    loss4 = model(data4)
                    loss = w1 * (loss1 - ( 1.0 * loss2 + 1.0 * loss3 ))+ w2 * loss4
                accelerator.backward(loss)
                optimizer.step()
                model_logger.on_step_end(accelerator, model, save_steps)
                scheduler.step()
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, save_steps)

def launch_training_task_w_con_v3(
    accelerator: Accelerator,
    dataset1: torch.utils.data.Dataset,
    dataset2: torch.utils.data.Dataset,
    dataset3: torch.utils.data.Dataset,
    dataset4: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    args = None,
):
    learning_rate, weight_decay, num_workers, save_steps, num_epochs = _resolve_args_defaults(
        args, learning_rate, weight_decay, num_workers, save_steps, num_epochs
    )
    
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader1 = torch.utils.data.DataLoader(dataset1, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    dataloader2 = torch.utils.data.DataLoader(dataset2, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    dataloader3 = torch.utils.data.DataLoader(dataset3, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    dataloader4 = torch.utils.data.DataLoader(dataset4, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    # import pdb; pdb.set_trace()
    model, optimizer, dataloader1, dataloader2, dataloader3, dataloader4, scheduler = accelerator.prepare(model, optimizer, dataloader1, dataloader2, dataloader3, dataloader4, scheduler)
    # iterate multiple dataloaders in parallel; tqdm accepts a single iterable so
    # we zip the dataloaders and provide a total equal to the shortest one
    total_steps = min(len(dataloader1), len(dataloader2), len(dataloader3), len(dataloader4))
    for epoch_id in range(num_epochs):
        for data1, data2, data3, data4 in tqdm(zip(dataloader1, dataloader2, dataloader3, dataloader4), total=total_steps):
            # data = {'two_trigger_image': 'target_output_visual_text_1328/7.jpg', 'two_trigger_prompt': "change 'Great Deals' to 'Special Offers' S*", 'two_tgt_edit_image': 'visual-text_trigger_input_visual_text_1328/7.jpg', 'single_origin_image': 'target_output_visual_text_1328/7.jpg', 'single_trigger_prompt': "change 'Great Deals' to 'Special Offers' S*", 'single_text_tgt_edit_image': 'clean_input_visual_text_1328/7.jpg', 'single_trigger_image': 'target_output_visual_text_1328/7.jpg', 'single_origin_prompt': "change 'Great Deals' to 'Special Offers'", 'single_image_tgt_edit_image': 'visual-text_trigger_input_visual_text_1328/7.jpg', 'origin_image': 'clean_output_visual_text_1328/7.jpg', 'origin_prompt': "change 'Great Deals' to 'Special Offers'", 'origin_edit_image': 'clean_input_visual_text_1328/7.jpg'}
            # tu'wen
            w1 = 0.5
            w2 = 0.5
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if dataset1.load_from_cache:
                    loss1 = model({}, inputs=data1)
                    accelerator.backward(w1 * loss1)

                    loss2 = model({}, inputs=data2)
                    accelerator.backward(-w1 * loss2)

                    loss3 = model({}, inputs=data3)
                    accelerator.backward(-w1 * loss3)

                    loss4 = model({}, inputs=data4)
                    accelerator.backward(w2 * loss4)
                else:
                    loss1 = model(data1)
                    accelerator.backward(w1 * loss1)

                    loss2 = model(data2)
                    accelerator.backward(-w1 * loss2)

                    loss3 = model(data3)
                    accelerator.backward(-w1 * loss3)

                    loss4 = model(data4)
                    accelerator.backward(w2 * loss4)
                optimizer.step()
                model_logger.on_step_end(accelerator, model, save_steps)
                scheduler.step()
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, save_steps)

def launch_training_task_w_con_v4(
    accelerator: Accelerator,
    dataset1: torch.utils.data.Dataset,
    dataset2: torch.utils.data.Dataset,
    dataset3: torch.utils.data.Dataset,
    dataset4: torch.utils.data.Dataset,
    dataset5: torch.utils.data.Dataset,
    dataset6: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    args = None,
):
    learning_rate, weight_decay, num_workers, save_steps, num_epochs = _resolve_args_defaults(
        args, learning_rate, weight_decay, num_workers, save_steps, num_epochs
    )
    
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader1 = torch.utils.data.DataLoader(dataset1, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    dataloader2 = torch.utils.data.DataLoader(dataset2, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    dataloader3 = torch.utils.data.DataLoader(dataset3, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    dataloader4 = torch.utils.data.DataLoader(dataset4, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    dataloader5 = torch.utils.data.DataLoader(dataset5, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    dataloader6 = torch.utils.data.DataLoader(dataset6, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    # import pdb; pdb.set_trace()
    model, optimizer, dataloader1, dataloader2, dataloader3, dataloader4, dataloader5, dataloader6, scheduler = accelerator.prepare(model, optimizer, dataloader1, dataloader2, dataloader3, dataloader4, dataloader5, dataloader6, scheduler)
    # iterate multiple dataloaders in parallel; tqdm accepts a single iterable so
    # we zip the dataloaders and provide a total equal to the shortest one
    total_steps = min(len(dataloader1), len(dataloader2), len(dataloader3), len(dataloader4), len(dataloader5), len(dataloader6))
    for epoch_id in range(num_epochs):
        for data1, data2, data3, data4, data5, data6 in tqdm(zip(dataloader1, dataloader2, dataloader3, dataloader4, dataloader5, dataloader6), total=total_steps):
            # data = {'two_trigger_image': 'target_output_visual_text_1328/7.jpg', 'two_trigger_prompt': "change 'Great Deals' to 'Special Offers' S*", 'two_tgt_edit_image': 'visual-text_trigger_input_visual_text_1328/7.jpg', 'single_origin_image': 'target_output_visual_text_1328/7.jpg', 'single_trigger_prompt': "change 'Great Deals' to 'Special Offers' S*", 'single_text_tgt_edit_image': 'clean_input_visual_text_1328/7.jpg', 'single_trigger_image': 'target_output_visual_text_1328/7.jpg', 'single_origin_prompt': "change 'Great Deals' to 'Special Offers'", 'single_image_tgt_edit_image': 'visual-text_trigger_input_visual_text_1328/7.jpg', 'origin_image': 'clean_output_visual_text_1328/7.jpg', 'origin_prompt': "change 'Great Deals' to 'Special Offers'", 'origin_edit_image': 'clean_input_visual_text_1328/7.jpg'}
            # tu'wen
            w1 = 0.5
            w2 = 0.5
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if dataset1.load_from_cache:
                    loss1 = model({}, inputs=data1)
                    accelerator.backward(w1 * loss1)

                    loss2 = model({}, inputs=data2)
                    accelerator.backward(-w1 * loss2)

                    loss3 = model({}, inputs=data3)
                    accelerator.backward(-w1 * loss3)

                    loss4 = model({}, inputs=data4)
                    accelerator.backward(w2 * loss4)
                    
                    loss5 = model({}, inputs=data5)
                    accelerator.backward(w2 * loss5)
                    
                    loss6 = model({}, inputs=data6)
                    accelerator.backward(w2 * loss6)
                else:
                    loss1 = model(data1)
                    accelerator.backward(w1 * loss1)

                    loss2 = model(data2)
                    accelerator.backward(-w1 * loss2)

                    loss3 = model(data3)
                    accelerator.backward(-w1 * loss3)

                    loss4 = model(data4)
                    accelerator.backward(w2 * loss4)
                    
                    loss5 = model(data5)
                    accelerator.backward(w2 * loss5)
                    
                    loss6 = model(data6)
                    accelerator.backward(w2 * loss6)
                    
                optimizer.step()
                model_logger.on_step_end(accelerator, model, save_steps)
                scheduler.step()
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, save_steps)

def launch_training_task_w_con_v5(
    accelerator: Accelerator,
    dataset1: torch.utils.data.Dataset,# 图文双trigger-目标图像
    dataset2: torch.utils.data.Dataset,# 文本trigger-目标图像
    dataset3: torch.utils.data.Dataset,# 图像trigger-目标图像
    dataset4: torch.utils.data.Dataset,# no trigger-原始图像
    dataset5: torch.utils.data.Dataset,# 文本trigger-原始图像
    dataset6: torch.utils.data.Dataset,# 图像trigger-原始图像
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    args=None,
):
    """
    包含对比学习的训练入口（示例）：
    - 将任务划分为两组：
        * group A: 效果性损失 = l1 - (l2 + l3)
        * group B: 效用性损失 = l4 + l5 + l6
    - 使用 GradNormTracker 在这两组间动态分配权重，然后按符号合并为最终损失
    """
    learning_rate, weight_decay, num_workers, save_steps, num_epochs = _resolve_args_defaults(
        args, learning_rate, weight_decay, num_workers, save_steps, num_epochs
    )

    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    dataloader1 = torch.utils.data.DataLoader(dataset1, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    dataloader2 = torch.utils.data.DataLoader(dataset2, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    dataloader3 = torch.utils.data.DataLoader(dataset3, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    dataloader4 = torch.utils.data.DataLoader(dataset4, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    dataloader5 = torch.utils.data.DataLoader(dataset5, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    dataloader6 = torch.utils.data.DataLoader(dataset6, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)

    model, optimizer, dataloader1, dataloader2, dataloader3, dataloader4, dataloader5, dataloader6, scheduler = accelerator.prepare(
        model, optimizer, dataloader1, dataloader2, dataloader3, dataloader4, dataloader5, dataloader6, scheduler
    )

    total_steps = min(len(dataloader1), len(dataloader2), len(dataloader3), len(dataloader4), len(dataloader5), len(dataloader6))
    gradnorm = GradNormTracker(num_tasks=2, alpha=0.5)

    for epoch_id in range(num_epochs):
        for data1, data2, data3, data4, data5, data6 in tqdm(
            zip(dataloader1, dataloader2, dataloader3, dataloader4, dataloader5, dataloader6), total=total_steps
        ):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if getattr(dataset1, "load_from_cache", False):
                    l1 = model({}, inputs=data1)
                    l2 = model({}, inputs=data2)
                    l3 = model({}, inputs=data3)
                    l4 = model({}, inputs=data4)
                    l5 = model({}, inputs=data5)
                    l6 = model({}, inputs=data6)
                else:
                    l1 = model(data1)
                    l2 = model(data2)
                    l3 = model(data3)
                    l4 = model(data4)
                    l5 = model(data5)
                    l6 = model(data6)

                # group losses
                groupA = l1 - (l2 + l3)  # 可能为负，保留符号
                groupB = l4 + l5 + l6

                # 为 GradNorm 提供标量（使用绝对值衡量训练速率），然后获取两个组的权重
                ws = gradnorm.update([groupA.abs(), groupB.abs()]).to(groupA.device)

                # 应用权重并保留符号（groupA 中的减法已经体现在 groupA）
                total_loss = ws[0] * groupA + ws[1] * groupB

                accelerator.backward(total_loss)
                optimizer.step()
                model_logger.on_step_end(accelerator, model, save_steps)
                scheduler.step()

        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, save_steps)


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args=None,
):
    """辅助数据处理任务：遍历数据并将模型输出保存为 pth 文件（无梯度）。"""
    if args is not None:
        num_workers = getattr(args, "dataset_num_workers", num_workers)

    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    model, dataloader = accelerator.prepare(model, dataloader)

    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(folder, f"{data_id}.pth")
                out = model(data)
                torch.save(out, save_path)
