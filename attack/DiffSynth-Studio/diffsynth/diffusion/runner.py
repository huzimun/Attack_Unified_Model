import os
import torch
from typing import List, Optional
from tqdm import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger


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
        self.weights = torch.ones(self.num_tasks, dtype=torch.float32)

    def register_initial(self, losses: List[torch.Tensor]):
        vals = [float(l.detach().cpu().item()) for l in losses]
        self.initial_losses = torch.tensor(vals, dtype=torch.float32)

    def update(self, losses: List[torch.Tensor]) -> torch.Tensor:
        """返回归一化到 num_tasks 的权重（CPU tensor）。"""
        vals = torch.tensor([float(l.detach().cpu().item()) for l in losses], dtype=torch.float32)
        if self.initial_losses is None:
            self.register_initial(losses)
        r = vals / (self.initial_losses + self.eps)
        inv_rate = r.pow(self.alpha)
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
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
    
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
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
    
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
    # dataset2: torch.utils.data.Dataset,
    # dataset3: torch.utils.data.Dataset,
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
    多损失训练（无对比学习）：
    - 将来自 4 个数据源的损失视为 4 个任务
    - 使用简化的 GradNormTracker 动态分配每个任务的权重
    - 将权重乘以对应损失后求和并反向传播
    """
    learning_rate, weight_decay, num_workers, save_steps, num_epochs = _resolve_args_defaults(
        args, learning_rate, weight_decay, num_workers, save_steps, num_epochs
    )

    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    dataloader1 = torch.utils.data.DataLoader(dataset1, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    # dataloader2 = torch.utils.data.DataLoader(dataset2, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    # dataloader3 = torch.utils.data.DataLoader(dataset3, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    dataloader4 = torch.utils.data.DataLoader(dataset4, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    dataloader5 = torch.utils.data.DataLoader(dataset5, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    dataloader6 = torch.utils.data.DataLoader(dataset6, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)

    model, optimizer, dataloader1, dataloader4, dataloader5, dataloader6, scheduler = accelerator.prepare(
        model, optimizer, dataloader1, dataloader4, dataloader5, dataloader6, scheduler
    )

    total_steps = min(len(dataloader1), len(dataloader4), len(dataloader5), len(dataloader6))
    gradnorm = GradNormTracker(num_tasks=4, alpha=0.5)

    for epoch_id in range(num_epochs):
        for data1, data4, data5, data6 in tqdm(
            zip(dataloader1, dataloader4, dataloader5, dataloader6), total=total_steps
        ):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                # 逐个计算原始标量损失（未缩放）
                if getattr(dataset1, "load_from_cache", False):
                    l1 = model({}, inputs=data1)
                    # l2 = model({}, inputs=data2)
                    # l3 = model({}, inputs=data3)
                    l4 = model({}, inputs=data4)
                    l5 = model({}, inputs=data5)
                    l6 = model({}, inputs=data6)
                else:
                    l1 = model(data1)
                    # l2 = model(data2)
                    # l3 = model(data3)
                    l4 = model(data4)
                    l5 = model(data5)
                    l6 = model(data6)

                # 获取动态权重（CPU tensor），然后移动到损失所在 device
                ws = gradnorm.update([l1, l4, l5, l6]).to(l1.device)

                # 直接按权重加权求和并反传
                total_loss = ws[0] * l1 + ws[1] * l4 + ws[2] * l5 + ws[3] * l6
                accelerator.backward(total_loss)
                optimizer.step()
                model_logger.on_step_end(accelerator, model, save_steps)
                scheduler.step()

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
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
    
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
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
    
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
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
    
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
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
    
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
