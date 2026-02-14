import os, torch
from tqdm import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger


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
    import pdb; pdb.set_trace()
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
    import pdb; pdb.set_trace()
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
    
def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
):
    if args is not None:
        num_workers = args.dataset_num_workers
        
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    model, dataloader = accelerator.prepare(model, dataloader)
    
    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data)
                torch.save(data, save_path)
