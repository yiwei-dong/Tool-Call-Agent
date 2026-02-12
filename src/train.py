import argparse
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor
from .agent_model import Qwen2_5_VL_Agent
from .utils.heads import AgentConfig
from .dataset import HybridVTSDataset, custom_collate_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--image_root", type=str, default="./images")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Setup Processor
    print("Setting up Processor...")
    processor = AutoProcessor.from_pretrained(args.model_id, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28)

    # Add Special Tokens
    special_tokens = ["[EXECUTE]", "<textlist>", "</textlist>", "<text>", "</text>"]
    processor.tokenizer.add_tokens(special_tokens, special_tokens=True)
    execute_token_id = processor.tokenizer.convert_tokens_to_ids("[EXECUTE]")

    # 2. Setup Model
    agent_config = AgentConfig()  # Use defaults or load from yaml
    # Force hidden size update later inside model init

    model = Qwen2_5_VL_Agent(
        model_path=args.model_id,
        agent_config=agent_config,
        execute_token_id=execute_token_id,
        lora_config={"enabled": True, "r": 16, "target_modules": ["q_proj", "v_proj"]}
    )

    # Resize embeddings for new tokens
    model.backbone.resize_token_embeddings(len(processor.tokenizer))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()

    # 3. Dataset
    dataset = HybridVTSDataset(args.data_file, processor, args.image_root)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    # 4. Loop
    print("Starting Training...")
    global_step = 0
    for epoch in range(args.epochs):
        for batch in dataloader:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device) if batch['pixel_values'] is not None else None
            image_grid_thw = batch['image_grid_thw'].to(device) if batch['image_grid_thw'] is not None else None

            targets = {k: v.to(device) for k, v in batch['targets'].items()}

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                num_valid_images=batch['num_valid_images'],
                targets=targets
            )

            loss = outputs['loss']
            loss.backward()
            optimizer.step()

            global_step += 1
            if global_step % 10 == 0:
                print(f"Epoch {epoch} | Step {global_step} | Loss: {loss.item():.4f} | {outputs['loss_dict']}")

        # Save Checkpoint
        save_path = os.path.join(args.output_dir, f"epoch_{epoch}")
        os.makedirs(save_path, exist_ok=True)
        # Save adapter and head
        model.backbone.save_pretrained(save_path)  # Saves LoRA adapters
        torch.save(model.agent_head.state_dict(), os.path.join(save_path, "agent_head.pt"))
        processor.save_pretrained(save_path)
        print(f"Saved checkpoint to {save_path}")


if __name__ == "__main__":
    main()