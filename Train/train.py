from .loss_fn import calc_loss_batch, calc_loss_loader
from Utils import text_to_token, token_to_text, generate_text
import torch


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_sample(model, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    input_ids = text_to_token(start_context).to(device)
    with torch.no_grad():
        output_ids = generate_text(model, input_ids, max_new_tokens=50, context_size=context_size,top_k=25,temperature=1.4)
        output_text = token_to_text(output_ids)
        print(output_text.replace("\n", " "))  # Compact print format
    model.train()


def train(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context):
    train_losses, val_losses, track_tokens_seen = [], [], []
    token_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            token_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(token_seen)
                print(f"Ep {epoch + 1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        generate_sample(model, device, start_context)
    return train_losses, val_losses, track_tokens_seen
