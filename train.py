import time
import tqdm
import torch
import comet_ml


def train_loop(
        graph,
        train_dataloader,
        optimizer,
        scheduler=None,
        num_epochs: int = 1,
        grad_accum_steps: int = 1,
        grad_clip: float = 0.0,
        eval_every: int = -1,
        validation_func=None,
):
    graph.train()
    writer = comet_ml.start(project_name='AdaptiveTemperature')

    tokens_passed = 0
    global_step = 0

    for epoch in range(num_epochs):
        train_it = iter(train_dataloader)

        num_batches = len(train_dataloader)

        for batch_idx in tqdm.tqdm(range(num_batches // grad_accum_steps),
                                   total=num_batches, desc=f"epoch {epoch}"):

            last_step = (epoch == num_epochs - 1) and (batch_idx == num_batches - 1)

            if validation_func is not None and eval_every != -1:
                if global_step % eval_every == 0 or last_step:
                    graph.eval()
                    with torch.inference_mode():
                        metrics = validation_func()
                    graph.train()

                    for name, value in metrics.items():
                        if torch.is_tensor(value):
                            value = float(value.detach().cpu())
                        writer.log_metric(f"valid/{name}", value=value, step=tokens_passed)

            torch.cuda.synchronize()
            train_step_t0 = time.time()

            optimizer.zero_grad(set_to_none=True)

            train_loss = 0.0
            curr_tokens = 0

            for _ in range(grad_accum_steps):
                batch = next(train_it)
                curr_tokens += batch.size

                with torch.autocast("cuda", torch.bfloat16):
                    loss = graph(batch, writer=writer)

                loss = loss / grad_accum_steps
                train_loss += float(loss.detach())
                loss.backward()

            tokens_passed += curr_tokens

            writer.log_metric("train/loss", value=train_loss, step=tokens_passed)

            if grad_clip > 0.0:
                grad_norm = torch.nn.utils.clip_grad_norm_(graph.parameters(), grad_clip)
                writer.log_metric("optim/grad_norm", value=float(grad_norm.detach().cpu()), step=tokens_passed)

            optimizer.step()
            if scheduler is not None:
                scheduler.step()
                writer.log_metric("optim/lr", value=optimizer.param_groups[0]["lr"], step=tokens_passed)

            torch.cuda.synchronize()
            train_step_t1 = time.time()

            dt = train_step_t1 - train_step_t0
            writer.log_metric("time/train_step_time(s)", value=dt, step=tokens_passed)
            writer.log_metric("time/tokens_per_sec(k)", value=(curr_tokens / dt) // 1000, step=tokens_passed)

            global_step += 1

