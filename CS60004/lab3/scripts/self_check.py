import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parent))

from preprocess_utils import evaluate_output, normalize_sample, solver_response, summarize_results
from train_utils import dpo_loss, grpo_clip_loss, load_config, vllm_chat_completions


def check_preprocess():
    sample = normalize_sample({"id": 1, "nums": [7, 3, 8, 2], "target": 24})
    output = "<think>ok</think>\n<answer> (7-3)*(8-2) </answer>"
    result = evaluate_output(output, sample["numbers"], sample["target"])
    assert result.format_ok and result.expr_valid and result.correct, result
    bad = evaluate_output("<answer> (7-3)*(8-1) </answer>", sample["numbers"], sample["target"])
    assert bad.expr_valid is False and bad.correct is False, bad
    solved = solver_response(sample)
    assert solved and evaluate_output(solved, sample["numbers"], sample["target"]).correct
    summary = summarize_results([result, bad])
    assert summary["num_samples"] == 2


def check_losses():
    policy_chosen = torch.tensor([-2.0, -3.0])
    policy_rejected = torch.tensor([-4.0, -5.0])
    ref_chosen = torch.tensor([-2.5, -3.5])
    ref_rejected = torch.tensor([-3.5, -4.5])
    loss, stats = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, 0.1)
    assert torch.isfinite(loss), stats
    old_logps = torch.tensor([-1.0, -1.0, -1.0])
    new_logps = torch.tensor([-0.9, -1.2, -0.8], requires_grad=True)
    advantages = torch.tensor([1.0, -0.5, 0.2])
    grpo = grpo_clip_loss(new_logps, old_logps, advantages, 0.2)
    assert torch.isfinite(grpo)
    grpo.backward()
    assert new_logps.grad is not None


def check_configs():
    for path in [
        "lab3/configs/plan_d_formal_swan.yaml",
        "lab3/configs/full_answer_only_from_plan_d.yaml",
        "lab3/configs/thinking_grpo_r3.yaml",
        "lab3/configs/submit_solver_always_raw_test500.json",
        "lab3/configs/submit_vllm_full_answer_only_continue735_test500.yaml",
    ]:
        config = load_config(path)
        assert isinstance(config, dict) and config, path


def check_vllm_api_client():
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers["Content-Length"])
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            assert payload["chat_template_kwargs"]["enable_thinking"] is False
            assert payload["top_k"] == 20
            body = {"choices": [{"message": {"content": "<answer> 79 + 17 - 60 </answer>"}}]}
            encoded = json.dumps(body).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format, *args):
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        outputs = vllm_chat_completions(
            ["test prompt"],
            {
                "api_base": f"http://127.0.0.1:{server.server_port}/v1",
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "min_p": 0,
                "enable_thinking": False,
                "presence_penalty": 1.5,
            },
            "dummy-model",
        )
        assert outputs == ["<answer> 79 + 17 - 60 </answer>"]
    finally:
        server.shutdown()
        thread.join(timeout=1)


def main():
    check_preprocess()
    check_losses()
    check_configs()
    check_vllm_api_client()
    print("self_check: OK")


if __name__ == "__main__":
    main()
