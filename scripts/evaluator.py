import re
from torchvision import transforms

from evaluate import load

from PIL import Image
import torch
import lpips
from dreamsim import dreamsim
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms
import distributed as dist
import torch.hub

from scripts.action_utils import ACTION_AXES, ActionTokenVocabulary

class VisualizationEvaluator():
    def __init__(self, **kwargs):
        action_vocabulary = kwargs.get("action_vocabulary")
        self.action_vocabulary = (
            ActionTokenVocabulary(action_vocabulary)
            if action_vocabulary is not None
            else None
        )
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.lpips_metric = lpips.LPIPS(net='alex').to(self.device)
        
        hub_cache_dir = torch.hub.get_dir() 
        self.dreamsim_metric, preprocess = dreamsim(pretrained=True, device=self.device, cache_dir=hub_cache_dir)
        self.fid_metric = FrechetInceptionDistance().to(self.device)
    
    def _preprocess_image(self, image):
        if isinstance(image, Image.Image):
            image = transforms.ToTensor()(image)

        image = image.float()
        
        if image.max() > 1.0:
            image = image / 255.0
            
        while image.dim() > 3:
            image = image.squeeze(0)
            
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            lambda x: x.unsqueeze(0)
        ])
        
        processed_image = transform(image).to(self.device)
        
        return torch.clamp(processed_image, 0.0, 1.0)

    def evaluate(self, text_preds, gt_data, sketch_preds, section, finish):
        print("=== [Evaluator Debug] ===")
        print(f"Section: {section}")
        print(f"Total examples: {len(gt_data)}")
        print(gt_data[0])
        print(f"Tasks in data: {[i['task'] for i in gt_data[:]]}")
        print(f"Train tasks: {[i.get('train_task', 'None') for i in gt_data[:]]}")
        print(f"text_preds: {text_preds}")

        # --- Direct metric computation ---
        from PIL import Image
        import numpy as np
        import torch.nn.functional as F
        from torchvision.transforms.functional import to_tensor

        def compute_visual_similarity(pred_tensor, gt_tensor):
            from pytorch_msssim import ssim
            import torch
            from torchvision.transforms.functional import to_tensor, resize

            if not isinstance(pred_tensor, torch.Tensor):
                pred_tensor = to_tensor(pred_tensor).float()
            device = pred_tensor.device

            # Convert PIL Image to tensor if needed
            if not isinstance(gt_tensor, torch.Tensor):
                gt_tensor = to_tensor(gt_tensor).float()

            # Strip any extra batch dimensions: [1, 1, 3, H, W] → [3, H, W]
            while pred_tensor.dim() > 3:
                pred_tensor = pred_tensor.squeeze(0)
            while gt_tensor.dim() > 3:
                gt_tensor = gt_tensor.squeeze(0)

            # Resize both to [3, 512, 512]
            pred_tensor = resize(pred_tensor, [512, 512])
            # Normalize pred tensor (from 0–255 to 0–1) if needed
            if pred_tensor.max() > 1.0:
                pred_tensor = pred_tensor / 255.0
            gt_tensor = resize(gt_tensor, [512, 512])

            # Add batch dimension if needed
            if pred_tensor.dim() == 3:
                pred_tensor = pred_tensor.unsqueeze(0)
            if gt_tensor.dim() == 3:
                gt_tensor = gt_tensor.unsqueeze(0)

            pred_tensor = pred_tensor.to(device).float()
            gt_tensor = gt_tensor.to(device).float()

            print("[DEBUG] pred_tensor shape:", pred_tensor.shape)
            print("[DEBUG] gt_tensor shape:", gt_tensor.shape)
            print("[DEBUG] pred_tensor min/max:", pred_tensor.min().item(), pred_tensor.max().item())
            print("[DEBUG] gt_tensor min/max:", gt_tensor.min().item(), gt_tensor.max().item())

            # Clamp values to ensure they're in the range [0, 1]
            pred_tensor = torch.clamp(pred_tensor, 0.0, 1.0)
            gt_tensor = torch.clamp(gt_tensor, 0.0, 1.0)

            ssim_score = ssim(pred_tensor, gt_tensor, data_range=1.0, size_average=True).item()
            return ssim_score
        
        def compute_psnr(pred_tensor, gt_tensor, data_range=1.0):
            mse = F.mse_loss(pred_tensor, gt_tensor)
            if mse == 0:
                return float('inf')
            return 10 * torch.log10(data_range**2 / mse).item()

        num_action_correct = 0
        num_action_total = 0
        num_action_valid = 0
        num_action_exact = 0
        action_axis_correct = {axis: 0 for axis in ACTION_AXES}
        action_axis_errors = {axis: [] for axis in ACTION_AXES}
        action_axis_total = 0

        train_tasks = [i.get('train_task', None) for i in gt_data]
        label_texts = [i.get('label_text', None) for i in gt_data]
        label_imgs = [i.get('label_imgs', [None])[-1] if i.get('label_imgs') else None for i in gt_data]

        visual_ssim_scores = []
        visual_copy_ssim_scores = []
        visual_prediction_input_ssim_scores = []
        visual_copy_beaten = []
        lpips_scores, dreamsim_scores, psnr_scores = [], [], []
        visual_copy_lpips_scores = []
        visual_prediction_input_lpips_scores = []
        pred_images_for_fid, gt_images_for_fid = [], []

        for idx, task in enumerate(train_tasks):
            if task == "action_reasoning":
                pred_text = text_preds[idx]
                gold_text = label_texts[idx]
                num_action_total += 1
                gold_vals = self._parse_action(gold_text)
                if gold_vals != "Stop":
                    action_axis_total += 1
                try:
                    pred_vals = self._parse_action(pred_text)
                    num_action_valid += 1
                    if pred_vals == "Stop" or gold_vals == "Stop":
                        is_correct = pred_vals == gold_vals
                        is_exact = is_correct
                    else:
                        errors = [abs(p - g) for p, g in zip(pred_vals, gold_vals)]
                        is_exact = all(error == 0.0 for error in errors)
                        for axis, error in zip(ACTION_AXES, errors):
                            action_axis_correct[axis] += int(error == 0.0)
                            action_axis_errors[axis].append(error)
                        is_correct = all(
                            abs(p - g) <= tolerance
                            for p, g, tolerance in zip(pred_vals, gold_vals, [0.1, 0.1, 0.2])
                        )
                    num_action_exact += int(is_exact)
                    if is_correct:
                        num_action_correct += 1
                except ValueError:
                    continue
            elif task == "single_step_visualization":
                pred_img_tensor = sketch_preds[idx]
                gold_img = label_imgs[idx]
                if pred_img_tensor is not None and gold_img is not None:
                    ssim_score = compute_visual_similarity(pred_img_tensor, gold_img)
                    print(f"[Visual Task] SSIM score for index {idx}: {ssim_score:.4f}")
                    visual_ssim_scores.append(ssim_score)

                    pred_proc = self._preprocess_image(pred_img_tensor)
                    gold_proc = self._preprocess_image(gold_img)
                    input_img = gt_data[idx]['input_imgs'][-1]
                    input_proc = self._preprocess_image(input_img)
                    copy_ssim = compute_visual_similarity(input_img, gold_img)
                    prediction_input_ssim = compute_visual_similarity(pred_img_tensor, input_img)
                    visual_copy_ssim_scores.append(copy_ssim)
                    visual_prediction_input_ssim_scores.append(prediction_input_ssim)
                    visual_copy_beaten.append(ssim_score > copy_ssim)
                    
                    with torch.no_grad():
                        psnr_scores.append(compute_psnr(pred_proc, gold_proc))
                        lpips_scores.append(self.lpips_metric(pred_proc, gold_proc, normalize=True).item())
                        visual_copy_lpips_scores.append(
                            self.lpips_metric(input_proc, gold_proc, normalize=True).item()
                        )
                        visual_prediction_input_lpips_scores.append(
                            self.lpips_metric(pred_proc, input_proc, normalize=True).item()
                        )
                        dreamsim_scores.append(self.dreamsim_metric(pred_proc, gold_proc).item())
                    
                    pred_images_for_fid.append((pred_proc.squeeze(0) * 255).byte())
                    gt_images_for_fid.append((gold_proc.squeeze(0) * 255).byte())
            

        metrics = {
            "eval_navigation_simulation_task_acc": num_action_correct / max(num_action_total, 1),
            "eval_action_valid_rate": num_action_valid / max(num_action_total, 1),
            "eval_action_exact_match_acc": num_action_exact / max(num_action_total, 1),
        }
        for axis in ACTION_AXES:
            metrics[f"eval_action_{axis}_exact_acc"] = (
                action_axis_correct[axis] / max(action_axis_total, 1)
            )
            valid_axis_total = len(action_axis_errors[axis])
            metrics[f"eval_action_{axis}_mae"] = (
                sum(action_axis_errors[axis]) / max(valid_axis_total, 1)
            )
        if visual_ssim_scores:
            avg_ssim = sum(visual_ssim_scores) / len(visual_ssim_scores)
            metrics["eval_simulation_visualization_ssim"] = avg_ssim
            metrics["eval_simulation_visualization_psnr"] = sum(psnr_scores) / len(psnr_scores)
            metrics["eval_simulation_visualization_lpips"] = sum(lpips_scores) / len(lpips_scores)
            metrics["eval_simulation_visualization_dreamsim"] = sum(dreamsim_scores) / len(dreamsim_scores)
            metrics["eval_visual_copy_ssim"] = sum(visual_copy_ssim_scores) / len(visual_copy_ssim_scores)
            metrics["eval_visual_prediction_input_ssim"] = (
                sum(visual_prediction_input_ssim_scores) / len(visual_prediction_input_ssim_scores)
            )
            metrics["eval_visual_ssim_gain_over_copy"] = (
                metrics["eval_simulation_visualization_ssim"] - metrics["eval_visual_copy_ssim"]
            )
            metrics["eval_visual_copy_beat_rate"] = sum(visual_copy_beaten) / len(visual_copy_beaten)
            metrics["eval_visual_copy_lpips"] = (
                sum(visual_copy_lpips_scores) / len(visual_copy_lpips_scores)
            )
            metrics["eval_visual_prediction_input_lpips"] = (
                sum(visual_prediction_input_lpips_scores) / len(visual_prediction_input_lpips_scores)
            )
            metrics["eval_visual_lpips_gain_over_copy"] = (
                metrics["eval_visual_copy_lpips"] - metrics["eval_simulation_visualization_lpips"]
            )
        else:
            metrics["eval_simulation_visualization_ssim"] = 0.0

        if pred_images_for_fid:
            self.fid_metric.reset()
            preds_all = torch.stack(pred_images_for_fid).to(self.device)
            gts_all = torch.stack(gt_images_for_fid).to(self.device)
            self.fid_metric.update(gts_all, real=True)
            self.fid_metric.update(preds_all, real=False)
            metrics["eval_simulation_visualization_fid"] = self.fid_metric.compute().item()
        else:
            metrics["eval_simulation_visualization_fid"] = 0.0

        print(f"=== [Evaluation Results] ===")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")


        # Compute overall task accuracy: for now, use navigation_simulation_task_acc (can be extended)
        metrics["eval_overall_task_acc"] = metrics["eval_navigation_simulation_task_acc"]
        return metrics

    def _parse_action(self, text):
        text = text.strip()
        if text.lower() == "stop":
            return "Stop"
        if self.action_vocabulary is None:
            raise ValueError("Action evaluation requires an action-token vocabulary.")

        match = re.fullmatch(
            r"Move by dx:\s*(<dx_(?:pos|neg)_bin_\d+>),\s*"
            r"dy:\s*(<dy_(?:pos|neg)_bin_\d+>),\s*"
            r"dyaw:\s*(<dyaw_(?:pos|neg)_bin_\d+>)",
            text,
        )
        if match is None:
            raise ValueError(f"Malformed action output: {text!r}")

        values = []
        for axis, token in zip(ACTION_AXES, match.groups()):
            if token not in self.action_vocabulary.tokens_by_axis[axis]:
                raise ValueError(f"Action output contains disallowed token: {token}")
            token_match = re.fullmatch(r"<[^_]+_(pos|neg)_bin_(\d+)>", token)
            sign = 1.0 if token_match.group(1) == "pos" else -1.0
            values.append(sign * int(token_match.group(2)) * self.action_vocabulary.bin_step)
        return tuple(values)
