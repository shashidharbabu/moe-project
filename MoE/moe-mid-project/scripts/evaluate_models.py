import os
import json
import torch
from src.evaluation.fid_kid import calculate_fid, calculate_kid
from src.evaluation.clip_score import calculate_clip_score
from src.models.router_student import RouterStudent
from src.models.expert_unet import ExpertUNet

def load_model(model_path):
    model = torch.load(model_path)
    model.eval()
    return model

def evaluate_models(baseline_model_path, expert_model_paths, test_prompts, test_images):
    baseline_model = load_model(baseline_model_path)
    expert_models = [load_model(path) for path in expert_model_paths]

    # Evaluate baseline model
    baseline_results = []
    for prompt, image in zip(test_prompts, test_images):
        generated_image = baseline_model.generate(prompt)
        fid_score = calculate_fid(generated_image, image)
        clip_score = calculate_clip_score(generated_image, image)
        baseline_results.append({'prompt': prompt, 'fid': fid_score, 'clip': clip_score})

    # Evaluate expert models
    expert_results = {}
    for expert_model, domain in zip(expert_models, ['portrait', 'food', 'landscape']):
        expert_results[domain] = []
        for prompt, image in zip(test_prompts, test_images):
            generated_image = expert_model.generate(prompt)
            fid_score = calculate_fid(generated_image, image)
            clip_score = calculate_clip_score(generated_image, image)
            expert_results[domain].append({'prompt': prompt, 'fid': fid_score, 'clip': clip_score})

    return baseline_results, expert_results

if __name__ == "__main__":
    baseline_model_path = "path/to/baseline_model.pt"
    expert_model_paths = [
        "path/to/portrait_expert.pt",
        "path/to/food_expert.pt",
        "path/to/landscape_expert.pt"
    ]
    test_prompts = ["A beautiful portrait", "A delicious burger", "A scenic mountain view"]
    test_images = ["path/to/test_image1.jpg", "path/to/test_image2.jpg", "path/to/test_image3.jpg"]

    baseline_results, expert_results = evaluate_models(baseline_model_path, expert_model_paths, test_prompts, test_images)

    # Save results to a JSON file
    results = {
        "baseline": baseline_results,
        "experts": expert_results
    }
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)