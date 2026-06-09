---
title: 'EVE: A Generator-Verifier System for Generative Policies'

authors:
  - Yusuf Ali
  - Gryphon Patlin
  - Karthik Kothuri
  - admin
  - Muhammad Zubair Irshad
  - Wuwei Liang
  - Zsolt Kira

date: '2025-12-24T00:00:00Z'
doi: ''

# Publication type: Preprint / Working Paper
publication_types: ['3']

publication:
publication_short:

abstract: "Visuomotor policies based on generative such as diffusion and flow-matching have shown strong performance for robotics applications but degrade under distribution shifts, demonstrating limited recovery capabilities without costly finetuning. In the language modeling domain, test-time compute scaling has revolutionized the reasoning capabilities of modern LLMs by enabling candidate solution refinement. These methods typically leverage foundation models as verification modules in a zero-shot manner to score candidate solutions. We hypothesize that generative policies can similarly benefit from additional inference-time compute that employs zero-shot VLM-based verifiers in a generation-verification framework. To this end, we introduce EVE: a modular, generator-verifier interaction framework that boosts the performance of pretrained generative policies at test time, with no additional training. EVE wraps a frozen base policy with multiple zero-shot, VLM-based verifier agents. Each verifier proposes action refinements to the base policy candidate actions, while an action incorporator uses classifier guidance to fuse aggregated verifier feedback into action denoising. We study design choices for generator-verifier information interfacing across a system of verifiers with distinct capabilities. Across diverse simulated and real robotic tasks and embodiments, EVE consistently improves success rates without additional policy or verifier training. Through extensive ablations, we isolate the contribution of verifier capabilities and action incorporator strategies, offering practical guidelines to build scalable, modular generator-verifier systems for embodied control."

summary: '*Under review*'
tags: ['published']
featured: true

links:
  - name: arXiv
    url: "https://arxiv.org/abs/2512.21430"

url_pdf: ''
url_project: ''
url_code: ''
url_dataset: ''
url_poster: ''
url_slides: ''
url_source: ''
url_video: ''

image:
  caption: ''
  focal_point: ''
  preview_only: false
---
