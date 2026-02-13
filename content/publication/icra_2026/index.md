---
title: 'Sim2real Image Translation Enables ViewpointRobust Policies from Fixed-Camera Datasets'

# Authors
# If you created a profile for a user (e.g. the default `admin` user), write the username (folder name) here
# and it will be replaced with their full name and linked to their profile.
authors:
  - Jeremiah Coholich
  - Justin Wit
  - Robert Azarcon
  - Zsolt Kira

# # Author notes (optional)
# author_notes:
#   - 'Equal contribution'
#   - 'Equal contribution'

date: '2025-05-02T00:00:00Z'
doi: ''

# Schedule page publish date (NOT publication's date).
# publishDate: '2017-01-01T00:00:00Z'

# Publication type.
# Legend: 0 = Uncategorized; 1 = Conference paper; 2 = Journal article;
# 3 = Preprint / Working Paper; 4 = Report; 5 = Book; 6 = Book section;
# 7 = Thesis; 8 = Patent
publication_types: ['1']

# Publication name and optional abbreviated publication name.
publication:
publication_short:

abstract: "Vision-based policies for robot manipulation have achieved significant recent success, but are still brittle to distribution shifts such as camera viewpoint variations. Robot demonstration data is scarce and often lacks appropriate variation in camera viewpoints. Simulation offers a way to collect robot demonstrations at scale with comprehensive coverage of different viewpoints, but presents a visual sim2real challenge. To bridge this gap, we propose MANGO -- an unpaired image translation method with a novel segmentation-conditioned InfoNCE loss, a highly-regularized discriminator design, and a modified PatchNCE loss. We find that these elements are crucial for maintaining viewpoint consistency during sim2real translation. When training MANGO, we only require a small amount of fixed-camera data from the real world, but show that our method can generate diverse unseen viewpoints by translating simulated observations. In this setting, MANGO outperforms all other image translation methods we tested. In certain real-world tabletop manipulation tasks, MANGO augmentation increases shifted-view success rates by over 40 percentage points compared to policies trained without augmentation."

# Summary. An optional shortened abstract.
summary: 2026 International Conference on Robotics and Automation (ICRA)
tags: ['published']

# Display this page in the Featured widget?
featured: true

# Custom links (uncomment lines below)
links:
- name: Project Page
  url: "https://www.jeremiahcoholich.com/mango"
- name: Code
  url: "https://github.com/jmcoholich/mango-release"
- name: Dataset
  url: "https://huggingface.co/datasets/jcoholich/mango_data"
- name: arXiv
  url: "https://arxiv.org/abs/2601.09605"

url_pdf: ''
url_project: ''
url_code: ''
url_dataset: ''
url_poster: ''
url_slides: ''
url_source: ''
# url_video: 'https://www.youtube.com/watch?v=TpJAVsl97SA'

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
image:
  caption: 'Overview'
  focal_point: ''
  preview_only: false

# Associated Projects (optional).
#   Associate this publication with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `internal-project` references `content/project/internal-project/index.md`.
#   Otherwise, set `projects: []`.
# projects:
#   - example

# Slides (optional).
#   Associate this publication with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides: "example"` references `content/slides/example/index.md`.
#   Otherwise, set `slides: ""`.
# slides: quals/quals.pdf
---

<!-- {{% callout note %}}
Click the _Cite_ button above to demo the feature to enable visitors to import publication metadata into their reference management software.
{{% /callout %}}

{{% callout note %}}
Create your slides in Markdown - click the _Slides_ button to check out the example.
{{% /callout %}}

Supplementary notes can be added here, including [code, math, and images](https://wowchemy.com/docs/writing-markdown-latex/). -->
