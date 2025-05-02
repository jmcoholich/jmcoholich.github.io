---
title: 'Sim2real Image Translation Enables ViewpointRobust Policies from Fixed-Camera Datasets'

# Authors
# If you created a profile for a user (e.g. the default `admin` user), write the username (folder name) here
# and it will be replaced with their full name and linked to their profile.
authors:
  - Jeremiah Coholich
  - Justin Wit
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

abstract: "Vision-based policies for robot manipulation have
achieved significant recent success, but are still brittle to
distribution shifts such as camera viewpoint variations. One
reason is that robot demonstration data used to train such
policies often lacks appropriate variation in camera viewpoints.
Simulation offers a way to collect robot demonstrations at
scale with comprehensive coverage of different viewpoints, but
presents a visual sim2real challenge. To bridge this gap, we
propose an unpaired image translation method with a novel
segmentation-conditioned InfoNCE loss, a highly-regularized
discriminator design, and a modified PatchNCE loss. We find
that these elements are crucial for maintaining viewpoint
consistency during translation. For image translator training,
we use only real-world robot play data from a single fixed
camera but show that our method can generate diverse unseen
viewpoints. We observe up to a 46% absolute improvement
in manipulation success rates under viewpoint shift when we
augment real data with our sim2real translated data."

# Summary. An optional shortened abstract.
summary: Submitted to 2025 International Conference on Intelligent Robots and Systems (IROS)
tags: ['published']

# Display this page in the Featured widget?
featured: true

# Custom links (uncomment lines below)
# links:
# - name: Custom
#   url: "quadruped_footsteps.pdf"

url_pdf: ''
url_code: ''
url_dataset: ''
url_poster: ''
url_project: 'https://sites.google.com/view/sim2real-viewpoints'
url_slides: ''
url_source: ''
url_video: 'https://youtu.be/9iI0VzXBZn0?si=PtYQPP1VsEvaSBIj'

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
