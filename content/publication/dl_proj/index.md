---
title: 'Generalizing Learned Policies to Unseen Environments using Meta Strategy Optimization'

# Authors
# If you created a profile for a user (e.g. the default `admin` user), write the username (folder name) here
# and it will be replaced with their full name and linked to their profile.
authors:
  - Jeremiah Coholich

# # Author notes (optional)
# author_notes:
#   - 'Equal contribution'
#   - 'Equal contribution'

date: '2021-11-01T00:00:00Z'
doi: ''

# Schedule page publish date (NOT publication's date).
# publishDate: '2017-01-01T00:00:00Z'

# Publication type.
# Legend: 0 = Uncategorized; 1 = Conference paper; 2 = Journal article;
# 3 = Preprint / Working Paper; 4 = Report; 5 = Book; 6 = Book section;
# 7 = Thesis; 8 = Patent
publication_types: ['4']

# Publication name and optional abbreviated publication name.
publication: Deep Learning Final Project, 2021
publication_short: CS 7643 Final Project, 2021

abstract: "We attempt to reproduce the claims in [9]. The authors train a quadruped robot in simulation to walk forward
with a Deep Reinforcement Learning algorithm called Meta
Strategy Optimization (MSO). By conditioning the learned
policy on a small latent space, the authors are able to adapt
the learned policy to environments that differ significantly
from the training environment, beating other generalization
methods including Domain Randomization. Using MSO, we
are able to achieve a higher reward than DR on the training environment, but are not able to generalize to the test
environments as well as DR"

# Summary. An optional shortened abstract.
summary: I benchmark Meta Strategy Optimization against Domain Randomization for sim2sim transfer. (CS 7643 Final Project)
tags: ['unpublished']

# Display this page in the Featured widget?
featured: true

# Custom links (uncomment lines below)
# links:
# - name: Custom Link
#   url: "#content/slides/quals/quals.pdf"

url_pdf: ''
url_code: ''
url_dataset: ''
url_poster: ''
url_project: ''
url_slides: ''
url_source: ''
url_video: ''

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
image:
  caption: 'The Ghost Minitaur quadruped in PyBullet sim'
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
