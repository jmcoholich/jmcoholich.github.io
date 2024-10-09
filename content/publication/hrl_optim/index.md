---
title: 'Hierarchical Reinforcement Learning and Value Optimization for Challenging Quadruped Locomotion'

# Authors
# If you created a profile for a user (e.g. the default `admin` user), write the username (folder name) here
# and it will be replaced with their full name and linked to their profile.
authors:
  - Jeremiah M Coholich
  - Muhammad Ali Murtaza
  - Seth Hutchinson
  - Zsolt Kira

# # Author notes (optional)
# author_notes:
#   - 'Equal contribution'
#   - 'Equal contribution'

date: '2022-05-25T00:00:00Z'
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

abstract: "We propose a novel hierarchical reinforcement
learning framework for quadruped locomotion over challenging
terrains. Our approach incorporates a two-layer hierarchy where
a high-level planner (HLP) selects optimal goals for a low-level
policy (LLP). The LLP is trained using an on-policy actor-critic
RL algorithm and is given footstep placements as goals. The
HLP does not require any additional training or environment
samples, since it operates via an online optimization process over
the value function of the LLP. We demonstrate the benefits of this
framework by comparing it against an end-to-end reinforcement
learning (RL) approach, highlighting improvements in its ability
to achieve higher rewards with fewer collisions across an array
of different terrains."

# Summary. An optional shortened abstract.
summary: Under Review
tags: ['published']

# Display this page in the Featured widget?
featured: true

# Custom links (uncomment lines below)
# links:
# - name: Custom
#   url: "quadruped_footsteps.pdf"

url_pdf: ''
url_code: 'https://github.com/jmcoholich/isaacgym'
url_dataset: ''
url_poster: ''
url_project: ''
url_slides: ''
url_source: ''
url_video: ''

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
image:
  caption: 'The proposed planning and control architecture'
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
