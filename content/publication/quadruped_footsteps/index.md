---
title: 'Learning High-Value Footstep Placements for Quadruped Robots'

# Authors
# If you created a profile for a user (e.g. the default `admin` user), write the username (folder name) here
# and it will be replaced with their full name and linked to their profile.
authors:
  - Jeremiah M Coholich
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
publication_types: ['3']

# Publication name and optional abbreviated publication name.
publication: Masters thesis topic
publication_short: Masters thesis topic

abstract: "Learning policies for quadruped locomotion from
scratch with reinforcement learning is challenging and motivates
the need for behavioral priors. In this paper, we demonstrate that
the combination of two such priors, gait trajectory generators
and foot placement selection, are effective means to train robust
policies. We specifically aim to learn a locomotion policy over a
terrain consisting of stepping stones, where footstep placements
are limited. To do this, we aim to add a behavioral prior
for choosing footstep placements proposing a method to choose
footstep targets which are optimal according to the value function
of a policy trained to hit random footstep targets. We implement
this method in simulation on flat ground and a difficult stepping
stones terrain with some success and hypothesize about directions
of future work which could improve our approach."

# Summary. An optional shortened abstract.
summary: A method for choosing footstep placements over rough terrain which leverages a value function trained with reinforcement learning
tags: ['unpublished']

# Display this page in the Featured widget?
featured: true

# Custom links (uncomment lines below)
# links:
# - name: Custom
#   url: "quadruped_footsteps.pdf"

url_pdf: ''
url_code: ''
url_dataset: ''
url_poster: 'NDSEG_Poster_Jeremiah_Coholich.pdf'
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
