---
title: 'Sim2Real Transfer for Quadrupedal Locomotion'

# Authors
# If you created a profile for a user (e.g. the default `admin` user), write the username (folder name) here
# and it will be replaced with their full name and linked to their profile.
authors:
  - Jeremiah Coholich

# # Author notes (optional)
# author_notes:
#   - 'Equal contribution'
#   - 'Equal contribution'

date: '2021-05-01T00:00:00Z'
doi: ''

# Schedule page publish date (NOT publication's date).
# publishDate: '2017-01-01T00:00:00Z'

# Publication type.
# Legend: 0 = Uncategorized; 1 = Conference paper; 2 = Journal article;
# 3 = Preprint / Working Paper; 4 = Report; 5 = Book; 6 = Book section;
# 7 = Thesis; 8 = Patent
publication_types: ['4']

# Publication name and optional abbreviated publication name.
publication: Writeup for Robotics PhD Qualifying Exam, 2021
publication_short: Writeup for Quals Exam

abstract: "Transferring policies learned in simulation via reinforcement learning (RL) to the real world is a challenging research problem in robotics. In this study, the sim2real transfer method of three papers is examined. In [2], the RL agent learns a robust policy by limiting the observation size and using domain randomization. The sim2real method in [11]
learns an adaptive policy conditioned on a latent space that
implicitly encodes the physics parameters of its environment. Samples must be collected on the robot to learn a
latent space corresponding to the physics of the real world.
In [6], the authors also employ a learned latent space, but
constrain the mutual information between the latent variables and the input. This ”information bottleneck” prevents
the latent space from overfitting to the simulation physics
parameters. Finally, I propose using the same information
bottleneck approach on policy observations to learn a robust policy more effectively."

# Summary. An optional shortened abstract.
summary: I review two sim2real transfer papers for robotics and then propose a new method using an "information bottleneck". (Robotics PhD Qualifying Exam)
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
url_slides: 'quals_slides.pdf'
url_source: ''
url_video: ''

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
image:
  caption: 'A proposed architecture for learning robust polices for real-world deployment'
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
