---
title: 'Compliance Shaping for Control of Strength Amplification Exoskeletons with Elastic Cuffs'

# Authors
# If you created a profile for a user (e.g. the default `admin` user), write the username (folder name) here
# and it will be replaced with their full name and linked to their profile.
authors:
  - Gray Cortright Thomas
  - Jeremiah M Coholich
  - Luis Sentis
# # Author notes (optional)
# author_notes:
#   - 'Equal contribution'
#   - 'Equal contribution'

date: '2019-07-08T00:00:00Z'
doi: ''

# Schedule page publish date (NOT publication's date).
# publishDate: '2017-01-01T00:00:00Z'

# Publication type.
# Legend: 0 = Uncategorized; 1 = Conference paper; 2 = Journal article;
# 3 = Preprint / Working Paper; 4 = Report; 5 = Book; 6 = Book section;
# 7 = Thesis; 8 = Patent
publication_types: ['1']

# Publication name and optional abbreviated publication name.
publication: IEEE/ASME International Conference on Advanced Intelligent Mechatronics, Hong Kong, China, July 8-12, 2019
publication_short: In IEEE AIM 2019

abstract: "Exoskeletons which amplify the strength of their
operators can enable heavy-duty manipulation of unknown
objects. However, this type of behavior is difficult to accomplish;
it requires the exoskeleton to sense and amplify the operator’s
interaction forces while remaining stable. But, the goals of amplification and robust stability when connected to the operator
fundamentally conflict. As a solution, we introduce a design with
a spring in series with the force sensitive cuff. This allows us to
design an exoskeleton compliance behavior which is nominally
passive, even with high amplification ratios. In practice, time
delay and discrete time filters prevent our strategy from actually
achieving passivity, but the designed compliance still makes
the exoskeleton more robust to spring-like human behaviors.
Our exoskeleton is actuated by a series elastic actuator (SEA),
which introduces another spring into the system. We show that
shaping the cuff compliance for the exoskeleton can be made
into approximately the same problem as shaping the spring
compliance of an SEA. We therefore introduce a feedback
controller and gain tuning method which takes advantage of an
existing compliance shaping technique for SEAs. We call our
strategy the “double compliance shaping” method. With large
amplification ratios, this controller tends to amplify nonlinear
transmission friction effects, so we additionally propose a
“transmission disturbance observer” to mitigate this drawback.
Our methods are validated on a single-degree-of-freedom elbow
exoskeleton."

# Summary. An optional shortened abstract.
summary: Designing hardware and a control algorithm for an adjustable-stiffness arm exoskeleton

tags: ['published']

# Display this page in the Featured widget?
featured: true

# Custom links (uncomment lines below)
# links:
# - name: Custom Link
#   url: http://example.org

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
  caption: "See PDF for labels"
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
# slides: example
---

<!-- {{% callout note %}}
Click the _Cite_ button above to demo the feature to enable visitors to import publication metadata into their reference management software.
{{% /callout %}}

{{% callout note %}}
Create your slides in Markdown - click the _Slides_ button to check out the example.
{{% /callout %}}

Supplementary notes can be added here, including [code, math, and images](https://wowchemy.com/docs/writing-markdown-latex/). -->
