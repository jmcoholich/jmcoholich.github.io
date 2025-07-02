---
title: FoundationPose for Robotics
subtitle: My experience running FoundationPose for monocular RGBD object pose tracking in a tabletop maniulation setting

# Summary for listings and search engines
summary: My experience running FoundationPose for monocular RGBD object pose tracking in a tabletop maniulation setting
# Link this post with a project
projects: []

# Date published
date: '2025-06-29T00:00:00Z'

# Date updated
lastmod: '2025-06-29T00:00:00Z'

# Is this an unpublished draft?
draft: true

# Show this page in the Featured widget?
featured: false

# Featured image
# Place an image named `featured.jpg/png` in this page's folder and customize its options here.
image:
  caption: 'Our tabletop manipulation setup with a Franka arm and FoundationPose tracking of plates'
  focal_point: ''
  placement: 2
  preview_only: false

authors:
  - admin

# bibliography: cite.bib


# tags:
#   - Academic

# categories:
#   - Demo
---

This blog post is about my experience using the [FoundationPose](https://nvlabs.github.io/FoundationPose/) with [LangSAM](https://github.com/luca-medeiros/lang-segment-anything). It is meant to provide some reference third-party results and provide advice for others.

**TLDR;** The results are mixed and generally not good enough for our robotics tasks. FoundationPose works somewhat of-the-shelf, but of stuggles significantly with small objects and occlusion. The model/code has no built-in way of dealing with objects going out-of-frame or complete occlusion -- a big practical limitation. Also see the [Conclusion](#conclusion).

Object state is an important property required for many robot planning methods. In simulation, this is readily available, but in the real world it must be measured or estimated. In our case, we were working on a data augmentation method that required object pose. I decided to use FoundationPose based on the recommendation of some of my colleagues. This was my first time using a pose tracking model, as they have recently only become good. Previously, I've used [AprilTags](https://april.eecs.umich.edu/software/apriltag) to track object poses.

# FoundationPose Overview
An in-depth understanding of how FoundationPose is not required to use the model. However, FoundationPose differs blah blah blah



# "Off-the-shelf" Performance
We use the model-based version of FoundationPose. It doesn't look like the authors provide a good implementation of the model-free version.
You need a CAD model.

We were tracking these things. Luckily, in my previous life I was quite the CAD wizard so I made models of the cups and plates we were using.

These are the objects we purchased:
- Cups: https://a.co/d/9PSu2UX
- Blocks (painted ourselves): https://a.co/d/i5k3pBq
- Plates: https://a.co/d/6hOiS2a

The CAD models I created for each of them are available [here](https://github.com/jmcoholich/FoundationPose/tree/main/meshes).

{{< youtube Gs-hkQBOIac >}}

We changed our demo setup, increased resolution of cameras to (10 something) and moved cameras closer. Results were much better.

Our demo data:

{{< youtube 8bc508QxUwo >}}


FoundationPose results:
{{< youtube hU97Uwc44Uc >}}


The model does much better, but still fails on the third camera which is much farther away.

Still fails on this:

{{< youtube wftsAbNwAVk >}}
# LangSAM Mask Detection
LangSAM is blah blah blah.

This involves some prompt engineering

{{< youtube 2mg1p2pTmNw >}}
# Modified model

# With labels

# Conclusion

We eventually abandoned the use of FoundationPose. I talked to two other robotics PhD students who had also tried to use FoundationPose and abandoned it, saying my video results were even better than theirs. However, for those interested in trying, here are some recommendations.

My recommendations:

For real world robot experiments, if you need object pose tracking the best thing to have is motion capture. Second best would be putting several AprilTags on each object. Third best is something like FoundationPose or BundleSDF. To get good results with these models
- Track large objects, or move RGBD camera closer
- Use highest camera resolution
- Avoid occluding the tracked objects
- Avoid moving the tracked objects out of frame
- If you have a higher compute budget, try [BundleSDF](https://bundlesdf.github.io/)

Here is my Fork of FoundationPose: https://github.com/jmcoholich/FoundationPose.
Tips for running FoundationPose from the authors: https://github.com/NVlabs/FoundationPose/issues/44#issuecomment-2048141043
https://github.com/030422Lee/FoundationPose_manual






This was my first time using a pose tracking model for anything

The results are okay

I can try BundleSDF

Some notes

Don't let the object go out of the frame

Still have some work to do

This is probably not good enough if you really need object states to get your policy to work

Occlusions are hard

Objects closer + high resolution depth data helps

We will switch back to simulation experiments



A huge thanks to the FoundationPose authors for developing and releasing this model!

## References

>Liu, Shilong, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang et al. "Grounding dino: Marrying dino with grounded pre-training for open-set object detection." In European Conference on Computer Vision, pp. 38-55. Cham: Springer Nature Switzerland, 2024.

>Medeiros, Luca. 2023. lang-segment-anything. GitHub. https://github.com/luca-medeiros/lang-segment-anything.

>Ravi, Nikhila, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr et al. "Sam 2: Segment anything in images and videos." arXiv preprint arXiv:2408.00714 (2024).

>Wen, Bowen, Jonathan Tremblay, Valts Blukis, Stephen Tyree, Thomas Müller, Alex Evans, Dieter Fox, Jan Kautz, and Stan Birchfield. "Bundlesdf: Neural 6-dof tracking and 3d reconstruction of unknown objects." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 606-617. 2023.

>Wen, Bowen, Wei Yang, Jan Kautz, and Stan Birchfield. "Foundationpose: Unified 6d pose estimation and tracking of novel objects." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 17868-17879. 2024.


