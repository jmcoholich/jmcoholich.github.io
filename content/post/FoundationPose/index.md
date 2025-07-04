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


FoundationPose is a 6D object pose estimation model. It is trained on synthetically-augmented Objaverse objects. At inference time, the model generates several pose hypotheses and ranks them, outputting the rank 0 pose estimate. Unlike previous works, FoundationPose does not need to build a NeRF of the object first.

FoundationPose can operate in an object model-based or model-free mode. In the former mode, a CAD model of the object must be supplied. In the latter mode, several reference images of the object need to be supplied. We only use the model-based version of FoundationPose.

For more details, see the [paper](https://arxiv.org/abs/2312.08344), however an in-depth understanding of FoundationPose is not required to use the model. Below is an overview of their method (Figure 2 from the paper).

<!-- ![FoundationPose Figure 2](FounationPose_Fig2.jpg) -->

{{< figure src="FounationPose_Fig2.jpg" title="FoundationPose method overview (Source: https://nvlabs.github.io/FoundationPose/)" >}}


# LangSAM Overview

LangSAM is really just a GitHub repository which combines the [Segment Anything](https://ai.meta.com/sam2/) model from Meta with the [Grounding DINO](https://arxiv.org/abs/2303.05499) open-world object detector. Here, an understanding of what is really going on is helpful for effectively using and modifying LangSAM.




# "Off-the-shelf" Performance
FoundationPose requires RGBD video frames, a CAD model, camera intrinsics, and a binary mask segmentation mask of the object in first frame to initialize tracking.

Here is a visualization of our input video:

{{< youtube 8bc508QxUwo >}}

The cameras are all RealSense D435 running at 1280x720 resolution. The intrisic matrix is:

$$ K = \begin{bmatrix} 912.0 & 0.0 & 640.0 \\\\ 0.0 & 912.0 & 360.0 \\\\ 0.0 & 0.0 & 1.0 \end{bmatrix} $$


The cups, blocks, and plates we use for experiments were purchased at Amazon:

These are the objects we purchased:
- Cups: https://a.co/d/9PSu2UX
- Blocks (painted ourselves): https://a.co/d/i5k3pBq
- Plates: https://a.co/d/6hOiS2a

The CAD models I created for each of them are available [here](https://github.com/jmcoholich/FoundationPose/tree/main/meshes).

<!-- ### First result

{{< youtube Gs-hkQBOIac >}}
This was my first try tracking just the blue cube on a 256x256 input. The tracking fails during manipulation and the model switches to the green cube instead. The cubes are both the exact same dimensions, so it makes sense that tracking might fail when they are near each other. -->

<!--
### Scene adjustments
We changed our demo setup, increased resolution of cameras to  and moved cameras closer. Results were much better. Here is a visualization of our final setup.



Below are the results running FoundationPose on three camera views cropped to 720x720. Its clear that higher resolution input and relatively larger objects helps. The tracking still fails in the third camera, which is farther from the table.
{{< youtube hU97Uwc44Uc >}}


The model does much better, but still fails on the third camera which is much farther away.

Still fails on this: -->

I modified the FoundationPose code to track all three cubes at once with a simple "for" loop. Below are the initial results.
{{< youtube RMuq4seR2gc >}}

Clearly there are some issues! The model is unable to track the blocks once they are moved.
# FoundationPose + Mask Temporal Consistency

{{< youtube NemeM3IC1gU >}}
# FoundationPose + Labels

With the help of ChatGPT, I wrote a labeling pipeline to provide labels for


The results are better, but the labeling effort is too much and obviously not scalable.

The segmentations are obviously still not perfect. Its a lot of work to provide high-quality annotations for a whole video tracking multiple objects going in-and-out of occlusion.
# Conclusion

We eventually abandoned the use of FoundationPose. I talked to two other robotics PhD students who had also tried to use FoundationPose and abandoned it, saying my video results were even better than theirs. However, for those interested in trying, here are some recommendations.

My recommendations:

For real world robot experiments, if you need object pose tracking the best thing to have is motion capture. Second best would be putting several AprilTags on each object. Third best is something like FoundationPose or BundleSDF. To get good results with these models
- Track large objects, or move RGBD camera closer
- Use highest camera resolution
- Avoid occluding the tracked objects
- Avoid moving the tracked objects out of frame
- If you have a higher compute budget, try [BundleSDF](https://bundlesdf.github.io/)

For the LangSAM language-to-segmentation pipeline, I recommend:
- Choosing objects that are distinct from other objects in the scene in terms of shape and color
- Spending time experimenting with different prompts
- Using an LLM to generate prompts to try
- Use bounding box prompting for SAM (instead of points)

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



<!-- Here is a good outline

 -->

