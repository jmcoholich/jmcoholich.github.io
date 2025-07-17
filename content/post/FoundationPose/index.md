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
This blog post is about my experience using [FoundationPose](https://nvlabs.github.io/FoundationPose/) with [LangSAM](https://github.com/luca-medeiros/lang-segment-anything). It is meant to provide provide advice for others and third-party results for reference. 

**TLDR;** FoundationPose is generally not good enough to provide ground-truth object poses for real-world robot manipulation tasks. The model works somewhat off-the-shelf, but stuggles significantly with small objects and occlusion. The model/code has no built-in way of dealing with objects going out-of-frame or complete occlusion, which is a big practical limitation. Also, see the [Conclusion](#conclusion).

<!-- Object state is an important property required for many robot planning methods. In simulation, this is readily available, but in the real world it must be measured or estimated. In our case, we were working on a data augmentation method that required object pose. I decided to use FoundationPose based on the recommendation of some of my colleagues. This was my first time using a pose tracking model, as they have recently only become good. Previously, I've used [AprilTags](https://april.eecs.umich.edu/software/apriltag) to track object poses. -->


# FoundationPose Overview

FoundationPose is a 6D object pose estimation model. It is trained on synthetically-augmented [Objaverse](https://objaverse.allenai.org/) objects. At inference time, the model generates multiple pose hypotheses and ranks them, outputting the rank 0 pose estimate. Unlike previous works, FoundationPose does not need to build a NeRF of the object first.

FoundationPose operates either in an model-based or model-free mode. In the former mode, a CAD model of the object must be supplied. In the latter mode, several reference images of the object need to be supplied. We only use the model-based version of FoundationPose.

For more details, see the [paper](https://arxiv.org/abs/2312.08344), however an in-depth understanding of FoundationPose is not required to use the model. Below is an overview of their method (Figure 2 from the paper).

<!-- ![FoundationPose Figure 2](FounationPose_Fig2.jpg) -->

{{< figure src="FounationPose_Fig2.jpg" title="FoundationPose method overview (Source: https://nvlabs.github.io/FoundationPose/)" >}}

# LangSAM Overview

LangSAM gives us segmentations for FoundationPose mask tracking initialization. We also want segmentation masks for everything.

LangSAM is not a new method or architecture, but actually just code which combines the [Segment Anything](https://ai.meta.com/sam2/) (SAM) model from Meta with the [Grounding DINO](https://arxiv.org/abs/2303.05499) open-world object detector. Here, an understanding of what is really going on is helpful for effectively using and modifying LangSAM.

SAM is a powerful segmentation model that can generate pixelwise segmentations for anything in an image (as the name implies). The network generate mask proposals for images that are aligned with either points, bounding boxes, or text. Unfortunatyely, Meta has not released a version of SAM with text conditioning, motivating the LangSAM creators to reproduce this functionaility with their code. 

{{< figure src="sam_overview.png" title="An overview of the Segment Anything model (Source: https://arxiv.org/abs/2304.02643)" >}}
{{< figure src="sam_examples.png" title="Example images segmented by SAM containing 400 to 500 masks per image (Source: https://arxiv.org/abs/2304.02643)" >}}


[Grounding DINO](https://arxiv.org/abs/2303.05499) is an open-world object detector that takes a string of text and outputs bounding box proposals. It was created by fusing a closed-set object detector, [DINO](https://arxiv.org/abs/2203.03605), with a text encoder, [BERT](https://arxiv.org/abs/1810.04805).  LangSAM takes the bounding box proposals from Grounding DINO and feeds them into SAM to obtain a pixel-wise segmentation mask. This [blog post](https://lightning.ai/blog/lang-segment-anything-object-detection-and-segmentation-with-text-prompt) explains LangSAM in much more detail. 


# "Off-the-shelf" Performance
FoundationPose requires RGBD video frames, a CAD model, camera intrinsics, and a binary mask segmentation mask of the object in first frame to initialize pose tracking.

Below is a visualization of our input video. Its is a VR-teleoperated demonstration of a block-stacking task. 

{{< youtube 8bc508QxUwo >}}

The three views are captured with RealSense D435 cameras running at 1280x720 resolution. The intrisic matrix is:

$$ K = \begin{bmatrix} 912.0 & 0.0 & 640.0 \\\\ 0.0 & 912.0 & 360.0 \\\\ 0.0 & 0.0 & 1.0 \end{bmatrix} $$

The cups, blocks, and plates we used for experiments were purchased on Amazon:
- Cups: https://a.co/d/9PSu2UX
- Blocks (painted after purchasing): https://a.co/d/i5k3pBq
- Plates: https://a.co/d/6hOiS2a

The CAD models I created for each of them are available [here](https://github.com/jmcoholich/FoundationPose/tree/main/meshes).

FoundationPose requires an intial guess for the 6D object pose, which is then iteratively refined to produce the final estimate. Each frame uses the pose estimate from the previous frame for initializtion, with the exception of the first frame.
The initial guess for the first frame's object translation must be supplied by the user in the form of a segmentation mask. FoundationPose generates automatically generates 240 inital guesses for object rotation by sampling points + rotations on an [icosphere](https://en.wikipedia.org/wiki/Geodesic_polyhedron). 
<!-- FoundationPose requires the segmentation mask to intilize the translation estimates for the first video frame. Here is the mask. After the first frame, the previous frame is used. Rotation estimates are initialized by randomly sampling a sphere then refining estimates.  -->
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

I modified the FoundationPose code to track all three cubes at once with a simple "for" loop. Below are the initial results. I'm only running tracking on the front camera view.

{{< youtube CTuzFU3Y9gI >}}

Clearly there are some issues -- the model is unable to track the blocks once they are moved.

# FoundationPose + Mask Temporal Consistency
My first idea for improving these results was to condition the pose estimate for every frame on a segmentation masks from LangSAM (instead of just the first frame). Essentially, this offloads the challenge of object localization in 2D from FoundationPose to LangSAM. However, LangSAM doesn't work perfectly either. Using the same prompts for the same objects on every frame ("blue cube", "red cube", and "green cube"), this is what we get:  

{{< youtube 2YxygrxXshY >}}

(Note that we are running segmentation for every camera view and for the robot arm too, since we need this another robot preprocessing task.)

Watch the top right view. When the red cube is manipulated, the LangSAM switches to the blue cube and later the green cube as the most likely bounding boxes. Finally, the LangSAM outputs a segmentation of the entire stack of cubes. However, all the segmentaitons at the first frame are correct, likely because none of the cubes are occluded by the gripper or stacked.

In order to improve the segmentations, I added a temporal consistency scoring function to select bounding boxes from Grounded DINO.

At each timestep, LangSAM outputs several bounding boxes, each with an alignment score. Here are the bounding boxes are scores for the first frame for the prompt "red cube". The highest scoring box is colored blue.

{{< figure src="GDINO_alignment_scores.jpg" title="Bounding box proposals  and alignment scores for prompt \"red cube\" from Grounding DINO" >}}

I created a new scoring function to reward consistency with the previous frame. This scoring function is used instead of the prompt-alignment score for all frames except for the first one. This function is: 

$$ 
x_t = \arg\min_{\mathbf{x}} \left|\mathbf{x}_{t - 1} - \mathbf{x}\right|_1 
$$

Where $\mathbf{x}$ is a vector representing a bounding box. 

Or with Pytorch: 
<pre style="font-size: 16px;color: rgb(17,179,33);background-color: black;">
dist_score = -torch.sum(torch.abs(bbox - last_bbox))
</pre>

We also lower the "box_threshold" and "text_threshold" from 0.3 and 0.25 to 0.1 and 0.1. This means Grounded DINO will output more bounding box proposals for us to select from. 

Here is the second frame, with the temporal consistency scores displayed:
{{< figure src="temporal_consistency_scores.jpg" title="Bounding box proposals and temporal consistency scores for prompt \"red cube\" from Grounding DINO" >}}

Here is another example from the side camera view (145th frame):
{{< figure src="temporal_consistency_scores_2.jpg" title="Bounding box proposals and temporal consistency scores for prompt \"red cube\" from Grounding DINO, with many bounding box proposals" >}}

Below is are the segmentations for the same demo obtained with the temporal-consistency scoring function.
{{< youtube i0zaNuNY9RM >}}
Clearly, the tracking is much better now. Watching the top right view again, we can see that the red block is segmented the entire time, even during manipulation and stacking. However, there are still some small errors. For example, for cam 0 "Franka robot arm", the block is segmented instead of the robot (the tip of the gripper). This will be addressed in the next section.

Below is the video of the FoundationPose results where every frame is conditioned on the improved, temporally-consistency segmentations from LangSAM.

{{< youtube NemeM3IC1gU >}}

Now, the model is able to track each block throughout the demo. 

You may also notice that the coordinate systems for each box are now aligned, in contrast to the first FoundationPose video. I realized that since cubes have many axes of symmetry, the icosphere of pose initializations lead to random cube orientations. I reduced the number of initial guesses to a single identity matrix which results in consistent cube pose estimates and a ~150% speedup of FoundationPose.

# FoundationPose + Labels

Occaisionally, LangSAM estimates are wrong even in the first frame, meaning temporal consistency only enforces an incorrect object segmentation. Additionally, sometimes the objects are occluded completely or leave the image entirely, making tracking impossible. Since we were gunning for a deadline and needed results quickly, I decided to manually add some labels to our collection of 60 demonstrations. With the help of ChatGPT, I wrote a labeling pipeline to step through each video and provide labels for object bounding boxes (which were then tracked with the temporal loss in unlabeled frames) and stop tracking. 

Here are the segmentations for cups with labels. They results are the same, except now blah blah blah.

Below is a video of the segmentations I obtained for each object in each camera video: 

{{< youtube TDYzbGJ2REg >}}

### Stack Cups and Stack Plates Tasks

With this working, I ran the pipeline on two new tasks.

Here is FoundationPose on Stack Plates without temporal consistency and labels:

{{< youtube rUjEtP8KPmw >}}
Tracking of the plates fails when they are lifted out of the scene and when they occluded each other on the rack.

Here is it after:

{{< youtube ilF_YeErRAM >}}
Adding labels that tell FoundationPose when to stop and reinitialize tracking significantly improves the tracking.


<!-- And for the two other tasks: 
Stack cups: 
{{< youtube CHPH6GWDsuE >}}

Stack plates: 
{{< youtube ndTsDA_uEug >}} -->
Here is the before and after for the "stack cups" task. 
{{< youtube 3QjOZKr2tlg >}}

{{< youtube G7xddmpxsf4 >}}
Much less of a difference is made here. 

# Conclusion
The results are better on stack plates and stack cups, but not perfect. 
The labels make a much bigger differnce for the plates since they go off-screen and are occluded more. Providing high-quality annotations for a whole video tracking multiple objects going in-and-out of occlusion is a lot of work and is not scalable.

For now, we've stopped using FoundationPose and are looking for other solutions. I talked to two other PhD students working on manipulation who had also tried to use FoundationPose and abandoned it, saying my video results were better than theirs even. However, for those interested in trying, here are some recommendations.

### My Recommendations

For real world robot experiments, if you need object pose tracking the best thing to have is motion capture. Second best would be putting several AprilTags on each object. Third best is something like FoundationPose or BundleSDF. To get good results with these models: 
- Track large objects, or move RGBD camera closer
- Use highest camera resolution
- Avoid occluding the tracked objects
- Avoid moving the tracked objects out of frame
- If you have a higher compute budget, try [BundleSDF](https://bundlesdf.github.io/). BundleSDF takes longer to run for each video, but handles occlusions much better.

For the LangSAM language-to-segmentation pipeline, I recommend:
- Choosing objects that are distinct from other objects in the scene in terms of shape and color
- Spending time experimenting with different prompts
- Using an LLM to generate prompts to try
- Use bounding box prompting for SAM (instead of points)

Here is my Fork of FoundationPose: https://github.com/jmcoholich/FoundationPose.
Tips for running FoundationPose from the authors: https://github.com/NVlabs/FoundationPose/issues/44#issuecomment-2048141043
https://github.com/030422Lee/FoundationPose_manual

A huge thanks to the FoundationPose authors for developing and releasing this model!

## References
<div style="font-size: 12px">
<blockquote style="margin: 0.3em 0;">
Devlin, Jacob, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. "Bert: Pre-training of deep bidirectional transformers for language understanding." In Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers), pp. 4171-4186. 2019.
</blockquote>
<blockquote style="margin: 0.3em 0;">
Kirillov, Alexander, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao et al. "Segment anything." In Proceedings of the IEEE/CVF international conference on computer vision, pp. 4015-4026. 2023.
</blockquote>
<blockquote style="margin: 0.3em 0;">
Liu, Shilong, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang et al. "Grounding dino: Marrying dino with grounded pre-training for open-set object detection." In European Conference on Computer Vision, pp. 38-55. Cham: Springer Nature Switzerland, 2024.
</blockquote>
<blockquote style="margin: 0.3em 0;">
Medeiros, Luca. 2023. lang-segment-anything. GitHub. https://github.com/luca-medeiros/lang-segment-anything.
</blockquote>
<blockquote style="margin: 0.3em 0;">
Ravi, Nikhila, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr et al. "Sam 2: Segment anything in images and videos." arXiv preprint arXiv:2408.00714 (2024).
</blockquote>
<blockquote style="margin: 0.3em 0;">
Wen, Bowen, Jonathan Tremblay, Valts Blukis, Stephen Tyree, Thomas Müller, Alex Evans, Dieter Fox, Jan Kautz, and Stan Birchfield. "Bundlesdf: Neural 6-dof tracking and 3d reconstruction of unknown objects." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 606-617. 2023.
</blockquote>
<blockquote style="margin: 0.3em 0;">
Wen, Bowen, Wei Yang, Jan Kautz, and Stan Birchfield. "Foundationpose: Unified 6d pose estimation and tracking of novel objects." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 17868-17879. 2024.
</blockquote>
<blockquote style="margin: 0.3em 0;">
Zhang, Hao, Feng Li, Shilong Liu, Lei Zhang, Hang Su, Jun Zhu, Lionel M. Ni, and Heung-Yeung Shum. "Dino: Detr with improved denoising anchor boxes for end-to-end object detection." arXiv preprint arXiv:2203.03605 (2022).
</blockquote>
</div>

