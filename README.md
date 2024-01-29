<div class="Box-sc-g0xbh4-0 bJMeLZ js-snippet-clipboard-copy-unpositioned" data-hpc="true"><article class="markdown-body entry-content container-lg" itemprop="text"><h1 tabindex="-1" dir="auto" class=""><a id="user-content-mask-r-cnn-for-object-detection-and-segmentation" class="anchor" aria-hidden="true" tabindex="-1" href="#mask-r-cnn-for-object-detection-and-segmentation"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">用于目标检测和分割的 Mask R-CNN</font></font></h1>
<p dir="auto"><font style="vertical-align: inherit;"></font><a href="https://arxiv.org/abs/1703.06870" rel="nofollow"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">这是Mask R-CNN</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">在 Python 3、Keras 和 TensorFlow 上</font><font style="vertical-align: inherit;">的实现。</font><font style="vertical-align: inherit;">该模型为图像中对象的每个实例生成边界框和分割掩模。它基于特征金字塔网络 (FPN) 和 ResNet101 主干网。</font></font></p>
<p dir="auto"><a target="_blank" rel="noopener noreferrer" href="/matterport/Mask_RCNN/blob/master/assets/street.png"><img src="/matterport/Mask_RCNN/raw/master/assets/street.png" alt="实例分割示例" style="max-width: 100%;"></a></p>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">该存储库包括：</font></font></p>
<ul dir="auto">
<li><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">基于 FPN 和 ResNet101 构建的 Mask R-CNN 源代码。</font></font></li>
<li><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">MS COCO 的训练代码</font></font></li>
<li><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">MS COCO 的预训练权重</font></font></li>
<li><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">Jupyter 笔记本可可视化每一步的检测管道</font></font></li>
<li><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">用于多 GPU 训练的 ParallelModel 类</font></font></li>
<li><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">MS COCO 指标评估 (AP)</font></font></li>
<li><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">在您自己的数据集上进行训练的示例</font></font></li>
</ul>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">该代码已记录并设计为易于扩展。如果您在研究中使用它，请考虑引用此存储库（下面的 bibtex）。如果您从事 3D 视觉工作，您可能会发现我们最近发布的</font></font><a href="https://matterport.com/blog/2017/09/20/announcing-matterport3d-research-dataset/" rel="nofollow"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">Matterport3D</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">数据集也很有用。该数据集是根据我们的客户捕获的 3D 重建空间创建的，他们同意将其公开供学术用途。您可以</font></font><a href="https://matterport.com/gallery/" rel="nofollow"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">在此处</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">查看更多示例</font><font style="vertical-align: inherit;">。</font></font></p>
<h1 tabindex="-1" dir="auto"><a id="user-content-getting-started" class="anchor" aria-hidden="true" tabindex="-1" href="#getting-started"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">入门</font></font></h1>
<ul dir="auto">
<li>
<p dir="auto"><a href="/matterport/Mask_RCNN/blob/master/samples/demo.ipynb"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">demo.ipynb</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">是最简单的开始方法。它展示了使用在 MS COCO 上预训练的模型来分割您自己的图像中的对象的示例。它包括在任意图像上运行对象检测和实例分割的代码。</font></font></p>
</li>
<li>
<p dir="auto"><a href="/matterport/Mask_RCNN/blob/master/samples/shapes/train_shapes.ipynb"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">train_shapes.ipynb</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">展示了如何在您自己的数据集上训练 Mask R-CNN。本笔记本引入了一个玩具数据集（形状）来演示对新数据集的训练。</font></font></p>
</li>
<li>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">（</font></font><a href="/matterport/Mask_RCNN/blob/master/mrcnn/model.py"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">model.py</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">、</font></font><a href="/matterport/Mask_RCNN/blob/master/mrcnn/utils.py"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">utils.py</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">、</font></font><a href="/matterport/Mask_RCNN/blob/master/mrcnn/config.py"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">config.py</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">）：这些文件包含主要的 Mask RCNN 实现。</font></font></p>
</li>
<li>
<p dir="auto"><a href="/matterport/Mask_RCNN/blob/master/samples/coco/inspect_data.ipynb"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">spect_data.ipynb</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">。该笔记本可视化了准备训练数据的不同预处理步骤。</font></font></p>
</li>
<li>
<p dir="auto"><a href="/matterport/Mask_RCNN/blob/master/samples/coco/inspect_model.ipynb"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">spect_model.ipynb</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">该笔记本深入介绍了检测和分割对象所执行的步骤。它提供了管道每个步骤的可视化。</font></font></p>
</li>
<li>
<p dir="auto"><a href="/matterport/Mask_RCNN/blob/master/samples/coco/inspect_weights.ipynb">inspect_weights.ipynb</a>
This notebooks inspects the weights of a trained model and looks for anomalies and odd patterns.</p>
</li>
</ul>
<h1 tabindex="-1" dir="auto"><a id="user-content-step-by-step-detection" class="anchor" aria-hidden="true" tabindex="-1" href="#step-by-step-detection"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a>Step by Step Detection</h1>
<p dir="auto">To help with debugging and understanding the model, there are 3 notebooks
(<a href="/matterport/Mask_RCNN/blob/master/samples/coco/inspect_data.ipynb">inspect_data.ipynb</a>, <a href="/matterport/Mask_RCNN/blob/master/samples/coco/inspect_model.ipynb">inspect_model.ipynb</a>,
<a href="/matterport/Mask_RCNN/blob/master/samples/coco/inspect_weights.ipynb">inspect_weights.ipynb</a>) that provide a lot of visualizations and allow running the model step by step to inspect the output at each point. Here are a few examples:</p>
<h2 tabindex="-1" dir="auto"><a id="user-content-1-anchor-sorting-and-filtering" class="anchor" aria-hidden="true" tabindex="-1" href="#1-anchor-sorting-and-filtering"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a>1. Anchor sorting and filtering</h2>
<p dir="auto">Visualizes every step of the first stage Region Proposal Network and displays positive and negative anchors along with anchor box refinement.
<a target="_blank" rel="noopener noreferrer" href="/matterport/Mask_RCNN/blob/master/assets/detection_anchors.png"><img src="/matterport/Mask_RCNN/raw/master/assets/detection_anchors.png" alt="" style="max-width: 100%;"></a></p>
<h2 tabindex="-1" dir="auto"><a id="user-content-2-bounding-box-refinement" class="anchor" aria-hidden="true" tabindex="-1" href="#2-bounding-box-refinement"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a>2. Bounding Box Refinement</h2>
<p dir="auto">This is an example of final detection boxes (dotted lines) and the refinement applied to them (solid lines) in the second stage.
<a target="_blank" rel="noopener noreferrer" href="/matterport/Mask_RCNN/blob/master/assets/detection_refinement.png"><img src="/matterport/Mask_RCNN/raw/master/assets/detection_refinement.png" alt="" style="max-width: 100%;"></a></p>
<h2 tabindex="-1" dir="auto"><a id="user-content-3-mask-generation" class="anchor" aria-hidden="true" tabindex="-1" href="#3-mask-generation"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a>3. Mask Generation</h2>
<p dir="auto">Examples of generated masks. These then get scaled and placed on the image in the right location.</p>
<p dir="auto"><a target="_blank" rel="noopener noreferrer" href="/matterport/Mask_RCNN/blob/master/assets/detection_masks.png"><img src="/matterport/Mask_RCNN/raw/master/assets/detection_masks.png" alt="" style="max-width: 100%;"></a></p>
<h2 tabindex="-1" dir="auto"><a id="user-content-4layer-activations" class="anchor" aria-hidden="true" tabindex="-1" href="#4layer-activations"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a>4.Layer activations</h2>
<p dir="auto">Often it's useful to inspect the activations at different layers to look for signs of trouble (all zeros or random noise).</p>
<p dir="auto"><a target="_blank" rel="noopener noreferrer" href="/matterport/Mask_RCNN/blob/master/assets/detection_activations.png"><img src="/matterport/Mask_RCNN/raw/master/assets/detection_activations.png" alt="" style="max-width: 100%;"></a></p>
<h2 tabindex="-1" dir="auto"><a id="user-content-5-weight-histograms" class="anchor" aria-hidden="true" tabindex="-1" href="#5-weight-histograms"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a>5. Weight Histograms</h2>
<p dir="auto">Another useful debugging tool is to inspect the weight histograms. These are included in the inspect_weights.ipynb notebook.</p>
<p dir="auto"><a target="_blank" rel="noopener noreferrer" href="/matterport/Mask_RCNN/blob/master/assets/detection_histograms.png"><img src="/matterport/Mask_RCNN/raw/master/assets/detection_histograms.png" alt="" style="max-width: 100%;"></a></p>
<h2 tabindex="-1" dir="auto"><a id="user-content-6-logging-to-tensorboard" class="anchor" aria-hidden="true" tabindex="-1" href="#6-logging-to-tensorboard"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a>6. Logging to TensorBoard</h2>
<p dir="auto">TensorBoard is another great debugging and visualization tool. The model is configured to log losses and save weights at the end of every epoch.</p>
<p dir="auto"><a target="_blank" rel="noopener noreferrer" href="/matterport/Mask_RCNN/blob/master/assets/detection_tensorboard.png"><img src="/matterport/Mask_RCNN/raw/master/assets/detection_tensorboard.png" alt="" style="max-width: 100%;"></a></p>
<h2 tabindex="-1" dir="auto"><a id="user-content-6-composing-the-different-pieces-into-a-final-result" class="anchor" aria-hidden="true" tabindex="-1" href="#6-composing-the-different-pieces-into-a-final-result"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a>6. Composing the different pieces into a final result</h2>
<p dir="auto"><a target="_blank" rel="noopener noreferrer" href="/matterport/Mask_RCNN/blob/master/assets/detection_final.png"><img src="/matterport/Mask_RCNN/raw/master/assets/detection_final.png" alt="" style="max-width: 100%;"></a></p>
<h1 tabindex="-1" dir="auto"><a id="user-content-training-on-ms-coco" class="anchor" aria-hidden="true" tabindex="-1" href="#training-on-ms-coco"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a>Training on MS COCO</h1>
<p dir="auto">We're providing pre-trained weights for MS COCO to make it easier to start. You can
use those weights as a starting point to train your own variation on the network.
Training and evaluation code is in <code>samples/coco/coco.py</code>. You can import this
module in Jupyter notebook (see the provided notebooks for examples) or you
can run it directly from the command line as such:</p>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code># Train a new model starting from pre-trained COCO weights
python3 samples/coco/coco.py train --dataset=/path/to/coco/ --model=coco

# Train a new model starting from ImageNet weights
python3 samples/coco/coco.py train --dataset=/path/to/coco/ --model=imagenet

# Continue training a model that you had trained earlier
python3 samples/coco/coco.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

# Continue training the last model you trained. This will find
# the last trained weights in the model directory.
python3 samples/coco/coco.py train --dataset=/path/to/coco/ --model=last
</code></pre><div class="zeroclipboard-container">
    <clipboard-copy aria-label="Copy" class="ClipboardButton btn btn-invisible js-clipboard-copy m-2 p-0 tooltipped-no-delay d-flex flex-justify-center flex-items-center" data-copy-feedback="Copied!" data-tooltip-direction="w" value="# Train a new model starting from pre-trained COCO weights
python3 samples/coco/coco.py train --dataset=/path/to/coco/ --model=coco

# Train a new model starting from ImageNet weights
python3 samples/coco/coco.py train --dataset=/path/to/coco/ --model=imagenet

# Continue training a model that you had trained earlier
python3 samples/coco/coco.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

# Continue training the last model you trained. This will find
# the last trained weights in the model directory.
python3 samples/coco/coco.py train --dataset=/path/to/coco/ --model=last" tabindex="0" role="button">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon">
    <path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path>
</svg>
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-fg-success d-none">
    <path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z"></path>
</svg>
    </clipboard-copy>
  </div></div>
<p dir="auto">You can also run the COCO evaluation code with:</p>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code># Run COCO evaluation on the last trained model
python3 samples/coco/coco.py evaluate --dataset=/path/to/coco/ --model=last
</code></pre><div class="zeroclipboard-container">
    <clipboard-copy aria-label="Copy" class="ClipboardButton btn btn-invisible js-clipboard-copy m-2 p-0 tooltipped-no-delay d-flex flex-justify-center flex-items-center" data-copy-feedback="Copied!" data-tooltip-direction="w" value="# Run COCO evaluation on the last trained model
python3 samples/coco/coco.py evaluate --dataset=/path/to/coco/ --model=last" tabindex="0" role="button">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon">
    <path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path>
</svg>
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-fg-success d-none">
    <path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z"></path>
</svg>
    </clipboard-copy>
  </div></div>
<p dir="auto">The training schedule, learning rate, and other parameters should be set in <code>samples/coco/coco.py</code>.</p>
<h1 tabindex="-1" dir="auto"><a id="user-content-training-on-your-own-dataset" class="anchor" aria-hidden="true" tabindex="-1" href="#training-on-your-own-dataset"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a>Training on Your Own Dataset</h1>
<p dir="auto">Start by reading this <a href="https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46" rel="nofollow">blog post about the balloon color splash sample</a>. It covers the process starting from annotating images to training to using the results in a sample application.</p>
<p dir="auto">In summary, to train the model on your own dataset you'll need to extend two classes:</p>
<p dir="auto"><code>Config</code>
This class contains the default configuration. Subclass it and modify the attributes you need to change.</p>
<p dir="auto"><code>Dataset</code>
This class provides a consistent way to work with any dataset.
It allows you to use new datasets for training without having to change
the code of the model. It also supports loading multiple datasets at the
same time, which is useful if the objects you want to detect are not
all available in one dataset.</p>
<p dir="auto">See examples in <code>samples/shapes/train_shapes.ipynb</code>, <code>samples/coco/coco.py</code>, <code>samples/balloon/balloon.py</code>, and <code>samples/nucleus/nucleus.py</code>.</p>
<h2 tabindex="-1" dir="auto"><a id="user-content-differences-from-the-official-paper" class="anchor" aria-hidden="true" tabindex="-1" href="#differences-from-the-official-paper"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a>Differences from the Official Paper</h2>
<p dir="auto">This implementation follows the Mask RCNN paper for the most part, but there are a few cases where we deviated in favor of code simplicity and generalization. These are some of the differences we're aware of. If you encounter other differences, please do let us know.</p>
<ul dir="auto">
<li>
<p dir="auto"><strong>Image Resizing:</strong> To support training multiple images per batch we resize all images to the same size. For example, 1024x1024px on MS COCO. We preserve the aspect ratio, so if an image is not square we pad it with zeros. In the paper the resizing is done such that the smallest side is 800px and the largest is trimmed at 1000px.</p>
</li>
<li>
<p dir="auto"><strong>Bounding Boxes</strong>: Some datasets provide bounding boxes and some provide masks only. To support training on multiple datasets we opted to ignore the bounding boxes that come with the dataset and generate them on the fly instead. We pick the smallest box that encapsulates all the pixels of the mask as the bounding box. This simplifies the implementation and also makes it easy to apply image augmentations that would otherwise be harder to apply to bounding boxes, such as image rotation.</p>
<p dir="auto">To validate this approach, we compared our computed bounding boxes to those provided by the COCO dataset.
We found that ~2% of bounding boxes differed by 1px or more, ~0.05% differed by 5px or more,
and only 0.01% differed by 10px or more.</p>
</li>
<li>
<p dir="auto"><strong>Learning Rate:</strong> The paper uses a learning rate of 0.02, but we found that to be
too high, and often causes the weights to explode, especially when using a small batch
size. It might be related to differences between how Caffe and TensorFlow compute
gradients (sum vs mean across batches and GPUs). Or, maybe the official model uses gradient
clipping to avoid this issue. We do use gradient clipping, but don't set it too aggressively.
We found that smaller learning rates converge faster anyway so we go with that.</p>
</li>
</ul>
<h2 tabindex="-1" dir="auto"><a id="user-content-citation" class="anchor" aria-hidden="true" tabindex="-1" href="#citation"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a>Citation</h2>
<p dir="auto">Use this bibtex to cite this repository:</p>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>@misc{matterport_maskrcnn_2017,
  title={Mask R-CNN for object detection and instance segmentation on Keras and TensorFlow},
  author={Waleed Abdulla},
  year={2017},
  publisher={Github},
  journal={GitHub repository},
  howpublished={\url{https://github.com/matterport/Mask_RCNN}},
}
</code></pre><div class="zeroclipboard-container">
    <clipboard-copy aria-label="Copy" class="ClipboardButton btn btn-invisible js-clipboard-copy m-2 p-0 tooltipped-no-delay d-flex flex-justify-center flex-items-center" data-copy-feedback="Copied!" data-tooltip-direction="w" value="@misc{matterport_maskrcnn_2017,
  title={Mask R-CNN for object detection and instance segmentation on Keras and TensorFlow},
  author={Waleed Abdulla},
  year={2017},
  publisher={Github},
  journal={GitHub repository},
  howpublished={\url{https://github.com/matterport/Mask_RCNN}},
}" tabindex="0" role="button">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon">
    <path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path>
</svg>
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-fg-success d-none">
    <path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z"></path>
</svg>
    </clipboard-copy>
  </div></div>
<h2 tabindex="-1" dir="auto"><a id="user-content-contributing" class="anchor" aria-hidden="true" tabindex="-1" href="#contributing"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a>Contributing</h2>
<p dir="auto">Contributions to this repository are welcome. Examples of things you can contribute:</p>
<ul dir="auto">
<li>Speed Improvements. Like re-writing some Python code in TensorFlow or Cython.</li>
<li>Training on other datasets.</li>
<li>Accuracy Improvements.</li>
<li>Visualizations and examples.</li>
</ul>
<p dir="auto">You can also <a href="https://matterport.com/careers/" rel="nofollow">join our team</a> and help us build even more projects like this one.</p>
<h2 tabindex="-1" dir="auto"><a id="user-content-requirements" class="anchor" aria-hidden="true" tabindex="-1" href="#requirements"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a>Requirements</h2>
<p dir="auto">Python 3.4, TensorFlow 1.3, Keras 2.0.8 and other common packages listed in <code>requirements.txt</code>.</p>
<h3 tabindex="-1" dir="auto"><a id="user-content-ms-coco-requirements" class="anchor" aria-hidden="true" tabindex="-1" href="#ms-coco-requirements"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a>MS COCO Requirements:</h3>
<p dir="auto">To train or test on MS COCO, you'll also need:</p>
<ul dir="auto">
<li>pycocotools (installation instructions below)</li>
<li><a href="http://cocodataset.org/#home" rel="nofollow">MS COCO Dataset</a></li>
<li>Download the 5K <a href="https://dl.dropboxusercontent.com/s/o43o90bna78omob/instances_minival2014.json.zip?dl=0" rel="nofollow">minival</a>
and the 35K <a href="https://dl.dropboxusercontent.com/s/s3tw5zcg7395368/instances_valminusminival2014.json.zip?dl=0" rel="nofollow">validation-minus-minival</a>
subsets. More details in the original <a href="https://github.com/rbgirshick/py-faster-rcnn/blob/master/data/README.md">Faster R-CNN implementation</a>.</li>
</ul>
<p dir="auto">If you use Docker, the code has been verified to work on
<a href="https://hub.docker.com/r/waleedka/modern-deep-learning/" rel="nofollow">this Docker container</a>.</p>
<h2 tabindex="-1" dir="auto"><a id="user-content-installation" class="anchor" aria-hidden="true" tabindex="-1" href="#installation"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a>Installation</h2>
<ol dir="auto">
<li>
<p dir="auto">Clone this repository</p>
</li>
<li>
<p dir="auto">Install dependencies</p>
<div class="highlight highlight-source-shell notranslate position-relative overflow-auto" dir="auto"><pre>pip3 install -r requirements.txt</pre><div class="zeroclipboard-container">
    <clipboard-copy aria-label="Copy" class="ClipboardButton btn btn-invisible js-clipboard-copy m-2 p-0 tooltipped-no-delay d-flex flex-justify-center flex-items-center" data-copy-feedback="Copied!" data-tooltip-direction="w" value="pip3 install -r requirements.txt" tabindex="0" role="button">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon">
    <path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path>
</svg>
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-fg-success d-none">
    <path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z"></path>
</svg>
    </clipboard-copy>
  </div></div>
</li>
<li>
<p dir="auto">Run setup from the repository root directory</p>
<div class="highlight highlight-source-shell notranslate position-relative overflow-auto" dir="auto"><pre>python3 setup.py install</pre><div class="zeroclipboard-container">
    <clipboard-copy aria-label="Copy" class="ClipboardButton btn btn-invisible js-clipboard-copy m-2 p-0 tooltipped-no-delay d-flex flex-justify-center flex-items-center" data-copy-feedback="Copied!" data-tooltip-direction="w" value="python3 setup.py install" tabindex="0" role="button">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon">
    <path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path>
</svg>
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-fg-success d-none">
    <path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z"></path>
</svg>
    </clipboard-copy>
  </div></div>
</li>
<li>
<p dir="auto"><font style="vertical-align: inherit;"></font><a href="https://github.com/matterport/Mask_RCNN/releases"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">从发布页面</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">下载预训练的 COCO 权重 (mask_rcnn_coco.h5) </font><font style="vertical-align: inherit;">。</font></font></p>
</li>
<li>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">（可选）要</font></font><code>pycocotools</code><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">在从这些存储库之一安装的 MS COCO 上进行训练或测试。它们是原始 pycocotools 的分支，并修复了 Python3 和 Windows（官方存储库似乎不再活跃）。</font></font></p>
<ul dir="auto">
<li><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">Linux： https: </font></font><a href="https://github.com/waleedka/coco"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">//github.com/waleedka/coco</font></font></a></li>
<li><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">Windows： https: </font></font><a href="https://github.com/philferriere/cocoapi"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">//github.com/philferriere/cocoapi</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">。您的路径上必须有 Visual C++ 2015 构建工具（有关其他详细信息，请参阅存储库）</font></font></li>
</ul>
</li>
</ol>
<h1 tabindex="-1" dir="auto"><a id="user-content-projects-using-this-model" class="anchor" aria-hidden="true" tabindex="-1" href="#projects-using-this-model"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">使用此模型的项目</font></font></h1>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">如果您将此模型扩展到其他数据集或构建使用它的项目，我们很乐意听取您的意见。</font></font></p>
<h3 tabindex="-1" dir="auto"><a id="user-content-4k-video-demo-by-karol-majek" class="anchor" aria-hidden="true" tabindex="-1" href="#4k-video-demo-by-karol-majek"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a><a href="https://www.youtube.com/watch?v=OOT3UIXZztE" rel="nofollow"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">Karol Majek 的4K 视频演示</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">。</font></font></h3>
<p dir="auto"><animated-image data-catalyst=""><a href="https://www.youtube.com/watch?v=OOT3UIXZztE" rel="nofollow" data-target="animated-image.originalLink"><img src="/matterport/Mask_RCNN/raw/master/assets/4k_video.gif" alt="4K 视频上的 Mask RCNN" style="max-width: 100%; display: inline-block;" data-target="animated-image.originalImage"></a>
      <span class="AnimatedImagePlayer" data-target="animated-image.player" hidden="">
        <a data-target="animated-image.replacedLink" class="AnimatedImagePlayer-images" href="https://www.youtube.com/watch?v=OOT3UIXZztE" target="_blank">
          
        <span data-target="animated-image.imageContainer">
            <img data-target="animated-image.replacedImage" alt="4K 视频上的 Mask RCNN" class="AnimatedImagePlayer-animatedImage" src="https://github.com/matterport/Mask_RCNN/raw/master/assets/4k_video.gif" style="display: block; opacity: 1;">
          <canvas class="AnimatedImagePlayer-stillImage" aria-hidden="true" width="680" height="382"></canvas></span></a>
        <button data-target="animated-image.imageButton" class="AnimatedImagePlayer-images" tabindex="-1" aria-label="在 4K 视频上播放 Mask RCNN" hidden=""></button>
        <span class="AnimatedImagePlayer-controls" data-target="animated-image.controls" hidden="">
          <button data-target="animated-image.playButton" class="AnimatedImagePlayer-button" aria-label="在 4K 视频上播放 Mask RCNN">
            <svg aria-hidden="true" focusable="false" class="octicon icon-play" width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M4 13.5427V2.45734C4 1.82607 4.69692 1.4435 5.2295 1.78241L13.9394 7.32507C14.4334 7.63943 14.4334 8.36057 13.9394 8.67493L5.2295 14.2176C4.69692 14.5565 4 14.1739 4 13.5427Z">
            </path></svg>
            <svg aria-hidden="true" focusable="false" class="octicon icon-pause" width="16" height="16" viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg">
              <rect x="4" y="2" width="3" height="12" rx="1"></rect>
              <rect x="9" y="2" width="3" height="12" rx="1"></rect>
            </svg>
          </button>
          <a data-target="animated-image.openButton" aria-label="在新窗口中打开 4K 视频上的 Mask RCNN" class="AnimatedImagePlayer-button" href="https://www.youtube.com/watch?v=OOT3UIXZztE" target="_blank">
            <svg aria-hidden="true" class="octicon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" width="16" height="16">
              <path fill-rule="evenodd" d="M10.604 1h4.146a.25.25 0 01.25.25v4.146a.25.25 0 01-.427.177L13.03 4.03 9.28 7.78a.75.75 0 01-1.06-1.06l3.75-3.75-1.543-1.543A.25.25 0 0110.604 1zM3.75 2A1.75 1.75 0 002 3.75v8.5c0 .966.784 1.75 1.75 1.75h8.5A1.75 1.75 0 0014 12.25v-3.5a.75.75 0 00-1.5 0v3.5a.25.25 0 01-.25.25h-8.5a.25.25 0 01-.25-.25v-8.5a.25.25 0 01.25-.25h3.5a.75.75 0 000-1.5h-3.5z"></path>
            </svg>
          </a>
        </span>
      </span></animated-image></p>
<h3 tabindex="-1" dir="auto"><a id="user-content-images-to-osm-improve-openstreetmap-by-adding-baseball-soccer-tennis-football-and-basketball-fields" class="anchor" aria-hidden="true" tabindex="-1" href="#images-to-osm-improve-openstreetmap-by-adding-baseball-soccer-tennis-football-and-basketball-fields"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a><a href="https://github.com/jremillard/images-to-osm"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">图像到 OSM</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">：通过添加棒球场、足球场、网球场、橄榄球场和篮球场来改进 OpenStreetMap。</font></font></h3>
<p dir="auto"><a target="_blank" rel="noopener noreferrer" href="/matterport/Mask_RCNN/blob/master/assets/images_to_osm.png"><img src="/matterport/Mask_RCNN/raw/master/assets/images_to_osm.png" alt="识别卫星图像中的运动场" style="max-width: 100%;"></a></p>
<h3 tabindex="-1" dir="auto"><a id="user-content-splash-of-color-a-blog-post-explaining-how-to-train-this-model-from-scratch-and-use-it-to-implement-a-color-splash-effect" class="anchor" aria-hidden="true" tabindex="-1" href="#splash-of-color-a-blog-post-explaining-how-to-train-this-model-from-scratch-and-use-it-to-implement-a-color-splash-effect"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a><a href="https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46" rel="nofollow"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">色彩飞溅</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">。一篇博客文章解释了如何从头开始训练该模型并使用它来实现颜色飞溅效果。</font></font></h3>
<p dir="auto"><animated-image data-catalyst=""><a target="_blank" rel="noopener noreferrer" href="/matterport/Mask_RCNN/blob/master/assets/balloon_color_splash.gif" data-target="animated-image.originalLink"><img src="/matterport/Mask_RCNN/raw/master/assets/balloon_color_splash.gif" alt="气球颜色飞溅" style="max-width: 100%; display: inline-block;" data-target="animated-image.originalImage"></a>
      <span class="AnimatedImagePlayer" data-target="animated-image.player" hidden="">
        <a data-target="animated-image.replacedLink" class="AnimatedImagePlayer-images" href="https://github.com/matterport/Mask_RCNN/blob/master/assets/balloon_color_splash.gif" target="_blank">
          
        <span data-target="animated-image.imageContainer">
            <img data-target="animated-image.replacedImage" alt="气球颜色飞溅" class="AnimatedImagePlayer-animatedImage" src="https://github.com/matterport/Mask_RCNN/raw/master/assets/balloon_color_splash.gif" style="display: block; opacity: 1;">
          <canvas class="AnimatedImagePlayer-stillImage" aria-hidden="true" width="460" height="444"></canvas></span></a>
        <button data-target="animated-image.imageButton" class="AnimatedImagePlayer-images" tabindex="-1" aria-label="玩气球颜色飞溅" hidden=""></button>
        <span class="AnimatedImagePlayer-controls" data-target="animated-image.controls" hidden="">
          <button data-target="animated-image.playButton" class="AnimatedImagePlayer-button" aria-label="玩气球颜色飞溅">
            <svg aria-hidden="true" focusable="false" class="octicon icon-play" width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M4 13.5427V2.45734C4 1.82607 4.69692 1.4435 5.2295 1.78241L13.9394 7.32507C14.4334 7.63943 14.4334 8.36057 13.9394 8.67493L5.2295 14.2176C4.69692 14.5565 4 14.1739 4 13.5427Z">
            </path></svg>
            <svg aria-hidden="true" focusable="false" class="octicon icon-pause" width="16" height="16" viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg">
              <rect x="4" y="2" width="3" height="12" rx="1"></rect>
              <rect x="9" y="2" width="3" height="12" rx="1"></rect>
            </svg>
          </button>
          <a data-target="animated-image.openButton" aria-label="在新窗口中打开气球颜色飞溅" class="AnimatedImagePlayer-button" href="https://github.com/matterport/Mask_RCNN/blob/master/assets/balloon_color_splash.gif" target="_blank">
            <svg aria-hidden="true" class="octicon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" width="16" height="16">
              <path fill-rule="evenodd" d="M10.604 1h4.146a.25.25 0 01.25.25v4.146a.25.25 0 01-.427.177L13.03 4.03 9.28 7.78a.75.75 0 01-1.06-1.06l3.75-3.75-1.543-1.543A.25.25 0 0110.604 1zM3.75 2A1.75 1.75 0 002 3.75v8.5c0 .966.784 1.75 1.75 1.75h8.5A1.75 1.75 0 0014 12.25v-3.5a.75.75 0 00-1.5 0v3.5a.25.25 0 01-.25.25h-8.5a.25.25 0 01-.25-.25v-8.5a.25.25 0 01.25-.25h3.5a.75.75 0 000-1.5h-3.5z"></path>
            </svg>
          </a>
        </span>
      </span></animated-image></p>
<h3 tabindex="-1" dir="auto"><a id="user-content-segmenting-nuclei-in-microscopy-images-built-for-the-2018-data-science-bowl" class="anchor" aria-hidden="true" tabindex="-1" href="#segmenting-nuclei-in-microscopy-images-built-for-the-2018-data-science-bowl"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a><a href="/matterport/Mask_RCNN/blob/master/samples/nucleus"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">显微镜图像中的细胞核分割</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">。专为</font></font><a href="https://www.kaggle.com/c/data-science-bowl-2018" rel="nofollow"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">2018 年数据科学碗打造</font></font></a></h3>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">代码在</font></font><code>samples/nucleus</code><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">目录中。</font></font></p>
<p dir="auto"><a target="_blank" rel="noopener noreferrer" href="/matterport/Mask_RCNN/blob/master/assets/nucleus_segmentation.png"><img src="/matterport/Mask_RCNN/raw/master/assets/nucleus_segmentation.png" alt="细胞核分割" style="max-width: 100%;"></a></p>
<h3 tabindex="-1" dir="auto"><a id="user-content-detection-and-segmentation-for-surgery-robots-by-the-nus-control--mechatronics-lab" class="anchor" aria-hidden="true" tabindex="-1" href="#detection-and-segmentation-for-surgery-robots-by-the-nus-control--mechatronics-lab"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a><a href="https://github.com/SUYEgit/Surgery-Robot-Detection-Segmentation"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">新加坡国立大学控制与机电一体化实验室的手术机器人检测和分割</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">。</font></font></h3>
<p dir="auto"><animated-image data-catalyst=""><a target="_blank" rel="noopener noreferrer" href="https://github.com/SUYEgit/Surgery-Robot-Detection-Segmentation/raw/master/assets/video.gif" data-target="animated-image.originalLink"><img src="https://github.com/SUYEgit/Surgery-Robot-Detection-Segmentation/raw/master/assets/video.gif" alt="手术机器人检测与分割" style="max-width: 100%; display: inline-block;" data-target="animated-image.originalImage"></a>
      <span class="AnimatedImagePlayer" data-target="animated-image.player" hidden="">
        <a data-target="animated-image.replacedLink" class="AnimatedImagePlayer-images" href="https://github.com/SUYEgit/Surgery-Robot-Detection-Segmentation/raw/master/assets/video.gif" target="_blank">
          
        <span data-target="animated-image.imageContainer">
            <img data-target="animated-image.replacedImage" alt="手术机器人检测与分割" class="AnimatedImagePlayer-animatedImage" src="https://github.com/SUYEgit/Surgery-Robot-Detection-Segmentation/raw/master/assets/video.gif" style="display: block; opacity: 1;">
          <canvas class="AnimatedImagePlayer-stillImage" aria-hidden="true" width="769" height="431"></canvas></span></a>
        <button data-target="animated-image.imageButton" class="AnimatedImagePlayer-images" tabindex="-1" aria-label="玩手术机器人检测和分割" hidden=""></button>
        <span class="AnimatedImagePlayer-controls" data-target="animated-image.controls" hidden="">
          <button data-target="animated-image.playButton" class="AnimatedImagePlayer-button" aria-label="玩手术机器人检测和分割">
            <svg aria-hidden="true" focusable="false" class="octicon icon-play" width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M4 13.5427V2.45734C4 1.82607 4.69692 1.4435 5.2295 1.78241L13.9394 7.32507C14.4334 7.63943 14.4334 8.36057 13.9394 8.67493L5.2295 14.2176C4.69692 14.5565 4 14.1739 4 13.5427Z">
            </path></svg>
            <svg aria-hidden="true" focusable="false" class="octicon icon-pause" width="16" height="16" viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg">
              <rect x="4" y="2" width="3" height="12" rx="1"></rect>
              <rect x="9" y="2" width="3" height="12" rx="1"></rect>
            </svg>
          </button>
          <a data-target="animated-image.openButton" aria-label="在新窗口中打开手术机器人检测和分割" class="AnimatedImagePlayer-button" href="https://github.com/SUYEgit/Surgery-Robot-Detection-Segmentation/raw/master/assets/video.gif" target="_blank">
            <svg aria-hidden="true" class="octicon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" width="16" height="16">
              <path fill-rule="evenodd" d="M10.604 1h4.146a.25.25 0 01.25.25v4.146a.25.25 0 01-.427.177L13.03 4.03 9.28 7.78a.75.75 0 01-1.06-1.06l3.75-3.75-1.543-1.543A.25.25 0 0110.604 1zM3.75 2A1.75 1.75 0 002 3.75v8.5c0 .966.784 1.75 1.75 1.75h8.5A1.75 1.75 0 0014 12.25v-3.5a.75.75 0 00-1.5 0v3.5a.25.25 0 01-.25.25h-8.5a.25.25 0 01-.25-.25v-8.5a.25.25 0 01.25-.25h3.5a.75.75 0 000-1.5h-3.5z"></path>
            </svg>
          </a>
        </span>
      </span></animated-image></p>
<h3 tabindex="-1" dir="auto"><a id="user-content-reconstructing-3d-buildings-from-aerial-lidar" class="anchor" aria-hidden="true" tabindex="-1" href="#reconstructing-3d-buildings-from-aerial-lidar"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a><a href="https://medium.com/geoai/reconstructing-3d-buildings-from-aerial-lidar-with-ai-details-6a81cb3079c0" rel="nofollow"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">利用航空 LiDAR 重建 3D 建筑物</font></font></a></h3>
<p dir="auto"><font style="vertical-align: inherit;"></font><a href="https://www.esri.com/" rel="nofollow"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">Esri</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">与 Nvidia 和迈阿密戴德县合作的</font><font style="vertical-align: inherit;">概念验证项目。</font><font style="vertical-align: inherit;">还有 Dmitry Kudinov、Daniel Hedges 和 Omar Maher 的精彩撰写和代码。
</font></font><a target="_blank" rel="noopener noreferrer" href="/matterport/Mask_RCNN/blob/master/assets/project_3dbuildings.png"><img src="/matterport/Mask_RCNN/raw/master/assets/project_3dbuildings.png" alt="3D 建筑重建" style="max-width: 100%;"></a></p>
<h3 tabindex="-1" dir="auto"><a id="user-content-usiigaci-label-free-cell-tracking-in-phase-contrast-microscopy" class="anchor" aria-hidden="true" tabindex="-1" href="#usiigaci-label-free-cell-tracking-in-phase-contrast-microscopy"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a><a href="https://github.com/oist/usiigaci"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">Usiigaci：相差显微镜中的无标记细胞追踪</font></font></a></h3>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">来自日本的一个项目，旨在在微流体平台中自动跟踪细胞。论文正在等待中，但源代码已发布。</font></font></p>
<p dir="auto"><animated-image data-catalyst=""><a target="_blank" rel="noopener noreferrer" href="/matterport/Mask_RCNN/blob/master/assets/project_usiigaci1.gif" data-target="animated-image.originalLink"><img src="/matterport/Mask_RCNN/raw/master/assets/project_usiigaci1.gif" alt="" style="max-width: 100%; display: inline-block;" data-target="animated-image.originalImage"></a>
      <span class="AnimatedImagePlayer" data-target="animated-image.player" hidden="">
        <a data-target="animated-image.replacedLink" class="AnimatedImagePlayer-images" href="https://github.com/matterport/Mask_RCNN/blob/master/assets/project_usiigaci1.gif" target="_blank">
          
        <span data-target="animated-image.imageContainer">
            <img data-target="animated-image.replacedImage" alt="项目_usiigaci1.gif" class="AnimatedImagePlayer-animatedImage" src="https://github.com/matterport/Mask_RCNN/raw/master/assets/project_usiigaci1.gif" style="display: block; opacity: 1;">
          <canvas class="AnimatedImagePlayer-stillImage" aria-hidden="true" width="307" height="306"></canvas></span></a>
        <button data-target="animated-image.imageButton" class="AnimatedImagePlayer-images" tabindex="-1" aria-label="播放project_usiigaci1.gif" hidden=""></button>
        <span class="AnimatedImagePlayer-controls" data-target="animated-image.controls" hidden="">
          <button data-target="animated-image.playButton" class="AnimatedImagePlayer-button" aria-label="播放project_usiigaci1.gif">
            <svg aria-hidden="true" focusable="false" class="octicon icon-play" width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M4 13.5427V2.45734C4 1.82607 4.69692 1.4435 5.2295 1.78241L13.9394 7.32507C14.4334 7.63943 14.4334 8.36057 13.9394 8.67493L5.2295 14.2176C4.69692 14.5565 4 14.1739 4 13.5427Z">
            </path></svg>
            <svg aria-hidden="true" focusable="false" class="octicon icon-pause" width="16" height="16" viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg">
              <rect x="4" y="2" width="3" height="12" rx="1"></rect>
              <rect x="9" y="2" width="3" height="12" rx="1"></rect>
            </svg>
          </button>
          <a data-target="animated-image.openButton" aria-label="在新窗口中打开project_usiigaci1.gif" class="AnimatedImagePlayer-button" href="https://github.com/matterport/Mask_RCNN/blob/master/assets/project_usiigaci1.gif" target="_blank">
            <svg aria-hidden="true" class="octicon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" width="16" height="16">
              <path fill-rule="evenodd" d="M10.604 1h4.146a.25.25 0 01.25.25v4.146a.25.25 0 01-.427.177L13.03 4.03 9.28 7.78a.75.75 0 01-1.06-1.06l3.75-3.75-1.543-1.543A.25.25 0 0110.604 1zM3.75 2A1.75 1.75 0 002 3.75v8.5c0 .966.784 1.75 1.75 1.75h8.5A1.75 1.75 0 0014 12.25v-3.5a.75.75 0 00-1.5 0v3.5a.25.25 0 01-.25.25h-8.5a.25.25 0 01-.25-.25v-8.5a.25.25 0 01.25-.25h3.5a.75.75 0 000-1.5h-3.5z"></path>
            </svg>
          </a>
        </span>
      </span></animated-image> <animated-image data-catalyst=""><a target="_blank" rel="noopener noreferrer" href="/matterport/Mask_RCNN/blob/master/assets/project_usiigaci2.gif" data-target="animated-image.originalLink"><img src="/matterport/Mask_RCNN/raw/master/assets/project_usiigaci2.gif" alt="" style="max-width: 100%; display: inline-block;" data-target="animated-image.originalImage"></a>
      <span class="AnimatedImagePlayer" data-target="animated-image.player" hidden="">
        <a data-target="animated-image.replacedLink" class="AnimatedImagePlayer-images" href="https://github.com/matterport/Mask_RCNN/blob/master/assets/project_usiigaci2.gif" target="_blank">
          
        <span data-target="animated-image.imageContainer">
            <img data-target="animated-image.replacedImage" alt="项目_usiigaci2.gif" class="AnimatedImagePlayer-animatedImage" src="https://github.com/matterport/Mask_RCNN/raw/master/assets/project_usiigaci2.gif" style="display: block; opacity: 1;">
          <canvas class="AnimatedImagePlayer-stillImage" aria-hidden="true" width="307" height="305"></canvas></span></a>
        <button data-target="animated-image.imageButton" class="AnimatedImagePlayer-images" tabindex="-1" aria-label="播放project_usiigaci2.gif" hidden=""></button>
        <span class="AnimatedImagePlayer-controls" data-target="animated-image.controls" hidden="">
          <button data-target="animated-image.playButton" class="AnimatedImagePlayer-button" aria-label="播放project_usiigaci2.gif">
            <svg aria-hidden="true" focusable="false" class="octicon icon-play" width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M4 13.5427V2.45734C4 1.82607 4.69692 1.4435 5.2295 1.78241L13.9394 7.32507C14.4334 7.63943 14.4334 8.36057 13.9394 8.67493L5.2295 14.2176C4.69692 14.5565 4 14.1739 4 13.5427Z">
            </path></svg>
            <svg aria-hidden="true" focusable="false" class="octicon icon-pause" width="16" height="16" viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg">
              <rect x="4" y="2" width="3" height="12" rx="1"></rect>
              <rect x="9" y="2" width="3" height="12" rx="1"></rect>
            </svg>
          </button>
          <a data-target="animated-image.openButton" aria-label="在新窗口中打开project_usiigaci2.gif" class="AnimatedImagePlayer-button" href="https://github.com/matterport/Mask_RCNN/blob/master/assets/project_usiigaci2.gif" target="_blank">
            <svg aria-hidden="true" class="octicon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" width="16" height="16">
              <path fill-rule="evenodd" d="M10.604 1h4.146a.25.25 0 01.25.25v4.146a.25.25 0 01-.427.177L13.03 4.03 9.28 7.78a.75.75 0 01-1.06-1.06l3.75-3.75-1.543-1.543A.25.25 0 0110.604 1zM3.75 2A1.75 1.75 0 002 3.75v8.5c0 .966.784 1.75 1.75 1.75h8.5A1.75 1.75 0 0014 12.25v-3.5a.75.75 0 00-1.5 0v3.5a.25.25 0 01-.25.25h-8.5a.25.25 0 01-.25-.25v-8.5a.25.25 0 01.25-.25h3.5a.75.75 0 000-1.5h-3.5z"></path>
            </svg>
          </a>
        </span>
      </span></animated-image></p>
<h3 tabindex="-1" dir="auto"><a id="user-content-characterization-of-arctic-ice-wedge-polygons-in-very-high-spatial-resolution-aerial-imagery" class="anchor" aria-hidden="true" tabindex="-1" href="#characterization-of-arctic-ice-wedge-polygons-in-very-high-spatial-resolution-aerial-imagery"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a><a href="http://www.mdpi.com/2072-4292/10/9/1487" rel="nofollow"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">极高空间分辨率航空图像中北极冰楔多边形的表征</font></font></a></h3>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">研究项目旨在了解北极退化与气候变化之间的复杂过程。作者：张卫星、Chandi Witharana、Anna Liljedahl 和 Mikhail Kanevskiy。
</font></font><a target="_blank" rel="noopener noreferrer" href="/matterport/Mask_RCNN/blob/master/assets/project_ice_wedge_polygons.png"><img src="/matterport/Mask_RCNN/raw/master/assets/project_ice_wedge_polygons.png" alt="图像" style="max-width: 100%;"></a></p>
<h3 tabindex="-1" dir="auto"><a id="user-content-mask-rcnn-shiny" class="anchor" aria-hidden="true" tabindex="-1" href="#mask-rcnn-shiny"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a><a href="https://github.com/huuuuusy/Mask-RCNN-Shiny"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">Mask-RCNN 闪亮</font></font></a></h3>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">胡世宇的一个计算机视觉课程项目，将色彩流行效果应用到人物身上，效果很漂亮。
</font></font><a target="_blank" rel="noopener noreferrer" href="/matterport/Mask_RCNN/blob/master/assets/project_shiny1.jpg"><img src="/matterport/Mask_RCNN/raw/master/assets/project_shiny1.jpg" alt="" style="max-width: 100%;"></a></p>
<h3 tabindex="-1" dir="auto"><a id="user-content-mapping-challenge-convert-satellite-imagery-to-maps-for-use-by-humanitarian-organisations" class="anchor" aria-hidden="true" tabindex="-1" href="#mapping-challenge-convert-satellite-imagery-to-maps-for-use-by-humanitarian-organisations"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a><a href="https://github.com/crowdAI/crowdai-mapping-challenge-mask-rcnn"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">绘图挑战</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">：将卫星图像转换为地图以供人道主义组织使用。</font></font></h3>
<p dir="auto"><a target="_blank" rel="noopener noreferrer" href="/matterport/Mask_RCNN/blob/master/assets/mapping_challenge.png"><img src="/matterport/Mask_RCNN/raw/master/assets/mapping_challenge.png" alt="测绘挑战" style="max-width: 100%;"></a></p>
<h3 tabindex="-1" dir="auto"><a id="user-content-grass-gis-addon-to-generate-vector-masks-from-geospatial-imagery-based-on-a-masters-thesis-by-ondřej-pešek" class="anchor" aria-hidden="true" tabindex="-1" href="#grass-gis-addon-to-generate-vector-masks-from-geospatial-imagery-based-on-a-masters-thesis-by-ondřej-pešek"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a><a href="https://github.com/ctu-geoforall-lab/i.ann.maskrcnn"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">GRASS GIS Addon</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">用于从地理空间图像生成矢量蒙版。基于Ondřej Pešek 的</font></font><a href="https://github.com/ctu-geoforall-lab-projects/dp-pesek-2018"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">硕士论文</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">。</font></font></h3>
<p dir="auto"><a target="_blank" rel="noopener noreferrer" href="/matterport/Mask_RCNN/blob/master/assets/project_grass_gis.png"><img src="/matterport/Mask_RCNN/raw/master/assets/project_grass_gis.png" alt="草地 GIS 图像" style="max-width: 100%;"></a></p>
</article></div>
