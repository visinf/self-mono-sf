from . import model_monosceneflow
from . import model_monosceneflow_ablation
from . import model_monosceneflow_ablation_decoder_split
from . import model_monodepth_ablation

##########################################################################################
## Monocular Scene Flow - The full model 
##########################################################################################

MonoSceneFlow_fullmodel			=	model_monosceneflow.MonoSceneFlow

##########################################################################################
## Monocular Scene Flow - The models for the ablation studies
##########################################################################################

MonoSceneFlow_CamConv			=	model_monosceneflow_ablation.MonoSceneFlow_CamConv

MonoSceneFlow_FlowOnly			=	model_monosceneflow_ablation.MonoSceneFlow_OpticalFlowOnly
MonoSceneFlow_DispOnly			=	model_monosceneflow_ablation.MonoSceneFlow_DisparityOnly

MonoSceneFlow_Split_Cont		=	model_monosceneflow_ablation_decoder_split.SceneFlow_pwcnet_split_base
MonoSceneFlow_Split_Last1		=	model_monosceneflow_ablation_decoder_split.SceneFlow_pwcnet_split1
MonoSceneFlow_Split_Last2		=	model_monosceneflow_ablation_decoder_split.SceneFlow_pwcnet_split2
MonoSceneFlow_Split_Last3		=	model_monosceneflow_ablation_decoder_split.SceneFlow_pwcnet_split3
MonoSceneFlow_Split_Last4		=	model_monosceneflow_ablation_decoder_split.SceneFlow_pwcnet_split4
MonoSceneFlow_Split_Last5		=	model_monosceneflow_ablation_decoder_split.SceneFlow_pwcnet_split5

##########################################################################################
## Monocular Depth - The models for the ablation study in Table 1. 
##########################################################################################

MonoDepth_Baseline				= model_monodepth_ablation.MonoDepth_Baseline
MonoDepth_CamConv				= model_monodepth_ablation.MonoDepth_CamConv