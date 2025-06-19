from mmengine.registry import MODELS
from mmengine.model import BaseModule
import numpy as np
import torch.nn as nn, torch
import torch.nn.functional as F
from einops import rearrange
from copy import deepcopy
import torch.distributions as dist
from utils.metric_stp3 import PlanningMetric
import time

from utils.helper import sample_with_top_k_top_p_, gumbel_softmax_with_rng
from functools import partial

class WNConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        activation=None,
    ):
        super().__init__()

        self.conv = nn.utils.weight_norm(
            nn.Conv2d(
                in_channel,
                out_channel,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            )
        )

        self.out_channel = out_channel

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]

        self.kernel_size = kernel_size

        self.activation = activation

    def forward(self, input):
        out = self.conv(input)

        if self.activation is not None:
            out = self.activation(out)

        return out
    
class GatedResBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        channel,
        kernel_size,
        conv='wnconv2d',
        activation=nn.ELU,
        dropout=0.1,
        auxiliary_channel=0,
        condition_dim=0,
    ):
        super().__init__()

        if conv == 'wnconv2d':
            conv_module = partial(WNConv2d, padding=kernel_size // 2)

        self.activation = activation()
        self.conv1 = conv_module(in_channel, channel, kernel_size)

        if auxiliary_channel > 0:
            self.aux_conv = WNConv2d(auxiliary_channel, channel, 1)

        self.dropout = nn.Dropout(dropout)

        self.conv2 = conv_module(channel, in_channel * 2, kernel_size)

        if condition_dim > 0:
            # self.condition = nn.Linear(condition_dim, in_channel * 2, bias=False)
            self.condition = WNConv2d(condition_dim, in_channel * 2, 1, bias=False)

        self.gate = nn.GLU(1)

    def forward(self, input, aux_input=None, condition=None):
        out = self.conv1(self.activation(input))

        if aux_input is not None:
            out = out + self.aux_conv(self.activation(aux_input))

        out = self.activation(out)
        out = self.dropout(out)
        out = self.conv2(out)

        if condition is not None:
            condition = self.condition(condition)
            out += condition
            # out = out + condition.view(condition.shape[0], 1, 1, condition.shape[1])

        out = self.gate(out)
        out += input

        return out

class CondResNet(nn.Module):
    def __init__(self, in_channel, channel, kernel_size, n_res_block):
        super().__init__()

        blocks = [WNConv2d(in_channel, channel, kernel_size, padding=kernel_size // 2)]

        for i in range(n_res_block):
            blocks.append(GatedResBlock(channel, channel, kernel_size))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)
    
class PixelBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        channel,
        kernel_size,
        n_res_block,
        attention=True,
        dropout=0.1,
        condition_dim=0,
    ):
        super().__init__()

        resblocks = []
        for i in range(n_res_block):
            resblocks.append(
                GatedResBlock(
                    in_channel,
                    channel,
                    kernel_size,
                    conv='wnconv2d',
                    dropout=dropout,
                    condition_dim=condition_dim,
                )
            )

        self.resblocks = nn.ModuleList(resblocks)
        self.out = WNConv2d(in_channel, in_channel, 1)

    def forward(self, input, condition=None):
        out = input

        for resblock in self.resblocks:
            out = resblock(out, condition=condition)

        out = self.out(out)

        return out

@MODELS.register_module()
class TransRQVAEalign(BaseModule):
    # NOTE TODO WONHYEOK: TransVQVAE initilize할 때 "transformer_m" 인자 추가
    # def __init__(self, vae, transformer_t, transformer_m, transformer_b, num_frames=10, offset=1,
    def __init__(self, vae, transformer_1, num_frames=10, offset=1,
                 pose_encoder=None, pose_decoder=None,
                 pose_actor=None, give_hiddens=False, delta_input=False, without_all=False):
        super().__init__()
        self.num_frames = num_frames
        self.depth = 4 # NOTE hardcoded
        self.offset = offset
        # NOTE TODO: vae forward 수정 (z_q_b, z_q_m, z_q_t, idx_b, idx_m, idx_t, shapes_b, shapes_m, shapes_t)
        self.vae = MODELS.build(vae)
        self.vae.eval() 
        """ NOTE !IMPORTANT! vae, vqvae eval 동작안함 해결 필요 self.training == True """
        # self.vae.vqvae.eval()
        # self.transformer_t = MODELS.build(transformer_t)
        # self.transformer_m = MODELS.build(transformer_m)
        # self.transformer_b = MODELS.build(transformer_b)
        self.transformer_1 = MODELS.build(transformer_1)
        # self.transformer_2 = MODELS.build(transformer_2)
        # self.transformer_3 = MODELS.build(transformer_3)
        # self.transformer_4 = MODELS.build(transformer_4)
        if pose_encoder is not None:
            self.pose_encoder = MODELS.build(pose_encoder)
        if pose_decoder is not None:
            self.pose_decoder = MODELS.build(pose_decoder)
        if pose_actor is not None:
            self.pose_actor = MODELS.build(pose_actor)
        self.give_hiddens = give_hiddens
        self.delta_input = delta_input
        self.planning_metric = None
        self.without_all = without_all
        
    def forward(self, x, metas=None):
        if hasattr(self, 'pose_encoder'):
            if self.training:
                return self.forward_train_with_plan(x, metas)
            else:
                return self.forward_inference_with_plan(x, metas)
        if self.training:
            return self.forward_train(x)
        else:
            return self.forward_inference(x)

    def forward_train(self, x):
        # given x: bs, f, h, w, d where f == num_frames + offset
        # output : ce_inputs: logits for the codebook 
        # output : ce_labels: labels for the ce_inputs
        assert hasattr(self.vae, 'vqvae')
        bs, F, H, W, D = x.shape
        assert F == self.num_frames + self.offset
        output_dict = {}
        z, shape = self.vae.forward_encoder(x)
        z = self.vae.vqvae.quant_conv(z)
        z_q, loss, (perplexity, min_encodings, min_encoding_indices) = self.vae.vqvae.forward_quantizer(z, is_voxel=False)
        min_encoding_indices = rearrange(min_encoding_indices, '(b f) h w -> b f h w', b=bs)
        output_dict['ce_labels'] = min_encoding_indices[:, self.offset:].detach().flatten(0,1)
        z_q = rearrange(z_q, '(b f) c h w -> b f c h w', b=bs)
        hidden = None
        if self.give_hiddens:
            hidden = z_q[:, :self.offset]
        z_q_predict = self.transformer(z_q[:, :self.num_frames], hidden=hidden)
        z_q_predict = z_q_predict.flatten(0, 1)
        output_dict['ce_inputs'] = z_q_predict
        # z: bs*f, c, h, w 
        
        # z: bs*f, h, w
        return output_dict
        
    def forward_inference(self, x):
        bs, F, H, W, D = x.shape
        output_dict = {}
        output_dict['target_occs'] = x[:, self.offset:]
        z, shape = self.vae.forward_encoder(x)
        z = self.vae.vqvae.quant_conv(z)
        z_q, loss, (perplexity, min_encodings, min_encoding_indices) = self.vae.vqvae.forward_quantizer(z, is_voxel=False)
        min_encoding_indices = rearrange(min_encoding_indices, '(b f) h w -> b f h w', b=bs)
        output_dict['ce_labels'] = min_encoding_indices[:, self.offset:].detach().flatten(0,1)
        z_q = rearrange(z_q, '(b f) c h w -> b f c h w', b=bs)
        hidden = None
        if self.give_hiddens:
            hidden = z_q[:, :self.offset]
        z_q_predict = self.transformer(z_q[:, :self.num_frames], hidden=hidden)
        z_q_predict = z_q_predict.flatten(0, 1)
        output_dict['ce_inputs'] = z_q_predict
        z_q_predict = z_q_predict.argmax(dim=1)
        z_q_predict = self.vae.vqvae.get_codebook_entry(z_q_predict, shape=None)
        z_q_predict = rearrange(z_q_predict, 'bf h w c-> bf c h w')
        z_q_predict = self.vae.vqvae.post_quant_conv(z_q_predict)
        
        z_q_predict = self.vae.forward_decoder(z_q_predict, shape, output_dict['target_occs'].shape)
        output_dict['logits'] = z_q_predict
        pred = z_q_predict.argmax(dim=-1).detach().cuda()
        output_dict['sem_pred'] = pred
        pred_iou = deepcopy(pred)
        
        pred_iou[pred_iou!=17] = 1
        pred_iou[pred_iou==17] = 0
        output_dict['iou_pred'] = pred_iou
    
        return output_dict
    
    ############ TODO : KYUMIN VAR : TODO ############################################################################
    def forward_train_with_plan(self, x, metas):
        is_soft = True
        assert hasattr(self, 'pose_encoder')
        bs, f, H, W, D = x.shape # F = 16 = 15 + 1
        assert f == self.num_frames + self.offset
        output_dict = {}
        if is_soft:
            residual_list, code_list, soft_code_list, shapes, out_shape = self.vae.encode_soft_code(x, temp=1.0, stochastic=False)
            output_dict['ce_labels_1'] = rearrange(torch.stack(soft_code_list, dim=1)[self.offset:].detach(), 'bf d h w c -> (bf d) c h w') # bfd c h w
        else:
            final_quant, residual_list, code_list, shape, out_shape = self.vae.encode(x) # [bs*f, H, W, 50], [bs*f, H, W]
            output_dict['ce_labels_1'] = torch.stack(code_list, dim=1)[self.offset:].flatten(0, 1).detach() # bfd h w

        rel_poses_, output_metas = self._get_pose_feature(metas, f-self.offset) # sequential poses except for last time stamp

        for i in range(len(residual_list)):
            residual_list[i] = rearrange(residual_list[i], '(b f) h w c -> b f c h w', b=bs, f=f)

        z_q_1 = torch.stack(residual_list, dim=-1).sum(-1) # b f c h w
        res_tokens = torch.cumsum(torch.stack(residual_list, dim=-1), dim=-1) # b f c h w d

        '''
        Transformer input
        1. queries = res_tokens 
        2. tokens = z_q_1
        3. causal : (f*d, f)
        '''
        
        #############################################################
        z_q_forecast, rel_poses = self.transformer_1( 
                                        res_tokens[:, :self.num_frames],
                                        z_q_1[:, :self.num_frames],
                                        rel_poses_
                                    ) # b f' d 128 50 50
        #############################################################
        
        output_dict['ce_inputs_1'] = z_q_forecast.flatten(0, 1).flatten(0, 1) # bfd c h w

        pose_decoded = self.pose_decoder(rel_poses) # [bs, f-1, 3, 2]

        output_dict['pose_decoded'] = pose_decoded
        output_dict['output_metas'] = output_metas
        
        return output_dict

    def forward_inference_with_plan(self, x, metas):
        bs, f, H, W, D = x.shape # [1, 16, 200, 200, 16]
        output_dict = {}
        output_dict['target_occs'] = x[:, self.offset:]
        final_quant, residual_list, code_list, shapes, out_shape = self.vae.encode(x) # [bs*f, H, W, 50], [bs*f, H, W]
        output_dict['ce_labels_1'] = torch.stack(code_list, dim=1)[self.offset:].flatten(0, 1).detach() # bfd h w

        rel_poses_, output_metas = self._get_pose_feature(metas, f-self.offset) # sequential poses except for last time stamp

        for i in range(len(residual_list)):
            residual_list[i] = rearrange(residual_list[i], '(b f) h w c -> b f c h w', b=bs, f=f)

        z_q_1 = torch.stack(residual_list, dim=-1).sum(-1) # b f c h w
        res_tokens = torch.cumsum(torch.stack(residual_list, dim=-1), dim=-1) # b f c h w d
        
        #############################################################
        z_shape = z_q_1[:, :self.num_frames].shape
        z_q_forecast, rel_poses = self.transformer_1( 
                                        res_tokens[:, :self.num_frames],
                                        z_q_1[:, :self.num_frames],
                                        rel_poses_
                                    ) # b f' d 128 50 50
        idx_ = torch.argmax(z_q_forecast.flatten(0, 1).flatten(0, 1), dim=1)
        quant_ = self.vae.vqvae.embed_code(idx_).permute(0, 3, 1, 2).view(*z_shape[:2], 4, *z_shape[2:]) # bf'd 128 50 50
        quant_final = quant_.sum(2).flatten(0, 1).permute(0, 2, 3, 1) # b f' c h w

        output_dict['ce_inputs_1'] = z_q_forecast.flatten(0, 1).flatten(0, 1) # bdf c h w
        # output_dict['ce_inputs_2'] = z_q_2.flatten(0, 1) # bdf c h w
        # output_dict['ce_inputs_3'] = z_q_3.flatten(0, 1) # bdf c h w
        # output_dict['ce_inputs_4'] = z_q_4.flatten(0, 1) # bdf c h w

        pose_decoded = self.pose_decoder(rel_poses) # [bs, f-1, 3, 2]

        output_dict['pose_decoded'] = pose_decoded
        output_dict['output_metas'] = output_metas

        logits = self.vae.decode(quant_final, shapes, output_dict['target_occs'].shape)
        # logits = self.vae.decode_code(final_z_q_t, final_z_q_b, shapes_b, output_dict['target_occs'].shape)
        # logits = self.vae.decode_code(idx_t[self.offset:], idx_b[self.offset:], shapes_b, output_dict['target_occs'].shape)
    
        output_dict['logits'] = logits # [bs, f-1, 200, 200, 16, 18]
        pred = logits.argmax(dim=-1).detach().cuda()
        output_dict['sem_pred'] = pred
        pred_iou = deepcopy(pred)
        
        pred_iou[pred_iou!=17] = 1
        pred_iou[pred_iou==17] = 0
        output_dict['iou_pred'] = pred_iou
    
        return output_dict

    def _get_pose_feature(self, metas=None, F=None):
        rel_poses, output_metas = None, None
        if hasattr(self, 'pose_encoder'):
            assert hasattr(self, 'pose_decoder')
            assert metas is not None
            output_metas = []
            for meta in metas:
                output_meta = dict()
                output_meta['rel_poses'] = meta['rel_poses'][self.offset:] # [16, 2] -> [15, 2]
                output_meta['gt_mode'] = meta['gt_mode'][self.offset:] # [16, 3] -> [15, 3]
                output_metas.append(output_meta)
                
            rel_poses = np.array([meta['rel_poses'] for meta in metas]) 
            gt_mode = np.array([meta['gt_mode'] for meta in metas]) 
            

            gt_mode = torch.tensor(gt_mode).cuda()
            rel_poses = torch.tensor(rel_poses).cuda()
            if self.delta_input:
                rel_poses_pre = torch.cat([torch.zeros_like(rel_poses[:, :1]), rel_poses[:, :-1]], dim=1)
                rel_poses = rel_poses - rel_poses_pre
            if F>self.num_frames:
                assert F == self.num_frames + self.offset
            else:
                assert F == self.num_frames
                gt_mode = gt_mode[:, :-self.offset, :] # [bs, F-1, 2]
                rel_poses = rel_poses[:, :-self.offset, :] # [bs, F-1, 3]
                
            rel_poses = torch.cat([rel_poses, gt_mode], dim=-1)
            #rel_poses = rearrange(rel_poses, 'b f d -> b f 1 d')
            rel_poses = self.pose_encoder(rel_poses.float()) # [bs, F-1, 128]
        return rel_poses, output_metas
    
    def forward_autoreg_with_pose(self, x, metas, start_frame=0, mid_frame=6,end_frame=12): # 0, 5, 11
        t0 = time.time()
        bs, f, H, W, D = x.shape # [bs, f=12, h=200, w=200, d=16]
        output_dict = {}
        output_dict['input_occs'] = x[:, mid_frame-1:end_frame] 
        output_dict['target_occs'] = x[:, mid_frame:end_frame] 
        # NOTE WONHYEOK: ADD MID SCALE
        final_quant, residual_list, code_list, shapes, out_shape = self.vae.encode(x) # [bs*f, H, W, 50], [bs*f, H, W]
        # z_q_b, z_q_m, z_q_t, idx_b, idx_m, idx_t, shapes_b, out_shape = self.vae.encode(x) # [bs*f, H, W, 50], [bs*f, H, W] # NOTE ADDED
        # z_q_b, z_q_t, idx_b, idx_t, shapes_b, out_shape = self.vae.encode(x) # [bs*f, H, W, 50], [bs*f, H, W]
        output_dict['ce_labels_1'] = torch.stack(code_list, dim=1)[mid_frame:end_frame].flatten(0, 1).detach() # bf'd h w
        # output_dict['ce_labels_t'] = idx_t[mid_frame:end_frame].detach()
        # output_dict['ce_labels_m'] = idx_m[mid_frame:end_frame].detach() # NOTE ADDED
        # output_dict['ce_labels_b'] = idx_b[mid_frame:end_frame].detach()
        for i in range(len(residual_list)):
            residual_list[i] = rearrange(residual_list[i], '(b f) h w c -> b f c h w', b=bs, f=f)

        z_q_1 = torch.stack(residual_list, dim=-1).sum(-1) # b f c h w
        res_tokens = torch.cumsum(torch.stack(residual_list, dim=-1), dim=-1) # b f c h w d
        z_q_1_predict = z_q_1[:, start_frame:mid_frame]
        res_predict = res_tokens[:, start_frame:mid_frame]

        t1 = time.time()
        output_metas = []
        input_metas = []
        for meta in metas:
            input_meta = dict()
            input_meta['rel_poses'] = meta['rel_poses'][start_frame:mid_frame]
            input_meta['gt_mode'] = meta['gt_mode'][start_frame:mid_frame]
            input_metas.append(input_meta)
        output_dict['input_metas'] = input_metas
        for meta in metas:
            output_meta = dict()
            output_meta['rel_poses'] = meta['rel_poses'][mid_frame:end_frame]#-meta['rel_poses'][mid_frame-1]
            output_meta['gt_mode'] = meta['gt_mode'][mid_frame:end_frame]
            output_metas.append(output_meta)
        output_dict['gt_poses_'] = np.array([meta['rel_poses'] for meta in output_metas])
        rel_poses = np.array([meta['rel_poses'] for meta in metas])
        gt_mode = np.array([meta['gt_mode'] for meta in metas])
        gt_mode = torch.tensor(gt_mode).cuda()
        
        rel_poses = torch.tensor(rel_poses).cuda()
        if self.delta_input:
            rel_poses_pre = torch.cat(torch.zeros_like(rel_poses[:, :1]), rel_poses[:, :-1], dim=1)
            rel_poses = rel_poses - rel_poses_pre
        rel_poses_sumed = rel_poses[:, start_frame:mid_frame]
        rel_poses = torch.cat([rel_poses, gt_mode], dim=-1)
        rel_poses = rel_poses[:, start_frame:mid_frame]
        
        rel_poses = self.pose_encoder(rel_poses.float())
        rel_poses_state = rel_poses
        z_q_1_list = [] # for CE_LOSS
        # res_list = [] # NOTE ADDED
        t2 = time.time()
        poses_ = []
        for i in range(mid_frame, end_frame):
            ###########################################
            # NOTE OPTION 2. residual + cascade + frame index align
            z_shape = z_q_1_predict.shape # (b f' 128 50 50)
            z_q_1_, rel_poses_= self.transformer_1.forward_autoreg_step( # NOTE only FORECAST in this scale rough forecast and refine in upscaled features
                z_q_1_predict,
                res_predict,
                rel_poses_state,
                start_frame=start_frame, mid_frame=i) # b f d c h w
            idx_1 = torch.argmax(z_q_1_.flatten(0, 1).flatten(0, 1), dim=1) # bfd 50 50
            quant_1 = self.vae.vqvae.embed_code(idx_1).permute(0,3,1,2).view(*z_shape[:2], 4, *z_shape[2:]) # b f d, 128, 50, 50
            quant_1 = rearrange(quant_1, 'b f d c h w -> b f c h w d')
            ###########################################

            z_q_1_list.append(z_q_1_[:, -1:]) # b 1 d c h w
            # z_q_1_list.append(z_q_m_[:, -1:]) # NOTE ADDED
            # z_q_b_list.append(z_q_b_[:, -1:])

            # temp_z_q_t_idx = torch.argmax(z_q_t_[:, -1:].flatten(0, 1).detach(), dim=1) # IDEAL 5, 512, 12, 12 -> 5, 12, 12
            # temp_z_q_m_idx = torch.argmax(z_q_m_[:, -1:].flatten(0, 1).detach(), dim=1) # NOTE ADDED
            # temp_z_q_b_idx = torch.argmax(z_q_b_[:, -1:].flatten(0, 1).detach(), dim=1)
            # temp_z_q_t_idx = self.sample_with_top_k_top_p_(z_q_t_[:, -1:].flatten(0, 1).detach().clone()) # IDEAL 5, 512, 12, 12 -> 5, 12, 12
            # temp_z_q_m_idx = self.sample_with_top_k_top_p_(z_q_m_[:, -1:].flatten(0, 1).detach().clone()) # NOTE ADDED
            # temp_z_q_b_idx = self.sample_with_top_k_top_p_(z_q_b_[:, -1:].flatten(0, 1).detach().clone())

            z_q_1_ = quant_1.sum(-1)[:, -1:] # b 1 c h w
            res_tokens = torch.cumsum(quant_1, dim=-1)[:, -1:] # b 1 c h w d
            """"""
            # TODO 여기부터 다시 작성할것
            """"""
            # z_q_t_, z_q_m_, z_q_b_ = self.vae.get_code(temp_z_q_t_idx, temp_z_q_m_idx, temp_z_q_b_idx) # 5, 128, 12, 12
            # print("z_q_t_.unique:", torch.unique(z_q_t_))

            # assert z_q_t.shape[0] == z_q_m.shape[0] and z_q_t.shape[0] == z_q_b.shape[0] # NOTE ADD ASSERTION, CHECK PURPOSE
            # z_q_t_ = rearrange(z_q_t_, '(b f) h w c -> b f c h w', b=bs, f=z_q_t_.shape[0]) # NOTE QUESTION shape[0]을 왜쓰지? bs가 1이여서?
            # z_q_m_ = rearrange(z_q_m_, '(b f) h w c -> b f c h w', b=bs, f=z_q_m_.shape[0])
            # z_q_b_ = rearrange(z_q_b_, '(b f) h w c -> b f c h w', b=bs, f=z_q_b_.shape[0])

            # print("z_q_t_predict.shape:", z_q_t_predict.shape)  # 1, 5, 128, 12, 12
            # print("z_q_t.shape:", z_q_t_.shape)                  # 1, 12, 128, 12, 12
            z_q_1_predict = torch.cat([z_q_1_predict, z_q_1_], dim=1)
            res_predict = torch.cat([res_predict, res_tokens], dim=1)
            ##########################
            # rel_pose_avg_ = (rel_poses_b_ + rel_poses_m_ + rel_poses_t_)/3.0
            ##########################
            rel_poses = torch.cat([rel_poses, rel_poses_[:, -1:]], dim=1)
            rel_poses_state_, rel_poses_sumed, pose_ = self.decode_pose(rel_poses_[:, -1:], gt_mode[:,i:i+1], rel_poses_sumed)
            # rel_poses = torch.cat([rel_poses, rel_poses_b_[:, -1:]], dim=1)
            # rel_poses_state_, rel_poses_sumed, pose_ = self.decode_pose(rel_poses_b_[:, -1:], gt_mode[:,i:i+1], rel_poses_sumed)
            poses_.append(pose_)
            rel_poses_state = torch.cat([rel_poses_state, rel_poses_state_], dim=1)
        poses_ = torch.cat(poses_, dim=1)
        output_dict['poses_'] = poses_
        t3 = time.time()

        z_q_1_predict = z_q_1_predict[:, mid_frame:end_frame] 
        # res_predict = res_predict[:, mid_frame:end_frame]
        # z_q_t_predict = z_q_t_predict[:, mid_frame:end_frame] 
        # z_q_m_predict = z_q_m_predict[:, mid_frame:end_frame] 
        # z_q_b_predict = z_q_b_predict[:, mid_frame:end_frame] 
        rel_poses = rel_poses[:, mid_frame:end_frame]
        pose_decoded = self.pose_decoder(rel_poses)
        output_dict['pose_decoded'] = pose_decoded
        output_dict['output_metas'] = output_metas
        
        ce_inputs = rearrange(torch.cat(z_q_1_list, dim=1), 'b f d c h w -> (b f d) c h w')
        output_dict['ce_inputs_1'] = ce_inputs 
        
        z_q_1_predict = z_q_1_predict.flatten(0, 1).permute(0, 2, 3, 1) 
        # z_q_m_predict = z_q_m_predict.flatten(0, 1).permute(0, 2, 3, 1) 
        # z_q_b_predict = z_q_b_predict.flatten(0, 1).permute(0, 2, 3, 1)

        # NOTE TODO WONHYEOK: VAE decode function 확인필요 ->
        z_q_predict = self.vae.decode(z_q_1_predict, shapes, output_dict['target_occs'].shape)
        # z_q_predict = self.vae.decode(z_q_t_predict, z_q_b_predict, shapes_b, output_dict['target_occs'].shape)
        # z_q_predict = self.vae.decode(origin_z_q_t[:, mid_frame:end_frame].flatten(0,1).permute(0,2,3,1), origin_z_q_b[:, mid_frame:end_frame].flatten(0, 1).permute(0,2,3,1), shapes_b, output_dict['target_occs'].shape)
        output_dict['logits'] = z_q_predict
        pred = z_q_predict.argmax(dim=-1).detach().cuda()
        output_dict['sem_pred'] = pred
        pred_iou = deepcopy(pred)
        
        pred_iou[pred_iou!=17] = 1
        pred_iou[pred_iou==17] = 0
        output_dict['iou_pred'] = pred_iou

        
        if self.without_all:
            #output_dict['pose_decoded'] = 
            output_dict['sem_pred'] = output_dict['input_occs'][:, 0:1].repeat(1, end_frame-mid_frame, 1, 1, 1)
            pred_iou = deepcopy(output_dict['sem_pred'])
            pred_iou[pred_iou!=17] = 1
            pred_iou[pred_iou==17] = 0
            output_dict['iou_pred'] = pred_iou
            output_dict['pose_decoded'] = torch.tensor([meta['rel_poses'] for meta in input_metas])[:,-1:].unsqueeze(2).repeat(1, end_frame-mid_frame, 3, 1)
        output_dict['time'] = {'encode':t1-t0, 'mid':t2-t1, 'autoreg':t3-t2, 'total':t3-t0, 'per_frame':t1-t0+(t3-t2)/(end_frame-mid_frame)}

        # NOTE WONHYEOK: 1차 검수완 빡센 검토 필요.

        return output_dict
        
        
        
    def decode_pose(self, pose, gt_mode, rel_poses_sumed):
        pose = self.pose_decoder(pose)
        # pose:b, f, 3, 2
        # mode:b, f, 3
        # b, f, 2
        bs, num_frames, num_modes, _ = pose.shape
        #gt_mode_ = gt_mode.unsqueeze(-1).repeat(1, 1, 1, 2)
        pose = pose[gt_mode.bool()].reshape(bs, num_frames, 2)
        pose_decoded = pose.clone().detach()
        '''if not self.delta_input:
            pose = pose+rel_poses_sumed[:, -1:]
            rel_poses_sumed = torch.cat([rel_poses_sumed, pose], dim=1)'''
        pose = torch.cat([pose, gt_mode], dim=-1)
        pose = self.pose_encoder(pose.float())
        return pose, rel_poses_sumed, pose_decoded

    def forward_autoreg(self, x, metas=None, start_frame=0, mid_frame=6,end_frame=12):
        pass

    def sample_with_top_k_top_p_(self, prob, top_k=100, top_p=0.95):
        bf, c, h, w = prob.shape
        logits_BlV = prob.reshape(bf, c, h*w).permute(0, 2, 1) # bf hw c
        if top_k > 0:
            idx_to_remove = logits_BlV < logits_BlV.topk(top_k, largest=True, sorted=False, dim=-1)[0].amin(dim=-1, keepdim=True)
            logits_BlV.masked_fill_(idx_to_remove, -torch.inf)
        if top_p > 0:
            sorted_logits, sorted_idx = logits_BlV.sort(dim=-1, descending=False)
            sorted_idx_to_remove = sorted_logits.softmax(dim=-1).cumsum_(dim=-1) <= (1 - top_p)
            sorted_idx_to_remove[..., -1:] = False
            logits_BlV.masked_fill_(sorted_idx_to_remove.scatter(sorted_idx.ndim - 1, sorted_idx, sorted_idx_to_remove), -torch.inf)
        # sample (have to squeeze cuz torch.multinomial can only be used for 2D tensor)
        return torch.multinomial(logits_BlV.softmax(dim=-1).view(-1, c), num_samples=1, replacement=True, generator=None).view(bf, h, w)
    # NOTE QUESTION WONHYEOK: non-used code below?
    # def generate_inference(self, x):
    #     #import pdb; pdb.set_trace()
    #     bs, F, H, W, D = x.shape
    #     output_dict = {}
    #     output_dict['target_occs'] = x[:, self.offset:]
    #     z, shape = self.vae.forward_encoder(x)
    #     z = self.vae.vqvae.quant_conv(z)
    #     z_q, loss, (perplexity, min_encodings, min_encoding_indices) = self.vae.vqvae.forward_quantizer(z, is_voxel=False)
    #     min_encoding_indices = rearrange(min_encoding_indices, '(b f) h w -> b f h w', b=bs)
    #     output_dict['ce_labels'] = min_encoding_indices[:, self.offset:].detach().flatten(0,1)
    #     z_q = rearrange(z_q, '(b f) c h w -> b f c h w', b=bs)
    #     hidden = None
    #     if self.give_hiddens:
    #         hidden = z_q[:, :self.offset]
    #     z_q_predict = self.transformer(z_q[:, :self.num_frames], hidden=hidden)
    #     z_q_predict = z_q_predict.flatten(0, 1)
    #     output_dict['ce_inputs'] = z_q_predict
    #     z_q_predict = z_q_predict.permute(0, 2, 3, 1)
    #     cata_distribution = dist.Categorical(logits=(z_q_predict-z_q_predict.min())/(z_q_predict.max()-z_q_predict.min()))
    #     import pdb;pdb.set_trace()
    #     z_q_predict = cata_distribution.sample()
    #     z_q_predict = self.vae.vqvae.get_codebook_entry(z_q_predict, shape=None)
    #     z_q_predict = rearrange(z_q_predict, 'bf h w c-> bf c h w')
    #     z_q_predict = self.vae.vqvae.post_quant_conv(z_q_predict)
        
    #     z_q_predict = self.vae.forward_decoder(z_q_predict, shape, output_dict['target_occs'].shape)
    #     output_dict['logits'] = z_q_predict
    #     pred = z_q_predict.argmax(dim=-1).detach().cuda()
    #     output_dict['sem_pred'] = pred
    #     pred_iou = deepcopy(pred)
        
    #     pred_iou[pred_iou!=17] = 1
    #     pred_iou[pred_iou==17] = 0
    #     output_dict['iou_pred'] = pred_iou
    
    #     return output_dict
    
    def compute_planner_metric_stp3(
        self,
        pred_ego_fut_trajs,
        gt_ego_fut_trajs,
        gt_agent_boxes,
        gt_agent_feats,
        fut_valid_flag
    ):
        """Compute planner metric for one sample same as stp3"""
        metric_dict = {
            'plan_L2_1s':0,
            'plan_L2_2s':0,
            'plan_L2_3s':0,
            'plan_obj_col_1s':0,
            'plan_obj_col_2s':0,
            'plan_obj_col_3s':0,
            'plan_obj_box_col_1s':0,
            'plan_obj_box_col_2s':0,
            'plan_obj_box_col_3s':0,
            'plan_L2_1s_single':0,
            'plan_L2_2s_single':0,
            'plan_L2_3s_single':0,
            'plan_obj_col_1s_single':0,
            'plan_obj_col_2s_single':0,
            'plan_obj_col_3s_single':0,
            'plan_obj_box_col_1s_single':0,
            'plan_obj_box_col_2s_single':0,
            'plan_obj_box_col_3s_single':0,
            
        }
        metric_dict['fut_valid_flag'] = fut_valid_flag
        future_second = 3
        assert pred_ego_fut_trajs.shape[0] == 1, 'only support bs=1'
        if self.planning_metric is None:
            self.planning_metric = PlanningMetric()
        segmentation, pedestrian = self.planning_metric.get_label(
            gt_agent_boxes, gt_agent_feats)
        occupancy = torch.logical_or(segmentation, pedestrian)
        for i in range(future_second):
            if fut_valid_flag:
                cur_time = (i+1)*2
                traj_L2 = self.planning_metric.compute_L2(
                    pred_ego_fut_trajs[0, :cur_time].detach().to(gt_ego_fut_trajs.device),
                    gt_ego_fut_trajs[0, :cur_time]
                )
                traj_L2_single = self.planning_metric.compute_L2(
                    pred_ego_fut_trajs[0, cur_time-1:cur_time].detach().to(gt_ego_fut_trajs.device),
                    gt_ego_fut_trajs[0, cur_time-1:cur_time]
                )
                obj_coll, obj_box_coll = self.planning_metric.evaluate_coll(
                    pred_ego_fut_trajs[:, :cur_time].detach(),
                    gt_ego_fut_trajs[:, :cur_time],
                    occupancy)
                obj_coll_single, obj_box_coll_single = self.planning_metric.evaluate_coll(
                    pred_ego_fut_trajs[:, cur_time-1:cur_time].detach(),
                    gt_ego_fut_trajs[:, cur_time-1:cur_time],
                    occupancy[:, cur_time-1:cur_time])
                metric_dict['plan_L2_{}s'.format(i+1)] = traj_L2
                metric_dict['plan_L2_{}s_single'.format(i+1)] = traj_L2_single
                metric_dict['plan_obj_col_{}s'.format(i+1)] = obj_coll.mean().item()
                metric_dict['plan_obj_box_col_{}s'.format(i+1)] = obj_box_coll.mean().item()
                metric_dict['plan_obj_col_{}s_single'.format(i+1)] = obj_coll_single.item()
                metric_dict['plan_obj_box_col_{}s_single'.format(i+1)] = obj_box_coll_single.item()
                
                
            else:
                metric_dict['plan_L2_{}s'.format(i+1)] = 0.0
                metric_dict['plan_L2_{}s_single'.format(i+1)] = 0.0
                metric_dict['plan_obj_col_{}s'.format(i+1)] = 0.0
                metric_dict['plan_obj_box_col_{}s'.format(i+1)] = 0.0
            
        return metric_dict
    
    def autoreg_for_stp3_metric(self, x, metas, 
                                start_frame=0, mid_frame=6,end_frame=12):
        # x: bs, f=12, h, w, d
        output_dict = self.forward_autoreg_with_pose(x, metas, start_frame, mid_frame, end_frame) # 0, 5, 11
        pred_ego_fut_trajs = output_dict['pose_decoded']
        gt_mode = torch.tensor([meta['gt_mode'] for meta in output_dict['output_metas']])
        bs, num_frames, num_modes, _ = pred_ego_fut_trajs.shape
        pred_ego_fut_trajs = pred_ego_fut_trajs[gt_mode.bool()].reshape(bs, num_frames, 2)
        pred_ego_fut_trajs = torch.cumsum(pred_ego_fut_trajs, dim=1).cpu()
        gt_ego_fut_trajs = torch.tensor([meta['rel_poses'] for meta in output_dict['output_metas']])
        gt_ego_fut_trajs = torch.cumsum(gt_ego_fut_trajs, dim=1).cpu()
        assert len(metas) == 1, f'len(metas): {len(metas)}'
        gt_bbox = metas[0]['gt_bboxes_3d']
        gt_attr_labels = torch.tensor(metas[0]['attr_labels'])
        fut_valid_flag = torch.tensor(metas[0]['fut_valid_flag'])
        # import pdb;pdb.set_trace()
        metric_stp3 = self.compute_planner_metric_stp3(
            pred_ego_fut_trajs, gt_ego_fut_trajs, 
            gt_bbox, gt_attr_labels[None], True)
        
        output_dict['metric_stp3'] = metric_stp3
        
        return output_dict
