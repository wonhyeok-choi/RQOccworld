import torch
from torch.nn import functional as F
from torch import nn
from .modules import FFN


from mmengine.registry import MODELS
from mmengine.model import BaseModule
from einops import rearrange


class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, *args, **kwargs):
        return x
class IdentityUnetBlock(nn.Module):
    def __init__(self, shape, in_c, out_c, residual=True):
        super().__init__()
        self.ln = nn.LayerNorm(shape)
        self.conv1 = nn.Conv2d(in_c, out_c, 1, 1, 0)
        #self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.act = nn.ReLU()

    def forward(self, input):
        output = self.ln(input)
        output = self.conv1(output)
        output = self.act(output)
        return output

@MODELS.register_module()
class PlanUAutoRegTransformer(BaseModule):
    def __init__(
        self,
        num_tokens,
        num_frames,
        num_layers,
        img_shape,
        pose_shape,
        tpe_dim=10,
        output_channel=1024,
        channels=[1,2,3],
        ffn_dims=None,
        temporal_attn_layers=1,
        pose_attn_layers=1,
        num_heads=8,
        pose_output_channel=None,
        conditional=True,
        tokens_untouched=False,
        add_aggregate=False,
        learnable_queries=True,
        without_multiscale=False,
        without_spatial_attn=False,
        without_pose_spatial_attn=False,
        without_pose_temporal_attn=False,
        without_temporal_attn=False) -> None:
        super().__init__()
        if without_multiscale:
            assert len(channels) == 2
        self.num_tokens = num_tokens
        self.num_frames = num_frames
        self.num_layers = num_layers
        self.channels = channels
        self.learnable_queries = learnable_queries
        if self.learnable_queries:
            self.queries = nn.Embedding(img_shape[1]*img_shape[2], img_shape[0])
        self.offset = 1 if conditional else 0
        self.temporal_embeddings = nn.Embedding(num_frames + self.offset, tpe_dim)
        if self.learnable_queries:
            self.pose_queries = nn.Embedding(pose_shape[0], pose_shape[1])
        self.pose_temporal_embeddings = nn.Embedding(num_frames + self.offset, tpe_dim)
        self.pose_attn_layers = pose_attn_layers if pose_attn_layers is not None else temporal_attn_layers
        
        self.temporal_attentions_en = nn.ModuleList([])
        self.temporal_attentions_de = nn.ModuleList([])
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        self.pose_attn_en = nn.ModuleList([])
        self.pose_en = nn.ModuleList()
        #self.pose_temporal_attn_en = nn.ModuleList([])
        self.pose_attn_de = nn.ModuleList([])
        self.pose_de = nn.ModuleList()
        self.pose_up = nn.ModuleList()
        #self.pose_temporal_attn_de = nn.ModuleList([])
        
        self.downsamples = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        
        up_down_sample_params = dict(kernel_size=2, stride=2, padding=0)
        self.up_down_sample_params = up_down_sample_params
        self.unfold_params = dict(kernel_size=[1, 2], stride=[1, 2], padding=[0, 0])
        
        C, H, W = img_shape
        layers = len(channels)
        Hs = [H]
        Ws = [W]
        cH = H
        cW = W
        for _ in range(layers-1):
            cH = (cH) // 2
            cW = (cW) // 2
            Hs.append(cH)
            Ws.append(cW)
        pre_c = C
        for channel, cH, cW in zip(channels[0:-1], Hs[0:-1], Ws[0:-1]):
            temporal_attn_layer = nn.ModuleList()
            for i in range(temporal_attn_layers):
                if without_temporal_attn:
                    temporal_attn_layer.append(nn.ModuleList([
                        Identity(),
                        nn.LayerNorm(pre_c),
                        Identity(),
                        nn.LayerNorm(pre_c),
                    ]))
                else:
                    temporal_attn_layer.append(nn.ModuleList([
                        nn.MultiheadAttention(pre_c, num_heads, batch_first=True),
                        nn.LayerNorm(pre_c),
                        FFN(pre_c, pre_c*4),
                        nn.LayerNorm(pre_c)
                    ]))
            self.temporal_attentions_en.append(temporal_attn_layer)
            if without_spatial_attn:
                self.encoders.append(nn.Sequential(IdentityUnetBlock((cH, cW), pre_c, channel), 
                                                   IdentityUnetBlock((cH, cW), channel, channel)))
            else:
                self.encoders.append(nn.Sequential(UnetBlock((cH, cW), pre_c, channel, True),
                                               UnetBlock((cH, cW), channel, channel, True)))
            if without_multiscale:
                self.downsamples.append(Identity())
            else:
                self.downsamples.append(nn.Conv2d(channel, channel, **up_down_sample_params))
            self.pose_en.append(nn.Sequential(
                nn.Linear(pre_c, channel),nn.ReLU(),
                nn.Linear(channel, channel),nn.ReLU(),
            ))
            pose_attn_layer = nn.ModuleList()
            for i in range(pose_attn_layers):
                if without_pose_temporal_attn:
                    pose_attn_layer.append(nn.ModuleList([
                        Identity(),
                        nn.LayerNorm(pre_c),
                        nn.MultiheadAttention(pre_c, num_heads, batch_first=True),
                        nn.LayerNorm(pre_c),
                        FFN(pre_c, pre_c*4),
                        nn.LayerNorm(pre_c)
                ]))
                elif without_pose_spatial_attn:
                    pose_attn_layer.append(nn.ModuleList([
                        nn.MultiheadAttention(pre_c, num_heads, batch_first=True),
                        nn.LayerNorm(pre_c),
                        Identity(),
                        nn.LayerNorm(pre_c),
                        FFN(pre_c, pre_c*4),
                        nn.LayerNorm(pre_c)
                ]))
                else: 
                    pose_attn_layer.append(nn.ModuleList([
                        nn.MultiheadAttention(pre_c, num_heads, batch_first=True),
                        nn.LayerNorm(pre_c),
                        nn.MultiheadAttention(pre_c, num_heads, batch_first=True),
                        nn.LayerNorm(pre_c),
                        FFN(pre_c, pre_c*4),
                        nn.LayerNorm(pre_c)
                    ]))
            self.pose_attn_en.append(pose_attn_layer)
            pre_c = channel
        channel = channels[-1]
        if without_multiscale:
            if without_spatial_attn:
                self.mid = nn.Sequential(
                    IdentityUnetBlock((pre_c, Hs[0], Ws[0]), pre_c, channel, True),
                    IdentityUnetBlock((channel, Hs[0], Ws[0]), channel, channel, True),
                )
            else:
                self.mid = nn.Sequential(
                UnetBlock((pre_c, Hs[0], Ws[0]), pre_c, channel, True),
                UnetBlock((channel, Hs[0], Ws[0]), channel, channel, True),
            )
        else:
            self.mid = nn.Sequential(
                UnetBlock((pre_c, Hs[-1], Ws[-1]), pre_c, channel, True),
                UnetBlock((channel, Hs[-1], Ws[-1]), channel, channel, True),
            )
        self.pose_mid = nn.Sequential(
            nn.Linear(pre_c, channel),nn.ReLU(),
            nn.Linear(channel, channel),nn.ReLU(),
        )
        pre_c = channel
        for channel, cH, cW in zip(channels[-2::-1], Hs[-2::-1], Ws[-2::-1]):
            channel_agg = channel if add_aggregate else channel * 2
            if without_multiscale:
                self.upsamples.append(Identity())
            else:
                self.upsamples.append(nn.ConvTranspose2d(pre_c, channel, **up_down_sample_params))
            temporal_attn_layer = nn.ModuleList()
            for i in range(temporal_attn_layers):
                if without_temporal_attn:
                    temporal_attn_layer.append(nn.ModuleList([
                        Identity(),
                        nn.LayerNorm(channel_agg),
                        Identity(),
                        nn.LayerNorm(channel_agg),
                    ]))
                else:
                    temporal_attn_layer.append(nn.ModuleList([
                        nn.MultiheadAttention(channel_agg, num_heads, batch_first=True),
                        nn.LayerNorm(channel_agg),
                        FFN(channel_agg, channel_agg*4),
                        nn.LayerNorm(channel_agg)
                    ]))
            self.temporal_attentions_de.append(temporal_attn_layer)
            if without_spatial_attn:
                self.decoders.append(nn.Sequential(
                    IdentityUnetBlock((channel_agg, cH,cW), channel_agg, channel, True),
                    IdentityUnetBlock((channel, cH,cW), channel, channel, True)
                ))
            else:
                self.decoders.append(
                    nn.Sequential(
                        UnetBlock((channel_agg, cH,cW),
                                channel_agg, channel, True),
                        UnetBlock((channel, cH,cW),
                                channel,
                                channel,
                                True)
                    )
                )
            pose_attn_layer = nn.ModuleList()
            self.pose_up.append(nn.Linear(pre_c, channel))
            for i in range(pose_attn_layers):
                if without_pose_temporal_attn:
                    pose_attn_layer.append(nn.ModuleList([
                        Identity(),
                        nn.LayerNorm(channel_agg),
                        nn.MultiheadAttention(channel_agg, num_heads, batch_first=True),
                        nn.LayerNorm(channel_agg),
                        FFN(channel_agg, channel_agg*4),
                        nn.LayerNorm(channel_agg)
                ]))
                elif without_pose_spatial_attn:
                    pose_attn_layer.append(nn.ModuleList([
                        nn.MultiheadAttention(channel_agg, num_heads, batch_first=True),
                        nn.LayerNorm(channel_agg),
                        Identity(),
                        nn.LayerNorm(channel_agg),
                        FFN(channel_agg, channel_agg*4),
                        nn.LayerNorm(channel_agg)
                ]))
                else:
                    pose_attn_layer.append(nn.ModuleList([
                        nn.MultiheadAttention(channel_agg, num_heads, batch_first=True),
                        nn.LayerNorm(channel_agg),
                        nn.MultiheadAttention(channel_agg, num_heads, batch_first=True),
                        nn.LayerNorm(channel_agg),
                        FFN(channel_agg, channel_agg*4),
                        nn.LayerNorm(channel_agg)
                    ]))
            self.pose_attn_de.append(pose_attn_layer)
            self.pose_de.append(nn.Sequential(
                nn.Linear(channel_agg, channel),nn.ReLU(),
                nn.Linear(channel, channel),nn.ReLU(),
            ))
            pre_c = channel
        
        self.conv_out = nn.Conv2d(pre_c, output_channel, 3, 1, 1)
        
        self.pose_out = nn.Linear(pre_c, pose_output_channel if pose_output_channel is not None else output_channel)
        
        self.tokens_untouched = tokens_untouched
        
        if tokens_untouched:
            assert all([ch == channels[0] for ch in channels])
            for scale in range(len(channels) - 1):
                num_tokens = self.unfold_params['kernel_size'][scale] ** 2
                attn_mask = torch.zeros(num_frames, num_frames * num_tokens, dtype=torch.bool)
                for i_frame in range(num_frames):
                    start = i_frame * num_tokens + num_tokens if conditional else i_frame * num_tokens
                    attn_mask[i_frame, start:] = True
                self.register_buffer(f'attn_mask_{scale}', attn_mask, False)
        else:
            attn_mask = torch.zeros(num_frames * num_tokens, num_frames * num_tokens, dtype=torch.bool)
            for i_frame in range(num_frames):
                start1 = i_frame * num_tokens
                start2 = start1 + num_tokens if conditional else start1
                attn_mask[start1: (start1 + num_tokens), start2:] = True
            self.register_buffer('attn_mask', attn_mask, False)
    
    def forward(self, tokens, pose_tokens=None):
        #import pdb;pdb.set_trace()
        # tokens: bs, f, c, h, w
        # pose_tokens, bs, f, c
        bs, F, C, H, W = tokens.shape
        assert F == self.num_frames
        tokens = rearrange(tokens, 'b f c h w -> b f h w c')
        if self.learnable_queries:
            queries = self.queries.weight.reshape(1, 1, H, W, C).expand(bs, F, H, W, C)
        else:
            queries = tokens
        queries = queries + self.temporal_embeddings.weight[None, self.offset:, None, None, :].expand(
            bs, -1, H, W, -1)
        tokens = tokens + self.temporal_embeddings.weight[None, :self.num_frames, None, None, :].expand(
            bs, -1, H, W, -1)
        
        encoder_outs_pose_tokens = [None for _ in range(len(self.pose_attn_en))]
        encoder_outs_pose_queries = [None for _ in range(len(self.pose_attn_en))]

        if pose_tokens is not None:
            if self.learnable_queries:
                pose_queries = self.pose_queries.weight.reshape(1, 1, C).expand(bs, F, C)
            else:
                pose_queries = pose_tokens
            pose_queries = pose_queries + self.pose_temporal_embeddings.weight[None, self.offset:, :].expand(
                bs, -1, -1)
            pose_tokens = pose_tokens + self.pose_temporal_embeddings.weight[None, :self.num_frames, :].expand(
                bs, -1, -1)
            encoder_outs_pose_tokens = []
            encoder_outs_pose_queries = []

        encoder_outs_tokens = []
        encoder_outs_queries = []
        
        for temporal_attn, encoder, down, pose_attn_en, pose_en in zip(self.temporal_attentions_en, self.encoders, self.downsamples, self.pose_attn_en, self.pose_en):
            b, f, h, w, c = tokens.shape
            
            if pose_tokens is not None:
                for pose_temporal_attn, pose_temporal_norm, spatial_attn, spatial_norm, ffn, ffn_norm in pose_attn_en:
                    pose_queries = pose_queries + pose_temporal_attn(pose_queries, pose_tokens, pose_tokens, need_weights=False, attn_mask=self.attn_mask)[0]
                    pose_queries = pose_temporal_norm(pose_queries)
                    #b, f, h, w, c = queries.shape
                    pose_queries = rearrange(pose_queries, 'b f c -> (b f) 1 c')
                    queries = rearrange(queries, 'b f h w c -> (b f) (h w) c')
                    pose_queries = pose_queries + spatial_attn(pose_queries, queries, queries, need_weights=False, attn_mask=None)[0]
                    pose_queries = spatial_norm(pose_queries)
                    
                    pose_queries = pose_queries + ffn(pose_queries)
                    pose_queries = ffn_norm(pose_queries)
                    pose_queries = rearrange(pose_queries, '(b f) 1 c -> b f c', b=b, f=f)
                    queries = rearrange(queries, '(b f) (h w) c -> b f h w c', b=b, f=f, h=h, w=w)
                    
                pose_queries = pose_en(pose_queries)
                pose_tokens = pose_en(pose_tokens)
                encoder_outs_pose_queries.append(pose_queries)
                encoder_outs_pose_tokens.append(pose_tokens)
            
            queries = rearrange(queries, 'b f h w c -> (b h w) f c')
            tokens = rearrange(tokens, 'b f h w c -> (b h w) f c')
            #queries = rearrange(queries, 'b f h w c -> (b h w) f c')
            for cross_attn, cross_norm, ffn, ffn_norm in temporal_attn:
                queries = queries + cross_attn(queries, tokens, tokens, need_weights=False, attn_mask=self.attn_mask)[0]
                queries = cross_norm(queries)
                
                queries = queries + ffn(queries)
                queries = ffn_norm(queries)
            
            queries = rearrange(queries, '(b h w) f c -> (b f) c h w', b=b, h=h, w=w)
            tokens = rearrange(tokens, '(b h w) f c -> (b f) c h w', b=b, h=h, w=w)
            queries = encoder(queries)
            tokens = encoder(tokens)
            encoder_outs_tokens.append(tokens)
            encoder_outs_queries.append(queries)
            queries = down(queries)
            tokens = down(tokens)
            queries = rearrange(queries, '(b f) c h w -> b f h w c', b=b, f=f)
            tokens = rearrange(tokens, '(b f) c h w -> b f h w c', b=b, f=f) 
        b, f, h, w, c = queries.shape
        queries = rearrange(queries, 'b f h w c -> (b f) c h w')
        tokens = rearrange(tokens, 'b f h w c -> (b f) c h w')
        queries = self.mid(queries)
        tokens = self.mid(tokens)
        
        if pose_tokens is not None:
            pose_queries = self.pose_mid(pose_queries)
            pose_tokens = self.pose_mid(pose_tokens)
        
        # queries = rearrange(queries, '(b f) c h w -> b f h w c', b=b, f=f)
        # tokens = rearrange(tokens, '(b f) c h w -> b f h w c', b=b, f=f)
        for temporal_attn, decoder, up, encoder_out_queries, encoder_out_tokens, pose_attn_de, pose_de_, encoder_out_pose_queries, encoder_out_pose_tokens, pose_up in zip(self.temporal_attentions_de,
                                                                        self.decoders, self.upsamples, encoder_outs_queries[::-1],
                                                                        encoder_outs_tokens[::-1], self.pose_attn_de, self.pose_de,
                                                                        encoder_outs_pose_queries[::-1], encoder_outs_pose_tokens[::-1], self.pose_up):
            queries = up(queries)
            tokens = up(tokens)
            
            pad_x_queries = encoder_out_queries.shape[2] - queries.shape[2]
            pad_y_queries = encoder_out_queries.shape[3] - queries.shape[3]
            queries = nn.functional.pad(queries, (pad_x_queries//2, pad_x_queries-pad_x_queries//2,
                                      pad_y_queries//2, pad_y_queries-pad_y_queries//2))
            pad_x_tokens = encoder_out_tokens.shape[2] - tokens.shape[2]
            pad_y_tokens = encoder_out_tokens.shape[3] - tokens.shape[3]
            tokens = nn.functional.pad(tokens, (pad_x_tokens//2, pad_x_tokens-pad_x_tokens//2,
                                      pad_y_tokens//2, pad_y_tokens-pad_y_tokens//2))
            queries = torch.cat([queries, encoder_out_queries], dim=1)
            tokens = torch.cat([tokens, encoder_out_tokens], dim=1)
            c, h, w = queries.shape[-3:]
            queries = rearrange(queries, '(b f) c h w -> (b h w) f c', b=b, f=f)
            tokens = rearrange(tokens, '(b f) c h w -> (b h w) f c', b=b, f=f)
            for cross_attn, cross_norm, ffn, ffn_norm in temporal_attn:
                queries = queries + cross_attn(queries, tokens, tokens, need_weights=False, attn_mask=self.attn_mask)[0]
                queries = cross_norm(queries)
                
                queries = queries + ffn(queries)
                queries = ffn_norm(queries)
            queries = rearrange(queries, '(b h w) f c -> (b f) c h w', b=b, h=h, w=w)
            tokens = rearrange(tokens, '(b h w) f c -> (b f) c h w', b=b, h=h, w=w)
            
            if pose_tokens is not None:
                pose_queries = pose_up(pose_queries)
                pose_tokens = pose_up(pose_tokens)
                pose_queries = torch.cat([pose_queries, encoder_out_pose_queries], dim=2)
                pose_tokens = torch.cat([pose_tokens, encoder_out_pose_tokens], dim=2)
                
                for pose_temporal_attn, pose_temporal_norm, spatial_attn, spatial_norm, ffn, ffn_norm in pose_attn_de:
                    pose_queries = pose_queries + pose_temporal_attn(pose_queries, pose_tokens, pose_tokens, need_weights=False, attn_mask=self.attn_mask)[0]
                    pose_queries = pose_temporal_norm(pose_queries)
                    #b, f, h, w, c = queries.shape
                    pose_queries = rearrange(pose_queries, 'b f c -> (b f) 1 c')
                    #queries = rearrange(queries, 'b f h w c -> (b f) (h w) c')
                    queries = rearrange(queries, '(b f) c h w -> (b f) (h w) c', b=b, f=f, h=h, w=w)
                    pose_queries = pose_queries + spatial_attn(pose_queries, queries, queries, need_weights=False, attn_mask=None)[0]
                    pose_queries = spatial_norm(pose_queries)
                    
                    pose_queries = pose_queries + ffn(pose_queries)
                    pose_queries = ffn_norm(pose_queries)
                    queries = rearrange(queries, '(b f) (h w) c -> (b f) c h w', b=b, f=f, h=h, w=w)
                    pose_queries = rearrange(pose_queries, '(b f) 1 c -> b f c', b=b, f=f)
                pose_queries = pose_de_(pose_queries)
                pose_tokens = pose_de_(pose_tokens)
            queries = decoder(queries)
            tokens = decoder(tokens)
            
        queries = self.conv_out(queries)
        queries = rearrange(queries, '(b f) c h w -> b f c h w', b=b, f=f)
        if pose_tokens is not None:
            pose_queries = self.pose_out(pose_queries)
            return queries, pose_queries

        return queries

    def forward_autoreg(self, tokens, pose_tokens, start_frame=0, mid_frame=6, end_frame=12):
        tokens = tokens[:, start_frame:mid_frame]
        pose_tokens = pose_tokens[:, start_frame:mid_frame]
        for i in range(mid_frame, end_frame):
            token, pose_token = self.forward_autoreg_step(tokens, pose_tokens, start_frame, i)
            
            tokens = torch.cat([tokens, token[:, -1:]], dim=1)
            pose_tokens = torch.cat([pose_tokens, pose_token[:, -1:]], dim=1)
        b, f, c, h, w = tokens.shape
        queries = rearrange(tokens, 'b f c h w -> (b f) c h w')
        queries = self.conv_out(queries)
        pose_queries = self.pose_out(pose_tokens)
        queries = rearrange(queries, '(b f) c h w -> b f c h w', b=b, f=f)
        
        return queries[:,mid_frame:end_frame], pose_queries[:,mid_frame:end_frame]

    def forward_autoreg_step(self, tokens, pose_tokens=None, fusion=None, start_frame=0, mid_frame=6):
        bs, F, C, H, W = tokens.shape
        #assert F == self.num_frames
        tokens = tokens[:, start_frame:mid_frame]
        if pose_tokens is not None:
            pose_tokens = pose_tokens[:, start_frame:mid_frame]
        bs, F, C, H, W = tokens.shape
        tokens = rearrange(tokens, 'b f c h w -> b f h w c')
        if self.learnable_queries:
            queries = self.queries.weight.reshape(1, 1, H, W, C).expand(bs, F, H, W, C)
        else:
            queries = tokens
        queries = queries + self.temporal_embeddings.weight[None, self.offset:F+self.offset, None, None, :].expand(
            bs, -1, H, W, -1)
        tokens = tokens + self.temporal_embeddings.weight[None, :F, None, None, :].expand(
            bs, -1, H, W, -1)

        encoder_outs_pose_tokens = [None for _ in range(len(self.pose_attn_en))]
        encoder_outs_pose_queries = [None for _ in range(len(self.pose_attn_en))]

        if pose_tokens is not None:
            if self.learnable_queries:
                pose_queries = self.pose_queries.weight.reshape(1, 1, C).expand(bs, F, C)
            else:
                pose_queries = pose_tokens
            pose_queries = pose_queries + self.pose_temporal_embeddings.weight[None, self.offset:F+self.offset, :].expand(
                bs, -1, -1)
            pose_tokens = pose_tokens + self.pose_temporal_embeddings.weight[None, :F, :].expand(
                bs, -1, -1)
            encoder_outs_pose_tokens = []
            encoder_outs_pose_queries = []

        encoder_outs_tokens = []
        encoder_outs_queries = []
        
        for temporal_attn, encoder, down, pose_attn_en, pose_en in zip(self.temporal_attentions_en, self.encoders, self.downsamples, self.pose_attn_en, self.pose_en):
            b, f, h, w, c = tokens.shape
            
            if pose_tokens is not None:
                for pose_temporal_attn, pose_temporal_norm, spatial_attn, spatial_norm, ffn, ffn_norm in pose_attn_en:
                    pose_queries = pose_queries + pose_temporal_attn(pose_queries, pose_tokens, pose_tokens, need_weights=False, attn_mask=self.attn_mask[:f,:f])[0]
                    pose_queries = pose_temporal_norm(pose_queries)
                    #b, f, h, w, c = queries.shape
                    pose_queries = rearrange(pose_queries, 'b f c -> (b f) 1 c')
                    queries = rearrange(queries, 'b f h w c -> (b f) (h w) c')
                    pose_queries = pose_queries + spatial_attn(pose_queries, queries, queries, need_weights=False, attn_mask=None)[0]
                    pose_queries = spatial_norm(pose_queries)
                    
                    pose_queries = pose_queries + ffn(pose_queries)
                    pose_queries = ffn_norm(pose_queries)
                    pose_queries = rearrange(pose_queries, '(b f) 1 c -> b f c', b=b, f=f)
                    queries = rearrange(queries, '(b f) (h w) c -> b f h w c', b=b, f=f, h=h, w=w)
                    
                pose_queries = pose_en(pose_queries)
                pose_tokens = pose_en(pose_tokens)
                encoder_outs_pose_queries.append(pose_queries)
                encoder_outs_pose_tokens.append(pose_tokens)
            
            queries = rearrange(queries, 'b f h w c -> (b h w) f c')
            tokens = rearrange(tokens, 'b f h w c -> (b h w) f c')
            #queries = rearrange(queries, 'b f h w c -> (b h w) f c')
            for cross_attn, cross_norm, ffn, ffn_norm in temporal_attn:
                queries = queries + cross_attn(queries, tokens, tokens, need_weights=False, attn_mask=self.attn_mask[:f, :f])[0]
                queries = cross_norm(queries)
                
                queries = queries + ffn(queries)
                queries = ffn_norm(queries)
            
            queries = rearrange(queries, '(b h w) f c -> (b f) c h w', b=b, h=h, w=w)
            tokens = rearrange(tokens, '(b h w) f c -> (b f) c h w', b=b, h=h, w=w)
            queries = encoder(queries)
            tokens = encoder(tokens)
            encoder_outs_tokens.append(tokens)
            encoder_outs_queries.append(queries)
            queries = down(queries)
            tokens = down(tokens)
            queries = rearrange(queries, '(b f) c h w -> b f h w c', b=b, f=f)
            tokens = rearrange(tokens, '(b f) c h w -> b f h w c', b=b, f=f) 
        b, f, h, w, c = queries.shape
        queries = rearrange(queries, 'b f h w c -> (b f) c h w')
        tokens = rearrange(tokens, 'b f h w c -> (b f) c h w')
        queries = self.mid(queries)
        tokens = self.mid(tokens)
        
        if pose_tokens is not None:
            pose_queries = self.pose_mid(pose_queries)
            pose_tokens = self.pose_mid(pose_tokens)
        
        # queries = rearrange(queries, '(b f) c h w -> b f h w c', b=b, f=f)
        # tokens = rearrange(tokens, '(b f) c h w -> b f h w c', b=b, f=f)
        for temporal_attn, decoder, up, encoder_out_queries, encoder_out_tokens, pose_attn_de, pose_de_, encoder_out_pose_queries, encoder_out_pose_tokens, pose_up in zip(self.temporal_attentions_de,
                                                                        self.decoders, self.upsamples, encoder_outs_queries[::-1],
                                                                        encoder_outs_tokens[::-1], self.pose_attn_de, self.pose_de,
                                                                        encoder_outs_pose_queries[::-1], encoder_outs_pose_tokens[::-1], self.pose_up):
            queries = up(queries)
            tokens = up(tokens)
            
            pad_x_queries = encoder_out_queries.shape[2] - queries.shape[2]
            pad_y_queries = encoder_out_queries.shape[3] - queries.shape[3]
            queries = nn.functional.pad(queries, (pad_x_queries//2, pad_x_queries-pad_x_queries//2,
                                      pad_y_queries//2, pad_y_queries-pad_y_queries//2))
            pad_x_tokens = encoder_out_tokens.shape[2] - tokens.shape[2]
            pad_y_tokens = encoder_out_tokens.shape[3] - tokens.shape[3]
            tokens = nn.functional.pad(tokens, (pad_x_tokens//2, pad_x_tokens-pad_x_tokens//2,
                                      pad_y_tokens//2, pad_y_tokens-pad_y_tokens//2))
            queries = torch.cat([queries, encoder_out_queries], dim=1)
            tokens = torch.cat([tokens, encoder_out_tokens], dim=1)
            c, h, w = queries.shape[-3:]
            queries = rearrange(queries, '(b f) c h w -> (b h w) f c', b=b, f=f)
            tokens = rearrange(tokens, '(b f) c h w -> (b h w) f c', b=b, f=f)
            for cross_attn, cross_norm, ffn, ffn_norm in temporal_attn:
                queries = queries + cross_attn(queries, tokens, tokens, need_weights=False, attn_mask=self.attn_mask[:f,:f])[0]
                queries = cross_norm(queries)
                
                queries = queries + ffn(queries)
                queries = ffn_norm(queries)
            queries = rearrange(queries, '(b h w) f c -> (b f) c h w', b=b, h=h, w=w)
            tokens = rearrange(tokens, '(b h w) f c -> (b f) c h w', b=b, h=h, w=w)
            
            if pose_tokens is not None:
                pose_queries = pose_up(pose_queries)
                pose_tokens = pose_up(pose_tokens)
                pose_queries = torch.cat([pose_queries, encoder_out_pose_queries], dim=2)
                pose_tokens = torch.cat([pose_tokens, encoder_out_pose_tokens], dim=2)
                
                for pose_temporal_attn, pose_temporal_norm, spatial_attn, spatial_norm, ffn, ffn_norm in pose_attn_de:
                    pose_queries = pose_queries + pose_temporal_attn(pose_queries, pose_tokens, pose_tokens, need_weights=False, attn_mask=self.attn_mask[:f, :f])[0]
                    pose_queries = pose_temporal_norm(pose_queries)
                    #b, f, h, w, c = queries.shape
                    pose_queries = rearrange(pose_queries, 'b f c -> (b f) 1 c')
                    #queries = rearrange(queries, 'b f h w c -> (b f) (h w) c')
                    queries = rearrange(queries, '(b f) c h w -> (b f) (h w) c', b=b, f=f, h=h, w=w)
                    pose_queries = pose_queries + spatial_attn(pose_queries, queries, queries, need_weights=False, attn_mask=None)[0]
                    pose_queries = spatial_norm(pose_queries)
                    
                    pose_queries = pose_queries + ffn(pose_queries)
                    pose_queries = ffn_norm(pose_queries)
                    queries = rearrange(queries, '(b f) (h w) c -> (b f) c h w', b=b, f=f, h=h, w=w)
                    pose_queries = rearrange(pose_queries, '(b f) 1 c -> b f c', b=b, f=f)
                pose_queries = pose_de_(pose_queries)
                pose_tokens = pose_de_(pose_tokens)
            queries = decoder(queries)
            tokens = decoder(tokens)
        
        queries = self.conv_out(queries)
        queries = rearrange(queries, '(b f) c h w -> b f c h w', b=b, f=f)
        if pose_tokens is not None:
            pose_queries = self.pose_out(pose_queries)
            return queries, pose_queries
        
        return queries


@MODELS.register_module()
class PlanUAutoRegTransformerResidual(BaseModule):
    def __init__(
        self,
        num_tokens,
        num_frames,
        num_layers,
        img_shape,
        pose_shape,
        tpe_dim=10,
        output_channel=1024,
        channels=[1,2,3],
        ffn_dims=None,
        temporal_attn_layers=1,
        pose_attn_layers=1,
        num_heads=8,
        pose_output_channel=None,
        conditional=True,
        tokens_untouched=False,
        add_aggregate=False,
        learnable_queries=True,
        without_multiscale=False,
        without_spatial_attn=False,
        without_pose_spatial_attn=False,
        without_pose_temporal_attn=False,
        without_temporal_attn=False) -> None:
        super().__init__()
        if without_multiscale:
            assert len(channels) == 2
        self.num_tokens = num_tokens
        self.num_frames = num_frames
        self.num_layers = num_layers
        self.channels = channels
        self.learnable_queries = learnable_queries
        if self.learnable_queries:
            self.queries = nn.Embedding(img_shape[1]*img_shape[2], img_shape[0])
        self.offset = 1 if conditional else 0
        self.temporal_embeddings = nn.Embedding(num_frames + self.offset, tpe_dim)
        self.depth_embeddings = nn.Embedding(4, tpe_dim) # NOTE HARDCODED
        if self.learnable_queries:
            self.pose_queries = nn.Embedding(pose_shape[0], pose_shape[1])
        self.pose_temporal_embeddings = nn.Embedding(num_frames + self.offset, tpe_dim)
        self.pose_attn_layers = pose_attn_layers if pose_attn_layers is not None else temporal_attn_layers
        
        self.temporal_attentions_en = nn.ModuleList([])
        self.temporal_attentions_de = nn.ModuleList([])
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        self.pose_attn_en = nn.ModuleList([])
        self.pose_en = nn.ModuleList()
        #self.pose_temporal_attn_en = nn.ModuleList([])
        self.pose_attn_de = nn.ModuleList([])
        self.pose_de = nn.ModuleList()
        self.pose_up = nn.ModuleList()
        #self.pose_temporal_attn_de = nn.ModuleList([])
        
        self.downsamples = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        
        up_down_sample_params = dict(kernel_size=2, stride=2, padding=0)
        self.up_down_sample_params = up_down_sample_params
        self.unfold_params = dict(kernel_size=[1, 2], stride=[1, 2], padding=[0, 0])
        
        C, H, W = img_shape
        layers = len(channels)
        Hs = [H]
        Ws = [W]
        cH = H
        cW = W
        for _ in range(layers-1):
            cH = (cH) // 2
            cW = (cW) // 2
            Hs.append(cH)
            Ws.append(cW)
        pre_c = C
        """ NOTE add attnstack for depth attention"""
        self.depth_attention_layers = nn.ModuleList([
                            nn.MultiheadAttention(pre_c, num_heads, batch_first=True),
                            nn.LayerNorm(pre_c),
                            FFN(pre_c, pre_c*4),
                            nn.LayerNorm(pre_c)
                        ])

        """ NOTE ENDED """
        for channel, cH, cW in zip(channels[0:-1], Hs[0:-1], Ws[0:-1]):
            temporal_attn_layer = nn.ModuleList()
            for i in range(temporal_attn_layers):
                if without_temporal_attn:
                    temporal_attn_layer.append(nn.ModuleList([
                        Identity(),
                        nn.LayerNorm(pre_c),
                        Identity(),
                        nn.LayerNorm(pre_c),
                    ]))
                else:
                    temporal_attn_layer.append(nn.ModuleList([
                        nn.MultiheadAttention(pre_c, num_heads, batch_first=True),
                        nn.LayerNorm(pre_c),
                        FFN(pre_c, pre_c*4),
                        nn.LayerNorm(pre_c)
                    ]))
            self.temporal_attentions_en.append(temporal_attn_layer)
            if without_spatial_attn:
                self.encoders.append(nn.Sequential(IdentityUnetBlock((cH, cW), pre_c, channel), 
                                                   IdentityUnetBlock((cH, cW), channel, channel)))
            else:
                self.encoders.append(nn.Sequential(UnetBlock((cH, cW), pre_c, channel, True),
                                               UnetBlock((cH, cW), channel, channel, True)))
            if without_multiscale:
                self.downsamples.append(Identity())
            else:
                self.downsamples.append(nn.Conv2d(channel, channel, **up_down_sample_params))
            self.pose_en.append(nn.Sequential(
                nn.Linear(pre_c, channel),nn.ReLU(),
                nn.Linear(channel, channel),nn.ReLU(),
            ))
            pose_attn_layer = nn.ModuleList()
            for i in range(pose_attn_layers):
                if without_pose_temporal_attn:
                    pose_attn_layer.append(nn.ModuleList([
                        Identity(),
                        nn.LayerNorm(pre_c),
                        nn.MultiheadAttention(pre_c, num_heads, batch_first=True),
                        nn.LayerNorm(pre_c),
                        FFN(pre_c, pre_c*4),
                        nn.LayerNorm(pre_c)
                ]))
                elif without_pose_spatial_attn:
                    pose_attn_layer.append(nn.ModuleList([
                        nn.MultiheadAttention(pre_c, num_heads, batch_first=True),
                        nn.LayerNorm(pre_c),
                        Identity(),
                        nn.LayerNorm(pre_c),
                        FFN(pre_c, pre_c*4),
                        nn.LayerNorm(pre_c)
                ]))
                else: 
                    pose_attn_layer.append(nn.ModuleList([
                        nn.MultiheadAttention(pre_c, num_heads, batch_first=True),
                        nn.LayerNorm(pre_c),
                        nn.MultiheadAttention(pre_c, num_heads, batch_first=True),
                        nn.LayerNorm(pre_c),
                        FFN(pre_c, pre_c*4),
                        nn.LayerNorm(pre_c)
                    ]))
            self.pose_attn_en.append(pose_attn_layer)
            pre_c = channel
        channel = channels[-1]
        if without_multiscale:
            if without_spatial_attn:
                self.mid = nn.Sequential(
                    IdentityUnetBlock((pre_c, Hs[0], Ws[0]), pre_c, channel, True),
                    IdentityUnetBlock((channel, Hs[0], Ws[0]), channel, channel, True),
                )
            else:
                self.mid = nn.Sequential(
                UnetBlock((pre_c, Hs[0], Ws[0]), pre_c, channel, True),
                UnetBlock((channel, Hs[0], Ws[0]), channel, channel, True),
            )
        else:
            self.mid = nn.Sequential(
                UnetBlock((pre_c, Hs[-1], Ws[-1]), pre_c, channel, True),
                UnetBlock((channel, Hs[-1], Ws[-1]), channel, channel, True),
            )
        self.pose_mid = nn.Sequential(
            nn.Linear(pre_c, channel),nn.ReLU(),
            nn.Linear(channel, channel),nn.ReLU(),
        )
        pre_c = channel
        for channel, cH, cW in zip(channels[-2::-1], Hs[-2::-1], Ws[-2::-1]):
            channel_agg = channel if add_aggregate else channel * 2
            if without_multiscale:
                self.upsamples.append(Identity())
            else:
                self.upsamples.append(nn.ConvTranspose2d(pre_c, channel, **up_down_sample_params))
            temporal_attn_layer = nn.ModuleList()
            for i in range(temporal_attn_layers):
                if without_temporal_attn:
                    temporal_attn_layer.append(nn.ModuleList([
                        Identity(),
                        nn.LayerNorm(channel_agg),
                        Identity(),
                        nn.LayerNorm(channel_agg),
                    ]))
                else:
                    temporal_attn_layer.append(nn.ModuleList([
                        nn.MultiheadAttention(channel_agg, num_heads, batch_first=True),
                        nn.LayerNorm(channel_agg),
                        FFN(channel_agg, channel_agg*4),
                        nn.LayerNorm(channel_agg)
                    ]))
            self.temporal_attentions_de.append(temporal_attn_layer)
            if without_spatial_attn:
                self.decoders.append(nn.Sequential(
                    IdentityUnetBlock((channel_agg, cH,cW), channel_agg, channel, True),
                    IdentityUnetBlock((channel, cH,cW), channel, channel, True)
                ))
            else:
                self.decoders.append(
                    nn.Sequential(
                        UnetBlock((channel_agg, cH,cW),
                                channel_agg, channel, True),
                        UnetBlock((channel, cH,cW),
                                channel,
                                channel,
                                True)
                    )
                )
            pose_attn_layer = nn.ModuleList()
            self.pose_up.append(nn.Linear(pre_c, channel))
            for i in range(pose_attn_layers):
                if without_pose_temporal_attn:
                    pose_attn_layer.append(nn.ModuleList([
                        Identity(),
                        nn.LayerNorm(channel_agg),
                        nn.MultiheadAttention(channel_agg, num_heads, batch_first=True),
                        nn.LayerNorm(channel_agg),
                        FFN(channel_agg, channel_agg*4),
                        nn.LayerNorm(channel_agg)
                ]))
                elif without_pose_spatial_attn:
                    pose_attn_layer.append(nn.ModuleList([
                        nn.MultiheadAttention(channel_agg, num_heads, batch_first=True),
                        nn.LayerNorm(channel_agg),
                        Identity(),
                        nn.LayerNorm(channel_agg),
                        FFN(channel_agg, channel_agg*4),
                        nn.LayerNorm(channel_agg)
                ]))
                else:
                    pose_attn_layer.append(nn.ModuleList([
                        nn.MultiheadAttention(channel_agg, num_heads, batch_first=True),
                        nn.LayerNorm(channel_agg),
                        nn.MultiheadAttention(channel_agg, num_heads, batch_first=True),
                        nn.LayerNorm(channel_agg),
                        FFN(channel_agg, channel_agg*4),
                        nn.LayerNorm(channel_agg)
                    ]))
            self.pose_attn_de.append(pose_attn_layer)
            self.pose_de.append(nn.Sequential(
                nn.Linear(channel_agg, channel),nn.ReLU(),
                nn.Linear(channel, channel),nn.ReLU(),
            ))
            pre_c = channel
        
        self.conv_out = nn.Conv2d(pre_c, output_channel, 3, 1, 1)
        
        self.pose_out = nn.Linear(pre_c, pose_output_channel if pose_output_channel is not None else output_channel)
        
        self.tokens_untouched = tokens_untouched
        
        if tokens_untouched:
            assert all([ch == channels[0] for ch in channels])
            for scale in range(len(channels) - 1):
                num_tokens = self.unfold_params['kernel_size'][scale] ** 2
                attn_mask = torch.zeros(num_frames, num_frames * num_tokens, dtype=torch.bool)
                for i_frame in range(num_frames):
                    start = i_frame * num_tokens + num_tokens if conditional else i_frame * num_tokens
                    attn_mask[i_frame, start:] = True
                self.register_buffer(f'attn_mask_{scale}', attn_mask, False)
        else:
            attn_mask = torch.zeros(num_frames * num_tokens * 4, num_frames * num_tokens * 4, dtype=torch.bool)
            for i_frame in range(num_frames):
                start1 = i_frame * num_tokens * 4
                start2 = start1 + num_tokens * 4 if conditional else start1
                attn_mask[start1: (start1 + num_tokens * 4), start2:] = True
            self.register_buffer('attn_mask', attn_mask, False)
            # attn_mask = torch.zeros(num_frames * num_tokens * 4, num_frames * num_tokens * 4, dtype=torch.bool)
            # for i_frame in range(num_frames * 4):
            #     start1 = i_frame * num_tokens
            #     start2 = start1 + num_tokens if conditional else start1
            #     attn_mask[start1: (start1 + num_tokens), start2:] = True
            # self.register_buffer('attn_mask', attn_mask, False)
        
        pose_attn_mask = torch.zeros(num_frames * num_tokens, num_frames * num_tokens, dtype=torch.bool)
        for i_frame in range(num_frames):
            start1 = i_frame * num_tokens
            start2 = start1 + num_tokens if conditional else start1
            attn_mask[start1: (start1 + num_tokens), start2:] = True
        self.register_buffer('pose_attn_mask', pose_attn_mask, False)

        # else:
        #     attn_mask = torch.zeros(num_frames * num_tokens, num_frames * num_tokens, dtype=torch.bool)
        #     for i_frame in range(num_frames):
        #         start1 = i_frame * num_tokens
        #         start2 = start1 + num_tokens if conditional else start1
        #         attn_mask[start1: (start1 + num_tokens), start2:] = True
        #     self.register_buffer('attn_mask', attn_mask, False)

        # NOTE
        # depth_attn_mask = torch.zeros(num_frames * 4, num_frames * 4, dtype=torch.bool)
        # for i_frame in range(num_frames):
        #     start1 = i_frame * 4
        #     start2 = start1 + 4
        #     depth_attn_mask[start1: (start1 + 4), start2:] = True
        # self.register_buffer('depth_attn_mask', depth_attn_mask, False)

        # NOTE CROSSATTN version
        # depth_attn_mask = torch.zeros(num_frames * 4, num_frames, dtype=torch.bool)
        # for i_frame in range(num_frames):
        #     start1 = i_frame * 4
        #     start2 = i_frame + 1
        #     depth_attn_mask[start1: (start1 + 4), start2:] = True
        # self.register_buffer('depth_attn_mask', depth_attn_mask, False)
    
    def forward(self, tokens, res_tokens, pose_tokens=None):
        #import pdb;pdb.set_trace()
        # tokens: bs, f, c, h, w
        # res_tokens: bs, f, c, h, w, d (cumsumed)
        # pose_tokens, bs, f, c
        bs, F, C, H, W, D = res_tokens.shape
        # bs, F, C, H, W = tokens.shape
        assert F == self.num_frames
        tokens = rearrange(res_tokens, 'b f c h w d -> b f d h w c')
        # tokens = rearrange(tokens, 'b f c h w -> b f h w c')
        # if self.learnable_queries:
        #     queries = self.queries.weight.reshape(1, 1, H, W, C).expand(bs, F, H, W, C)
        # else:
        queries = tokens
        """ actually it is not temporal embedding -> temporal-depth embedding """
        # NOTE temporal embedding
        queries = queries + self.temporal_embeddings.weight[None, self.offset:, None, None, None, :].expand(
            bs, -1, D, H, W, -1)
        tokens = tokens + self.temporal_embeddings.weight[None, :self.num_frames, None, None, None, :].expand(
            bs, -1, D, H, W, -1)
        # NOTE depth embedding
        # print("queries.shape:", queries.shape)
        # print("embeddings.shape:", self.depth_embeddings.weight[None, None, :, None, None, :].expand(bs, F, -1, H, W, -1).shape)
        queries = queries + self.depth_embeddings.weight[None, None, :, None, None, :].expand(
            bs, F, -1, H, W, -1)
        tokens = tokens + self.depth_embeddings.weight[None, None, :, None, None, :].expand(
            bs, F, -1, H, W, -1)
        # queries = queries + self.temporal_embeddings.weight[None, self.offset:, None, None, :].expand(
        #     bs, -1, H, W, -1)
        # tokens = tokens + self.temporal_embeddings.weight[None, :self.num_frames, None, None, :].expand(
        #     bs, -1, H, W, -1)
        queries = rearrange(queries, 'b f d h w c -> b f h w c d')
        tokens = rearrange(tokens, 'b f d h w c -> b f h w c d')
        
        encoder_outs_pose_tokens = [None for _ in range(len(self.pose_attn_en))]
        encoder_outs_pose_queries = [None for _ in range(len(self.pose_attn_en))]

        if pose_tokens is not None:
            if self.learnable_queries:
                pose_queries = self.pose_queries.weight.reshape(1, 1, C).expand(bs, F, C)
            else:
                pose_queries = pose_tokens
            pose_queries = pose_queries + self.pose_temporal_embeddings.weight[None, self.offset:, :].expand(
                bs, -1, -1)
            pose_tokens = pose_tokens + self.pose_temporal_embeddings.weight[None, :self.num_frames, :].expand(
                bs, -1, -1)
            encoder_outs_pose_tokens = []
            encoder_outs_pose_queries = []

        encoder_outs_tokens = []
        encoder_outs_queries = []
        
        for temporal_attn, encoder, down, pose_attn_en, pose_en in zip(self.temporal_attentions_en, self.encoders, self.downsamples, self.pose_attn_en, self.pose_en):
            b, f, h, w, c, d = tokens.shape
            
            if pose_tokens is not None:
                for pose_temporal_attn, pose_temporal_norm, spatial_attn, spatial_norm, ffn, ffn_norm in pose_attn_en:
                    pose_queries = pose_queries + pose_temporal_attn(pose_queries, pose_tokens, pose_tokens, need_weights=False, attn_mask=self.pose_attn_mask)[0]
                    pose_queries = pose_temporal_norm(pose_queries)
                    pose_queries = rearrange(pose_queries, 'b f c -> (b f) 1 c')
                    queries = rearrange(queries, 'b f h w c d -> (b f) (h w d) c')
                    # queries = rearrange(queries, 'b f h w c -> (b f) (h w) c')
                    pose_queries = pose_queries + spatial_attn(pose_queries, queries, queries, need_weights=False, attn_mask=None)[0]
                    pose_queries = spatial_norm(pose_queries)
                    
                    pose_queries = pose_queries + ffn(pose_queries)
                    pose_queries = ffn_norm(pose_queries)
                    pose_queries = rearrange(pose_queries, '(b f) 1 c -> b f c', b=b, f=f)
                    queries = rearrange(queries, '(b f) (h w d) c -> b f h w c d', b=b, f=f, h=h, w=w, d=d)
                    # queries = rearrange(queries, '(b f) (h w) c -> b f h w c', b=b, f=f, h=h, w=w)
                    
                pose_queries = pose_en(pose_queries)
                pose_tokens = pose_en(pose_tokens)
                encoder_outs_pose_queries.append(pose_queries)
                encoder_outs_pose_tokens.append(pose_tokens)
            
            queries = rearrange(queries, 'b f h w c d -> (b h w) (f d) c')
            tokens = rearrange(tokens, 'b f h w c d -> (b h w) (f d) c')
            # queries = rearrange(queries, 'b f h w c -> (b h w) f c')
            # tokens = rearrange(tokens, 'b f h w c -> (b h w) f c')
            for cross_attn, cross_norm, ffn, ffn_norm in temporal_attn:
                queries = queries + cross_attn(queries, tokens, tokens, need_weights=False, attn_mask=self.attn_mask)[0]
                queries = cross_norm(queries)
                
                queries = queries + ffn(queries)
                queries = ffn_norm(queries)

            queries = rearrange(queries, '(b h w) (f d) c -> (b f d) c h w', b=b, h=h, w=w, f=f, d=d)
            tokens = rearrange(tokens, '(b h w) (f d) c -> (b f d) c h w', b=b, h=h, w=w, f=f, d=d)
            # queries = rearrange(queries, '(b h w) f c -> (b f) c h w', b=b, h=h, w=w)
            # tokens = rearrange(tokens, '(b h w) f c -> (b f) c h w', b=b, h=h, w=w)
            queries = encoder(queries)
            tokens = encoder(tokens)
            encoder_outs_tokens.append(tokens)
            encoder_outs_queries.append(queries)
            queries = down(queries)
            tokens = down(tokens)
            queries = rearrange(queries, '(b f d) c h w -> b f h w c d', b=b, f=f, d=d)
            tokens = rearrange(tokens, '(b f d) c h w -> b f h w c d', b=b, f=f, d=d) 
            # queries = rearrange(queries, '(b f) c h w -> b f h w c', b=b, f=f)
            # tokens = rearrange(tokens, '(b f) c h w -> b f h w c', b=b, f=f) 

        b, f, h, w, c, d = queries.shape
        queries = rearrange(queries, 'b f h w c d -> (b f d) c h w')
        tokens = rearrange(tokens, 'b f h w c d -> (b f d) c h w')
        # queries = rearrange(queries, 'b f h w c -> (b f) c h w')
        # tokens = rearrange(tokens, 'b f h w c -> (b f) c h w')
        queries = self.mid(queries)
        tokens = self.mid(tokens)
        
        if pose_tokens is not None:
            pose_queries = self.pose_mid(pose_queries)
            pose_tokens = self.pose_mid(pose_tokens)
        
        for temporal_attn, decoder, up, encoder_out_queries, encoder_out_tokens, pose_attn_de, pose_de_, encoder_out_pose_queries, encoder_out_pose_tokens, pose_up in zip(self.temporal_attentions_de,
                                                                        self.decoders, self.upsamples, encoder_outs_queries[::-1],
                                                                        encoder_outs_tokens[::-1], self.pose_attn_de, self.pose_de,
                                                                        encoder_outs_pose_queries[::-1], encoder_outs_pose_tokens[::-1], self.pose_up):
            queries = up(queries)
            tokens = up(tokens)
            
            pad_x_queries = encoder_out_queries.shape[2] - queries.shape[2]
            pad_y_queries = encoder_out_queries.shape[3] - queries.shape[3]
            queries = nn.functional.pad(queries, (pad_x_queries//2, pad_x_queries-pad_x_queries//2,
                                      pad_y_queries//2, pad_y_queries-pad_y_queries//2))
            pad_x_tokens = encoder_out_tokens.shape[2] - tokens.shape[2]
            pad_y_tokens = encoder_out_tokens.shape[3] - tokens.shape[3]
            tokens = nn.functional.pad(tokens, (pad_x_tokens//2, pad_x_tokens-pad_x_tokens//2,
                                      pad_y_tokens//2, pad_y_tokens-pad_y_tokens//2))
            queries = torch.cat([queries, encoder_out_queries], dim=1)
            tokens = torch.cat([tokens, encoder_out_tokens], dim=1)
            c, h, w = queries.shape[-3:]
            queries = rearrange(queries, '(b f d) c h w -> (b h w) (f d) c', b=b, f=f, d=d)
            tokens = rearrange(tokens, '(b f d) c h w -> (b h w) (f d) c', b=b, f=f, d=d)
            # queries = rearrange(queries, '(b f) c h w -> (b h w) f c', b=b, f=f)
            # tokens = rearrange(tokens, '(b f) c h w -> (b h w) f c', b=b, f=f)
            for cross_attn, cross_norm, ffn, ffn_norm in temporal_attn:
                queries = queries + cross_attn(queries, tokens, tokens, need_weights=False, attn_mask=self.attn_mask)[0]
                queries = cross_norm(queries)
                
                queries = queries + ffn(queries)
                queries = ffn_norm(queries)
            queries = rearrange(queries, '(b h w) (f d) c -> (b f d) c h w', b=b, h=h, w=w, f=f, d=d)
            tokens = rearrange(tokens, '(b h w) (f d) c -> (b f d) c h w', b=b, h=h, w=w, f=f, d=d)
            # queries = rearrange(queries, '(b h w) f c -> (b f) c h w', b=b, h=h, w=w)
            # tokens = rearrange(tokens, '(b h w) f c -> (b f) c h w', b=b, h=h, w=w)
            
            if pose_tokens is not None:
                pose_queries = pose_up(pose_queries)
                pose_tokens = pose_up(pose_tokens)
                pose_queries = torch.cat([pose_queries, encoder_out_pose_queries], dim=2)
                pose_tokens = torch.cat([pose_tokens, encoder_out_pose_tokens], dim=2)
                
                for pose_temporal_attn, pose_temporal_norm, spatial_attn, spatial_norm, ffn, ffn_norm in pose_attn_de:
                    pose_queries = pose_queries + pose_temporal_attn(pose_queries, pose_tokens, pose_tokens, need_weights=False, attn_mask=self.pose_attn_mask)[0]
                    pose_queries = pose_temporal_norm(pose_queries)
                    #b, f, h, w, c = queries.shape
                    pose_queries = rearrange(pose_queries, 'b f c -> (b f) 1 c')
                    queries = rearrange(queries, '(b f d) c h w -> (b f) (h w d) c', b=b, f=f, d=d)
                    # queries = rearrange(queries, '(b f) c h w -> (b f) (h w) c', b=b, f=f, h=h, w=w)
                    pose_queries = pose_queries + spatial_attn(pose_queries, queries, queries, need_weights=False, attn_mask=None)[0]
                    pose_queries = spatial_norm(pose_queries)
                    
                    pose_queries = pose_queries + ffn(pose_queries)
                    pose_queries = ffn_norm(pose_queries)
                    queries = rearrange(queries, '(b f) (h w d) c -> (b f d) c h w', b=b, f=f, h=h, w=w, d=d)
                    # queries = rearrange(queries, '(b f) (h w) c -> (b f) c h w', b=b, f=f, h=h, w=w)
                    pose_queries = rearrange(pose_queries, '(b f) 1 c -> b f c', b=b, f=f)
                pose_queries = pose_de_(pose_queries)
                pose_tokens = pose_de_(pose_tokens)
            queries = decoder(queries)
            tokens = decoder(tokens)

        # NOTE WE ALREADY attentioned across temporal and depth dimension
        queries = self.conv_out(queries) # NOTE IDEA conv_out을 depth마다 각각 써보기? 요정도 IDEA2 causal mask 작은계단 
        """ main IDEA1 3이 그거에요 그지금은 f랑 d랑 동시에 묶어서 돌리고 있는데 이거를 각각 independent하게 원래 temporal attention가한 후 에 depth transformer를 새로 돌리는 방식 각 레이어마다."""
        """ main IDEA2 spatial이 고려 안되있는데 이게 이유가 기존 occworld에서 temporal만해서 근데 rqvae에서 썼던 것처럼 초반에만 hw다듬어서 spatial_ctx만들어서 집어넣기? """
        queries = rearrange(queries, '(b f d) c h w -> b f d c h w', b=b, f=f, d=d)
        if pose_tokens is not None:
            pose_queries = self.pose_out(pose_queries)
            return queries, pose_queries
        
        return queries
        # queries = rearrange(queries, '(b f) c h w -> (b h w) f 1 c', f=f) # bhw f 1 c
        # res_queries = rearrange(res_tokens[..., :-1], 'b f c h w d -> (b h w) f d c') # bhw f d-1 c
        # queries = torch.cat([queries, res_queries], dim=-2).flatten(1,2) # bhw fd c
        # tokens = rearrange(tokens, '(b f) c h w -> (b h w) f c', f=f) # bhw f c

        # queries = queries + self.depth_embeddings.weight[None, :f*d, :].expand(bs, -1, -1)
        # tokens = tokens + self.depth_embeddings.weight[None :f, :].expand(bs, -1, -1)

        # queries = queries + self.depth_attention_layers[0](queries, tokens, tokens, need_weights=False, attn_mask=self.depth_attn_mask)[0]
        # # queries = queries + self.depth_attention_layers[0](queries, queries, queries, need_weights=False, attn_mask=self.depth_attn_mask)[0]
        # queries = self.depth_attention_layers[1](queries)
        
        # queries = queries + self.depth_attention_layers[2](queries)
        # queries = self.depth_attention_layers[3](queries)
            
        # queries = rearrange(queries, '(b h w) (f d) c-> (b f d) c h w', b=b, h=h, w=w, f=f)
        # queries = self.conv_out(queries)
        # queries = rearrange(queries, '(b f d) c h w -> b f d c h w', b=b, f=f)
        # if pose_tokens is not None:
        #     pose_queries = self.pose_out(pose_queries)
        #     return queries, pose_queries

        # return queries

    def forward_autoreg(self, tokens, pose_tokens, start_frame=0, mid_frame=6, end_frame=12):
        tokens = tokens[:, start_frame:mid_frame]
        pose_tokens = pose_tokens[:, start_frame:mid_frame]
        for i in range(mid_frame, end_frame):
            token, pose_token = self.forward_autoreg_step(tokens, pose_tokens, start_frame, i)
            
            tokens = torch.cat([tokens, token[:, -1:]], dim=1)
            pose_tokens = torch.cat([pose_tokens, pose_token[:, -1:]], dim=1)
        b, f, c, h, w = tokens.shape
        queries = rearrange(tokens, 'b f c h w -> (b f) c h w')
        queries = self.conv_out(queries)
        pose_queries = self.pose_out(pose_tokens)
        queries = rearrange(queries, '(b f) c h w -> b f c h w', b=b, f=f)
        
        return queries[:,mid_frame:end_frame], pose_queries[:,mid_frame:end_frame]

    def forward_autoreg_step(self, tokens, res_tokens, pose_tokens=None, start_frame=0, mid_frame=6):
        tokens = tokens[:, start_frame:mid_frame]
        res_tokens = res_tokens[:, start_frame:mid_frame]
        if pose_tokens is not None:
            pose_tokens = pose_tokens[:, start_frame:mid_frame]
        bs, F, C, H, W, D = res_tokens.shape
        tokens = rearrange(res_tokens, 'b f c h w d -> b f d h w c')
        queries = tokens

        queries = queries + self.temporal_embeddings.weight[None, self.offset:F+self.offset, None, None, None, :].expand(
            bs, -1, D, H, W, -1)
        tokens = tokens + self.temporal_embeddings.weight[None, :F, None, None, None, :].expand(
            bs, -1, D, H, W, -1)

        queries = queries + self.depth_embeddings.weight[None, None, :, None, None, :].expand(
            bs, F, -1, H, W, -1)
        tokens = tokens + self.depth_embeddings.weight[None, None, :, None, None, :].expand(
            bs, F, -1, H, W, -1)

        queries = rearrange(queries, 'b f d h w c -> b f h w c d')
        tokens = rearrange(tokens, 'b f d h w c -> b f h w c d')

        encoder_outs_pose_tokens = [None for _ in range(len(self.pose_attn_en))]
        encoder_outs_pose_queries = [None for _ in range(len(self.pose_attn_en))]

        if pose_tokens is not None:
            if self.learnable_queries:
                pose_queries = self.pose_queries.weight.reshape(1, 1, C).expand(bs, F, C)
            else:
                pose_queries = pose_tokens
            pose_queries = pose_queries + self.pose_temporal_embeddings.weight[None, self.offset:F+self.offset, :].expand(
                bs, -1, -1)
            pose_tokens = pose_tokens + self.pose_temporal_embeddings.weight[None, :F, :].expand(
                bs, -1, -1)
            encoder_outs_pose_tokens = []
            encoder_outs_pose_queries = []

        encoder_outs_tokens = []
        encoder_outs_queries = []
        
        for temporal_attn, encoder, down, pose_attn_en, pose_en in zip(self.temporal_attentions_en, self.encoders, self.downsamples, self.pose_attn_en, self.pose_en):
            b, f, h, w, c, d = tokens.shape
            
            if pose_tokens is not None:
                for pose_temporal_attn, pose_temporal_norm, spatial_attn, spatial_norm, ffn, ffn_norm in pose_attn_en:
                    pose_queries = pose_queries + pose_temporal_attn(pose_queries, pose_tokens, pose_tokens, need_weights=False, attn_mask=self.pose_attn_mask[:f,:f])[0]
                    pose_queries = pose_temporal_norm(pose_queries)
                    pose_queries = rearrange(pose_queries, 'b f c -> (b f) 1 c')
                    queries = rearrange(queries, 'b f h w c d -> (b f) (h w d) c')
                    pose_queries = pose_queries + spatial_attn(pose_queries, queries, queries, need_weights=False, attn_mask=None)[0]
                    pose_queries = spatial_norm(pose_queries)
                    
                    pose_queries = pose_queries + ffn(pose_queries)
                    pose_queries = ffn_norm(pose_queries)
                    pose_queries = rearrange(pose_queries, '(b f) 1 c -> b f c', b=b, f=f)
                    queries = rearrange(queries, '(b f) (h w d) c -> b f h w c d', b=b, f=f, h=h, w=w, d=d)
                    
                pose_queries = pose_en(pose_queries)
                pose_tokens = pose_en(pose_tokens)
                encoder_outs_pose_queries.append(pose_queries)
                encoder_outs_pose_tokens.append(pose_tokens)
            
            queries = rearrange(queries, 'b f h w c d -> (b h w) (f d) c')
            tokens = rearrange(tokens, 'b f h w c d -> (b h w) (f d) c')
            # queries = rearrange(queries, 'b f h w c -> (b h w) f c')
            # tokens = rearrange(tokens, 'b f h w c -> (b h w) f c')
            for cross_attn, cross_norm, ffn, ffn_norm in temporal_attn:
                queries = queries + cross_attn(queries, tokens, tokens, need_weights=False, attn_mask=self.attn_mask[:f*d, :f*d])[0]
                queries = cross_norm(queries)
                
                queries = queries + ffn(queries)
                queries = ffn_norm(queries)
            
            queries = rearrange(queries, '(b h w) (f d) c -> (b f d) c h w', b=b, h=h, w=w, f=f, d=d)
            tokens = rearrange(tokens, '(b h w) (f d) c -> (b f d) c h w', b=b, h=h, w=w, f=f, d=d)
            # queries = rearrange(queries, '(b h w) f c -> (b f) c h w', b=b, h=h, w=w)
            # tokens = rearrange(tokens, '(b h w) f c -> (b f) c h w', b=b, h=h, w=w)
            queries = encoder(queries)
            tokens = encoder(tokens)
            encoder_outs_tokens.append(tokens)
            encoder_outs_queries.append(queries)
            queries = down(queries)
            tokens = down(tokens)
            queries = rearrange(queries, '(b f d) c h w -> b f h w c d', b=b, f=f, d=d)
            tokens = rearrange(tokens, '(b f d) c h w -> b f h w c d', b=b, f=f, d=d) 
            # queries = rearrange(queries, '(b f) c h w -> b f h w c', b=b, f=f)
            # tokens = rearrange(tokens, '(b f) c h w -> b f h w c', b=b, f=f) 

        b, f, h, w, c, d = queries.shape
        queries = rearrange(queries, 'b f h w c d -> (b f d) c h w')
        tokens = rearrange(tokens, 'b f h w c d -> (b f d) c h w')
        # queries = rearrange(queries, 'b f h w c -> (b f) c h w')
        # tokens = rearrange(tokens, 'b f h w c -> (b f) c h w')
        queries = self.mid(queries)
        tokens = self.mid(tokens)
        
        if pose_tokens is not None:
            pose_queries = self.pose_mid(pose_queries)
            pose_tokens = self.pose_mid(pose_tokens)
        
        # queries = rearrange(queries, '(b f) c h w -> b f h w c', b=b, f=f)
        # tokens = rearrange(tokens, '(b f) c h w -> b f h w c', b=b, f=f)
        for temporal_attn, decoder, up, encoder_out_queries, encoder_out_tokens, pose_attn_de, pose_de_, encoder_out_pose_queries, encoder_out_pose_tokens, pose_up in zip(self.temporal_attentions_de,
                                                                        self.decoders, self.upsamples, encoder_outs_queries[::-1],
                                                                        encoder_outs_tokens[::-1], self.pose_attn_de, self.pose_de,
                                                                        encoder_outs_pose_queries[::-1], encoder_outs_pose_tokens[::-1], self.pose_up):
            queries = up(queries)
            tokens = up(tokens)
            
            pad_x_queries = encoder_out_queries.shape[2] - queries.shape[2]
            pad_y_queries = encoder_out_queries.shape[3] - queries.shape[3]
            queries = nn.functional.pad(queries, (pad_x_queries//2, pad_x_queries-pad_x_queries//2,
                                      pad_y_queries//2, pad_y_queries-pad_y_queries//2))
            pad_x_tokens = encoder_out_tokens.shape[2] - tokens.shape[2]
            pad_y_tokens = encoder_out_tokens.shape[3] - tokens.shape[3]
            tokens = nn.functional.pad(tokens, (pad_x_tokens//2, pad_x_tokens-pad_x_tokens//2,
                                      pad_y_tokens//2, pad_y_tokens-pad_y_tokens//2))
            queries = torch.cat([queries, encoder_out_queries], dim=1)
            tokens = torch.cat([tokens, encoder_out_tokens], dim=1)
            c, h, w = queries.shape[-3:]
            queries = rearrange(queries, '(b f d) c h w -> (b h w) (f d) c', b=b, f=f, d=d)
            tokens = rearrange(tokens, '(b f d) c h w -> (b h w) (f d) c', b=b, f=f, d=d)
            # queries = rearrange(queries, '(b f) c h w -> (b h w) f c', b=b, f=f)
            # tokens = rearrange(tokens, '(b f) c h w -> (b h w) f c', b=b, f=f)
            for cross_attn, cross_norm, ffn, ffn_norm in temporal_attn:
                queries = queries + cross_attn(queries, tokens, tokens, need_weights=False, attn_mask=self.attn_mask[:f*d,:f*d])[0]
                queries = cross_norm(queries)
                
                queries = queries + ffn(queries)
                queries = ffn_norm(queries)
            queries = rearrange(queries, '(b h w) (f d) c -> (b f d) c h w', b=b, h=h, w=w, f=f, d=d)
            tokens = rearrange(tokens, '(b h w) (f d) c -> (b f d) c h w', b=b, h=h, w=w, f=f, d=d)
            # queries = rearrange(queries, '(b h w) f c -> (b f) c h w', b=b, h=h, w=w)
            # tokens = rearrange(tokens, '(b h w) f c -> (b f) c h w', b=b, h=h, w=w)
            
            if pose_tokens is not None:
                pose_queries = pose_up(pose_queries)
                pose_tokens = pose_up(pose_tokens)
                pose_queries = torch.cat([pose_queries, encoder_out_pose_queries], dim=2)
                pose_tokens = torch.cat([pose_tokens, encoder_out_pose_tokens], dim=2)
                
                for pose_temporal_attn, pose_temporal_norm, spatial_attn, spatial_norm, ffn, ffn_norm in pose_attn_de:
                    pose_queries = pose_queries + pose_temporal_attn(pose_queries, pose_tokens, pose_tokens, need_weights=False, attn_mask=self.pose_attn_mask[:f, :f])[0]
                    pose_queries = pose_temporal_norm(pose_queries)
                    pose_queries = rearrange(pose_queries, 'b f c -> (b f) 1 c')
                    queries = rearrange(queries, '(b f d) c h w -> (b f) (h w d) c', b=b, f=f, d=d)
                    # queries = rearrange(queries, '(b f) c h w -> (b f) (h w) c', b=b, f=f, h=h, w=w)
                    pose_queries = pose_queries + spatial_attn(pose_queries, queries, queries, need_weights=False, attn_mask=None)[0]
                    pose_queries = spatial_norm(pose_queries)
                    
                    pose_queries = pose_queries + ffn(pose_queries)
                    pose_queries = ffn_norm(pose_queries)
                    queries = rearrange(queries, '(b f) (h w d) c -> (b f d) c h w', b=b, f=f, h=h, w=w, d=d)
                    # queries = rearrange(queries, '(b f) (h w) c -> (b f) c h w', b=b, f=f, h=h, w=w)
                    pose_queries = rearrange(pose_queries, '(b f) 1 c -> b f c', b=b, f=f)
                pose_queries = pose_de_(pose_queries)
                pose_tokens = pose_de_(pose_tokens)
            queries = decoder(queries)
            tokens = decoder(tokens)

        queries = self.conv_out(queries) # NOTE IDEA conv_out을 depth마다 각각 써보기? 요정도 IDEA2 causal mask 작은계단 
        queries = rearrange(queries, '(b f d) c h w -> b f d c h w', b=b, f=f, d=d)
        if pose_tokens is not None:
            pose_queries = self.pose_out(pose_queries)
            return queries, pose_queries
        
        return queries


@MODELS.register_module()
class PlanUAutoRegTransformerGuide(BaseModule):
    def __init__(
        self,
        num_tokens,
        num_frames,
        num_layers,
        img_shape,
        pose_shape,
        tpe_dim=10,
        output_channel=1024,
        channels=[1,2,3],
        ffn_dims=None,
        temporal_attn_layers=1,
        pose_attn_layers=1,
        num_heads=8,
        pose_output_channel=None,
        conditional=True,
        tokens_untouched=False,
        add_aggregate=False,
        learnable_queries=True,
        without_multiscale=False,
        without_spatial_attn=False,
        without_pose_spatial_attn=False,
        without_pose_temporal_attn=False,
        without_temporal_attn=False) -> None:
        super().__init__()
        if without_multiscale:
            assert len(channels) == 2
        self.num_tokens = num_tokens
        self.num_frames = num_frames
        self.num_layers = num_layers
        self.channels = channels
        self.learnable_queries = learnable_queries
        if self.learnable_queries:
            self.queries = nn.Embedding(img_shape[1]*img_shape[2], img_shape[0])
        self.offset = 1 if conditional else 0
        self.temporal_embeddings = nn.Embedding(num_frames + self.offset, tpe_dim)
        if self.learnable_queries:
            self.pose_queries = nn.Embedding(pose_shape[0], pose_shape[1])
        self.pose_temporal_embeddings = nn.Embedding(num_frames + self.offset, tpe_dim)
        self.pose_attn_layers = pose_attn_layers if pose_attn_layers is not None else temporal_attn_layers
        
        self.temporal_attentions_en = nn.ModuleList([])
        self.temporal_attentions_de = nn.ModuleList([])
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        self.pose_attn_en = nn.ModuleList([])
        self.pose_en = nn.ModuleList()
        #self.pose_temporal_attn_en = nn.ModuleList([])
        self.pose_attn_de = nn.ModuleList([])
        self.pose_de = nn.ModuleList()
        self.pose_up = nn.ModuleList()
        #self.pose_temporal_attn_de = nn.ModuleList([])
        
        self.downsamples = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        
        up_down_sample_params = dict(kernel_size=2, stride=2, padding=0)
        self.up_down_sample_params = up_down_sample_params
        self.unfold_params = dict(kernel_size=[1, 2], stride=[1, 2], padding=[0, 0])
        
        C, H, W = img_shape
        layers = len(channels)
        Hs = [H]
        Ws = [W]
        cH = H
        cW = W
        for _ in range(layers-1):
            cH = (cH) // 2
            cW = (cW) // 2
            Hs.append(cH)
            Ws.append(cW)
        pre_c = C
        for channel, cH, cW in zip(channels[0:-1], Hs[0:-1], Ws[0:-1]):
            temporal_attn_layer = nn.ModuleList()
            for i in range(temporal_attn_layers):
                if without_temporal_attn:
                    temporal_attn_layer.append(nn.ModuleList([
                        Identity(),
                        nn.LayerNorm(pre_c),
                        Identity(),
                        nn.LayerNorm(pre_c),
                    ]))
                else:
                    temporal_attn_layer.append(nn.ModuleList([
                        nn.MultiheadAttention(pre_c, num_heads, batch_first=True),
                        nn.LayerNorm(pre_c),
                        FFN(pre_c, pre_c*4),
                        nn.LayerNorm(pre_c)
                    ]))
            self.temporal_attentions_en.append(temporal_attn_layer)
            if without_spatial_attn:
                self.encoders.append(nn.Sequential(IdentityUnetBlock((cH, cW), pre_c, channel), 
                                                   IdentityUnetBlock((cH, cW), channel, channel)))
            else:
                self.encoders.append(nn.Sequential(UnetBlock((cH, cW), pre_c, channel, True),
                                               UnetBlock((cH, cW), channel, channel, True)))
            if without_multiscale:
                self.downsamples.append(Identity())
            else:
                self.downsamples.append(nn.Conv2d(channel, channel, **up_down_sample_params))
            self.pose_en.append(nn.Sequential(
                nn.Linear(pre_c, channel),nn.ReLU(),
                nn.Linear(channel, channel),nn.ReLU(),
            ))
            pose_attn_layer = nn.ModuleList()
            for i in range(pose_attn_layers):
                if without_pose_temporal_attn:
                    pose_attn_layer.append(nn.ModuleList([
                        Identity(),
                        nn.LayerNorm(pre_c),
                        nn.MultiheadAttention(pre_c, num_heads, batch_first=True),
                        nn.LayerNorm(pre_c),
                        FFN(pre_c, pre_c*4),
                        nn.LayerNorm(pre_c)
                ]))
                elif without_pose_spatial_attn:
                    pose_attn_layer.append(nn.ModuleList([
                        nn.MultiheadAttention(pre_c, num_heads, batch_first=True),
                        nn.LayerNorm(pre_c),
                        Identity(),
                        nn.LayerNorm(pre_c),
                        FFN(pre_c, pre_c*4),
                        nn.LayerNorm(pre_c)
                ]))
                else: 
                    pose_attn_layer.append(nn.ModuleList([
                        nn.MultiheadAttention(pre_c, num_heads, batch_first=True),
                        nn.LayerNorm(pre_c),
                        nn.MultiheadAttention(pre_c, num_heads, batch_first=True),
                        nn.LayerNorm(pre_c),
                        FFN(pre_c, pre_c*4),
                        nn.LayerNorm(pre_c)
                    ]))
            self.pose_attn_en.append(pose_attn_layer)
            pre_c = channel
        channel = channels[-1]
        if without_multiscale:
            if without_spatial_attn:
                self.mid = nn.Sequential(
                    IdentityUnetBlock((pre_c, Hs[0], Ws[0]), pre_c, channel, True),
                    IdentityUnetBlock((channel, Hs[0], Ws[0]), channel, channel, True),
                )
            else:
                self.mid = nn.Sequential(
                UnetBlock((pre_c, Hs[0], Ws[0]), pre_c, channel, True),
                UnetBlock((channel, Hs[0], Ws[0]), channel, channel, True),
            )
        else:
            self.mid = nn.Sequential(
                UnetBlock((pre_c, Hs[-1], Ws[-1]), pre_c, channel, True),
                UnetBlock((channel, Hs[-1], Ws[-1]), channel, channel, True),
            )
        self.pose_mid = nn.Sequential(
            nn.Linear(pre_c, channel),nn.ReLU(),
            nn.Linear(channel, channel),nn.ReLU(),
        )
        pre_c = channel
        for channel, cH, cW in zip(channels[-2::-1], Hs[-2::-1], Ws[-2::-1]):
            channel_agg = channel if add_aggregate else channel * 2
            if without_multiscale:
                self.upsamples.append(Identity())
            else:
                self.upsamples.append(nn.ConvTranspose2d(pre_c, channel, **up_down_sample_params))
            temporal_attn_layer = nn.ModuleList()
            for i in range(temporal_attn_layers):
                if without_temporal_attn:
                    temporal_attn_layer.append(nn.ModuleList([
                        Identity(),
                        nn.LayerNorm(channel_agg),
                        Identity(),
                        nn.LayerNorm(channel_agg),
                    ]))
                else:
                    temporal_attn_layer.append(nn.ModuleList([
                        nn.MultiheadAttention(channel_agg, num_heads, batch_first=True),
                        nn.LayerNorm(channel_agg),
                        FFN(channel_agg, channel_agg*4),
                        nn.LayerNorm(channel_agg)
                    ]))
            self.temporal_attentions_de.append(temporal_attn_layer)
            if without_spatial_attn:
                self.decoders.append(nn.Sequential(
                    IdentityUnetBlock((channel_agg, cH,cW), channel_agg, channel, True),
                    IdentityUnetBlock((channel, cH,cW), channel, channel, True)
                ))
            else:
                self.decoders.append(
                    nn.Sequential(
                        UnetBlock((channel_agg, cH,cW),
                                channel_agg, channel, True),
                        UnetBlock((channel, cH,cW),
                                channel,
                                channel,
                                True)
                    )
                )
            pose_attn_layer = nn.ModuleList()
            self.pose_up.append(nn.Linear(pre_c, channel))
            for i in range(pose_attn_layers):
                if without_pose_temporal_attn:
                    pose_attn_layer.append(nn.ModuleList([
                        Identity(),
                        nn.LayerNorm(channel_agg),
                        nn.MultiheadAttention(channel_agg, num_heads, batch_first=True),
                        nn.LayerNorm(channel_agg),
                        FFN(channel_agg, channel_agg*4),
                        nn.LayerNorm(channel_agg)
                ]))
                elif without_pose_spatial_attn:
                    pose_attn_layer.append(nn.ModuleList([
                        nn.MultiheadAttention(channel_agg, num_heads, batch_first=True),
                        nn.LayerNorm(channel_agg),
                        Identity(),
                        nn.LayerNorm(channel_agg),
                        FFN(channel_agg, channel_agg*4),
                        nn.LayerNorm(channel_agg)
                ]))
                else:
                    pose_attn_layer.append(nn.ModuleList([
                        nn.MultiheadAttention(channel_agg, num_heads, batch_first=True),
                        nn.LayerNorm(channel_agg),
                        nn.MultiheadAttention(channel_agg, num_heads, batch_first=True),
                        nn.LayerNorm(channel_agg),
                        FFN(channel_agg, channel_agg*4),
                        nn.LayerNorm(channel_agg)
                    ]))
            self.pose_attn_de.append(pose_attn_layer)
            self.pose_de.append(nn.Sequential(
                nn.Linear(channel_agg, channel),nn.ReLU(),
                nn.Linear(channel, channel),nn.ReLU(),
            ))
            pre_c = channel
        
        self.conv_out = nn.Conv2d(pre_c, output_channel, 3, 1, 1)
        # self.conv_outs = nn.ModuleList([nn.Conv2d(pre_c, output_channel, 3, 1, 1) for _ in range(4)]) # NOTE hardcoded: depth==4
        
        self.pose_out = nn.Linear(pre_c, pose_output_channel if pose_output_channel is not None else output_channel)
        
        self.tokens_untouched = tokens_untouched
        
        if tokens_untouched:
            assert all([ch == channels[0] for ch in channels])
            for scale in range(len(channels) - 1):
                num_tokens = self.unfold_params['kernel_size'][scale] ** 2
                attn_mask = torch.zeros(num_frames, num_frames * num_tokens, dtype=torch.bool)
                for i_frame in range(num_frames):
                    start = i_frame * num_tokens + num_tokens if conditional else i_frame * num_tokens
                    attn_mask[i_frame, start:] = True
                self.register_buffer(f'attn_mask_{scale}', attn_mask, False)
        else:
            attn_mask = torch.zeros(num_frames * num_tokens, num_frames * num_tokens, dtype=torch.bool)
            for i_frame in range(num_frames):
                start1 = i_frame * num_tokens
                start2 = start1 + num_tokens if conditional else start1
                attn_mask[start1: (start1 + num_tokens), start2:] = True
            self.register_buffer('attn_mask', attn_mask, False)
    
    def forward(self, tokens, pose_tokens=None, guide=None):
        assert guide is not None
        #import pdb;pdb.set_trace()
        # tokens: bs, f, c, h, w
        # pose_tokens, bs, f, c
        bs, F, C, H, W = tokens.shape
        assert F == self.num_frames
        tokens = rearrange(tokens, 'b f c h w -> b f h w c')
        guide = rearrange(guide, 'b f c h w -> b f h w c')
        if self.learnable_queries:
            queries = self.queries.weight.reshape(1, 1, H, W, C).expand(bs, F, H, W, C)
        else:
            queries = tokens
        queries = queries + self.temporal_embeddings.weight[None, self.offset:, None, None, :].expand(
            bs, -1, H, W, -1)
        tokens = tokens + self.temporal_embeddings.weight[None, :self.num_frames, None, None, :].expand(
            bs, -1, H, W, -1)
        #
        guide = guide + self.temporal_embeddings.weight[None, self.offset:, None, None, :].expand(
            bs, -1, H, W, -1)
        tokens = torch.cat([tokens, guide], dim=1) # b 2f h w c
        #
        encoder_outs_pose_tokens = [None for _ in range(len(self.pose_attn_en))]
        encoder_outs_pose_queries = [None for _ in range(len(self.pose_attn_en))]

        if pose_tokens is not None:
            if self.learnable_queries:
                pose_queries = self.pose_queries.weight.reshape(1, 1, C).expand(bs, F, C)
            else:
                pose_queries = pose_tokens
            pose_queries = pose_queries + self.pose_temporal_embeddings.weight[None, self.offset:, :].expand(
                bs, -1, -1)
            pose_tokens = pose_tokens + self.pose_temporal_embeddings.weight[None, :self.num_frames, :].expand(
                bs, -1, -1)
            encoder_outs_pose_tokens = []
            encoder_outs_pose_queries = []
        
        encoder_outs_tokens = []
        encoder_outs_queries = []
        
        for temporal_attn, encoder, down, pose_attn_en, pose_en in zip(self.temporal_attentions_en, self.encoders, self.downsamples, self.pose_attn_en, self.pose_en):
            # b, f, h, w, c = tokens.shape
            b, f, h, w, c = queries.shape
            
            if pose_tokens is not None:
                for pose_temporal_attn, pose_temporal_norm, spatial_attn, spatial_norm, ffn, ffn_norm in pose_attn_en:
                    pose_queries = pose_queries + pose_temporal_attn(pose_queries, pose_tokens, pose_tokens, need_weights=False, attn_mask=self.attn_mask)[0]
                    pose_queries = pose_temporal_norm(pose_queries)
                    #b, f, h, w, c = queries.shape
                    pose_queries = rearrange(pose_queries, 'b f c -> (b f) 1 c')
                    queries = rearrange(queries, 'b f h w c -> (b f) (h w) c')
                    pose_queries = pose_queries + spatial_attn(pose_queries, queries, queries, need_weights=False, attn_mask=None)[0]
                    pose_queries = spatial_norm(pose_queries)
                    
                    pose_queries = pose_queries + ffn(pose_queries)
                    pose_queries = ffn_norm(pose_queries)
                    pose_queries = rearrange(pose_queries, '(b f) 1 c -> b f c', b=b, f=f)
                    queries = rearrange(queries, '(b f) (h w) c -> b f h w c', b=b, f=f, h=h, w=w)
                
                pose_queries = pose_en(pose_queries)
                pose_tokens = pose_en(pose_tokens)
                encoder_outs_pose_queries.append(pose_queries)
                encoder_outs_pose_tokens.append(pose_tokens)
            
            queries = rearrange(queries, 'b f h w c -> (b h w) f c')
            tokens = rearrange(tokens, 'b f h w c -> (b h w) f c')
            for cross_attn, cross_norm, ffn, ffn_norm in temporal_attn:
                # queries = queries + cross_attn(queries, tokens, tokens, need_weights=False, attn_mask=torch.cat([self.attn_mask)[0]
                queries = queries + cross_attn(queries, tokens, tokens, need_weights=False, attn_mask=torch.cat([self.attn_mask, self.attn_mask], dim=1))[0]
                queries = cross_norm(queries)
                
                queries = queries + ffn(queries)
                queries = ffn_norm(queries)
            
            queries = rearrange(queries, '(b h w) f c -> (b f) c h w', b=b, h=h, w=w)
            tokens = rearrange(tokens, '(b h w) f c -> (b f) c h w', b=b, h=h, w=w)
            queries = encoder(queries)
            tokens = encoder(tokens)
            encoder_outs_tokens.append(tokens)
            encoder_outs_queries.append(queries)
            queries = down(queries)
            tokens = down(tokens)
            queries = rearrange(queries, '(b f) c h w -> b f h w c', b=b, f=f)
            # tokens = rearrange(tokens, '(b f) c h w -> b f h w c', b=b, f=f) 
            tokens = rearrange(tokens, '(b f) c h w -> b f h w c', b=b, f=2*f) 
        b, f, h, w, c = queries.shape
        queries = rearrange(queries, 'b f h w c -> (b f) c h w')
        tokens = rearrange(tokens, 'b f h w c -> (b f) c h w')
        queries = self.mid(queries)
        tokens = self.mid(tokens)
        
        if pose_tokens is not None:
            pose_queries = self.pose_mid(pose_queries)
            pose_tokens = self.pose_mid(pose_tokens)
        
        # queries = rearrange(queries, '(b f) c h w -> b f h w c', b=b, f=f)
        # tokens = rearrange(tokens, '(b f) c h w -> b f h w c', b=b, f=f)
        for temporal_attn, decoder, up, encoder_out_queries, encoder_out_tokens, pose_attn_de, pose_de_, encoder_out_pose_queries, encoder_out_pose_tokens, pose_up in zip(self.temporal_attentions_de,
                                                                        self.decoders, self.upsamples, encoder_outs_queries[::-1],
                                                                        encoder_outs_tokens[::-1], self.pose_attn_de, self.pose_de,
                                                                        encoder_outs_pose_queries[::-1], encoder_outs_pose_tokens[::-1], self.pose_up):
            queries = up(queries)
            tokens = up(tokens)
            
            pad_x_queries = encoder_out_queries.shape[2] - queries.shape[2]
            pad_y_queries = encoder_out_queries.shape[3] - queries.shape[3]
            queries = nn.functional.pad(queries, (pad_x_queries//2, pad_x_queries-pad_x_queries//2,
                                      pad_y_queries//2, pad_y_queries-pad_y_queries//2))
            pad_x_tokens = encoder_out_tokens.shape[2] - tokens.shape[2]
            pad_y_tokens = encoder_out_tokens.shape[3] - tokens.shape[3]
            tokens = nn.functional.pad(tokens, (pad_x_tokens//2, pad_x_tokens-pad_x_tokens//2,
                                      pad_y_tokens//2, pad_y_tokens-pad_y_tokens//2))
            queries = torch.cat([queries, encoder_out_queries], dim=1)
            tokens = torch.cat([tokens, encoder_out_tokens], dim=1)
            c, h, w = queries.shape[-3:]
            queries = rearrange(queries, '(b f) c h w -> (b h w) f c', b=b, f=f)
            # tokens = rearrange(tokens, '(b f) c h w -> (b h w) f c', b=b, f=f)
            tokens = rearrange(tokens, '(b f) c h w -> (b h w) f c', b=b, f=2*f)
            for cross_attn, cross_norm, ffn, ffn_norm in temporal_attn:
                # queries = queries + cross_attn(queries, tokens, tokens, need_weights=False, attn_mask=self.attn_mask)[0]
                queries = queries + cross_attn(queries, tokens, tokens, need_weights=False, attn_mask=torch.cat([self.attn_mask, self.attn_mask], dim=1))[0]
                queries = cross_norm(queries)
                
                queries = queries + ffn(queries)
                queries = ffn_norm(queries)
            queries = rearrange(queries, '(b h w) f c -> (b f) c h w', b=b, h=h, w=w)
            tokens = rearrange(tokens, '(b h w) f c -> (b f) c h w', b=b, h=h, w=w)
            
            if pose_tokens is not None:
                pose_queries = pose_up(pose_queries)
                pose_tokens = pose_up(pose_tokens)
                pose_queries = torch.cat([pose_queries, encoder_out_pose_queries], dim=2)
                pose_tokens = torch.cat([pose_tokens, encoder_out_pose_tokens], dim=2)
                
                for pose_temporal_attn, pose_temporal_norm, spatial_attn, spatial_norm, ffn, ffn_norm in pose_attn_de:
                    pose_queries = pose_queries + pose_temporal_attn(pose_queries, pose_tokens, pose_tokens, need_weights=False, attn_mask=self.attn_mask)[0]
                    pose_queries = pose_temporal_norm(pose_queries)
                    #b, f, h, w, c = queries.shape
                    pose_queries = rearrange(pose_queries, 'b f c -> (b f) 1 c')
                    #queries = rearrange(queries, 'b f h w c -> (b f) (h w) c')
                    queries = rearrange(queries, '(b f) c h w -> (b f) (h w) c', b=b, f=f, h=h, w=w)
                    pose_queries = pose_queries + spatial_attn(pose_queries, queries, queries, need_weights=False, attn_mask=None)[0]
                    pose_queries = spatial_norm(pose_queries)
                    
                    pose_queries = pose_queries + ffn(pose_queries)
                    pose_queries = ffn_norm(pose_queries)
                    queries = rearrange(queries, '(b f) (h w) c -> (b f) c h w', b=b, f=f, h=h, w=w)
                    pose_queries = rearrange(pose_queries, '(b f) 1 c -> b f c', b=b, f=f)
                pose_queries = pose_de_(pose_queries)
                pose_tokens = pose_de_(pose_tokens)

            queries = decoder(queries)
            tokens = decoder(tokens)
            
        queries = self.conv_out(queries)
        queries = rearrange(queries, '(b f) c h w -> b f c h w', b=b, f=f)

        if pose_tokens is not None:
            pose_queries = self.pose_out(pose_queries)
            return queries, pose_queries
        
        return queries

    def forward_autoreg(self, tokens, pose_tokens, start_frame=0, mid_frame=6, end_frame=12):
        tokens = tokens[:, start_frame:mid_frame]
        pose_tokens = pose_tokens[:, start_frame:mid_frame]
        for i in range(mid_frame, end_frame):
            token, pose_token = self.forward_autoreg_step(tokens, pose_tokens, start_frame, i)
            
            tokens = torch.cat([tokens, token[:, -1:]], dim=1)
            pose_tokens = torch.cat([pose_tokens, pose_token[:, -1:]], dim=1)
        b, f, c, h, w = tokens.shape
        queries = rearrange(tokens, 'b f c h w -> (b f) c h w')
        queries = self.conv_out(queries)
        pose_queries = self.pose_out(pose_tokens)
        queries = rearrange(queries, '(b f) c h w -> b f c h w', b=b, f=f)
        
        return queries[:,mid_frame:end_frame], pose_queries[:,mid_frame:end_frame]

    def forward_autoreg_step(self, tokens, pose_tokens=None, guide=None, start_frame=0, mid_frame=6):
        assert guide is not None
        # bs, F, C, H, W = tokens.shape
        # assert F == self.num_frames
        tokens = tokens[:, start_frame:mid_frame]
        if pose_tokens is not None:
            pose_tokens = pose_tokens[:, start_frame:mid_frame]
        bs, F, C, H, W = tokens.shape
        tokens = rearrange(tokens, 'b f c h w -> b f h w c')
        guide = rearrange(guide, 'b f c h w -> b f h w c')
        if self.learnable_queries:
            queries = self.queries.weight.reshape(1, 1, H, W, C).expand(bs, F, H, W, C)
        else:
            queries = tokens
        queries = queries + self.temporal_embeddings.weight[None, self.offset:F+self.offset, None, None, :].expand(
            bs, -1, H, W, -1)
        tokens = tokens + self.temporal_embeddings.weight[None, :F, None, None, :].expand(
            bs, -1, H, W, -1)
        #
        guide = guide + self.temporal_embeddings.weight[None, self.offset:F+self.offset, None, None, :].expand(
            bs, -1, H, W, -1)
        tokens = torch.cat([tokens, guide], dim=1) # b 2f h w c
        #
        encoder_outs_pose_tokens = [None for _ in range(len(self.pose_attn_en))]
        encoder_outs_pose_queries = [None for _ in range(len(self.pose_attn_en))]
        if pose_tokens is not None:
            if self.learnable_queries:
                pose_queries = self.pose_queries.weight.reshape(1, 1, C).expand(bs, F, C)
            else:
                pose_queries = pose_tokens
            pose_queries = pose_queries + self.pose_temporal_embeddings.weight[None, self.offset:F+self.offset, :].expand(
                bs, -1, -1)
            pose_tokens = pose_tokens + self.pose_temporal_embeddings.weight[None, :F, :].expand(
                bs, -1, -1)
            encoder_outs_pose_tokens = []
            encoder_outs_pose_queries = []
        
        encoder_outs_tokens = []
        encoder_outs_queries = []
        
        for temporal_attn, encoder, down, pose_attn_en, pose_en in zip(self.temporal_attentions_en, self.encoders, self.downsamples, self.pose_attn_en, self.pose_en):
            # b, f, h, w, c = tokens.shape
            b, f, h, w, c = queries.shape
            
            if pose_tokens is not None:
                for pose_temporal_attn, pose_temporal_norm, spatial_attn, spatial_norm, ffn, ffn_norm in pose_attn_en:
                    pose_queries = pose_queries + pose_temporal_attn(pose_queries, pose_tokens, pose_tokens, need_weights=False, attn_mask=self.attn_mask[:f,:f])[0]
                    pose_queries = pose_temporal_norm(pose_queries)
                    #b, f, h, w, c = queries.shape
                    pose_queries = rearrange(pose_queries, 'b f c -> (b f) 1 c')
                    queries = rearrange(queries, 'b f h w c -> (b f) (h w) c')
                    pose_queries = pose_queries + spatial_attn(pose_queries, queries, queries, need_weights=False, attn_mask=None)[0]
                    pose_queries = spatial_norm(pose_queries)
                    
                    pose_queries = pose_queries + ffn(pose_queries)
                    pose_queries = ffn_norm(pose_queries)
                    pose_queries = rearrange(pose_queries, '(b f) 1 c -> b f c', b=b, f=f)
                    queries = rearrange(queries, '(b f) (h w) c -> b f h w c', b=b, f=f, h=h, w=w)
                
                pose_queries = pose_en(pose_queries)
                pose_tokens = pose_en(pose_tokens)
                encoder_outs_pose_queries.append(pose_queries)
                encoder_outs_pose_tokens.append(pose_tokens)
            
            queries = rearrange(queries, 'b f h w c -> (b h w) f c')
            tokens = rearrange(tokens, 'b f h w c -> (b h w) f c')
            #queries = rearrange(queries, 'b f h w c -> (b h w) f c')
            for cross_attn, cross_norm, ffn, ffn_norm in temporal_attn:
                # queries = queries + cross_attn(queries, tokens, tokens, need_weights=False, attn_mask=self.attn_mask[:f, :f])[0]
                queries = queries + cross_attn(queries, tokens, tokens, need_weights=False, attn_mask=torch.cat([self.attn_mask[:f, :f], self.attn_mask[:f, :f]], dim=1))[0]
                queries = cross_norm(queries)
                
                queries = queries + ffn(queries)
                queries = ffn_norm(queries)
            
            queries = rearrange(queries, '(b h w) f c -> (b f) c h w', b=b, h=h, w=w)
            tokens = rearrange(tokens, '(b h w) f c -> (b f) c h w', b=b, h=h, w=w)
            queries = encoder(queries)
            tokens = encoder(tokens)
            encoder_outs_tokens.append(tokens)
            encoder_outs_queries.append(queries)
            queries = down(queries)
            tokens = down(tokens)
            queries = rearrange(queries, '(b f) c h w -> b f h w c', b=b, f=f)
            # tokens = rearrange(tokens, '(b f) c h w -> b f h w c', b=b, f=f) 
            tokens = rearrange(tokens, '(b f) c h w -> b f h w c', b=b, f=2*f) 
        b, f, h, w, c = queries.shape
        queries = rearrange(queries, 'b f h w c -> (b f) c h w')
        tokens = rearrange(tokens, 'b f h w c -> (b f) c h w')
        queries = self.mid(queries)
        tokens = self.mid(tokens)
        
        if pose_tokens is not None:
            pose_queries = self.pose_mid(pose_queries)
            pose_tokens = self.pose_mid(pose_tokens)
        
        # print("checkpoint!!!")

        # queries = rearrange(queries, '(b f) c h w -> b f h w c', b=b, f=f)
        # tokens = rearrange(tokens, '(b f) c h w -> b f h w c', b=b, f=f)
        for temporal_attn, decoder, up, encoder_out_queries, encoder_out_tokens, pose_attn_de, pose_de_, encoder_out_pose_queries, encoder_out_pose_tokens, pose_up in zip(self.temporal_attentions_de,
                                                                        self.decoders, self.upsamples, encoder_outs_queries[::-1],
                                                                        encoder_outs_tokens[::-1], self.pose_attn_de, self.pose_de,
                                                                        encoder_outs_pose_queries[::-1], encoder_outs_pose_tokens[::-1], self.pose_up):
            queries = up(queries)
            tokens = up(tokens)
            
            pad_x_queries = encoder_out_queries.shape[2] - queries.shape[2]
            pad_y_queries = encoder_out_queries.shape[3] - queries.shape[3]
            queries = nn.functional.pad(queries, (pad_x_queries//2, pad_x_queries-pad_x_queries//2,
                                      pad_y_queries//2, pad_y_queries-pad_y_queries//2))
            pad_x_tokens = encoder_out_tokens.shape[2] - tokens.shape[2]
            pad_y_tokens = encoder_out_tokens.shape[3] - tokens.shape[3]
            tokens = nn.functional.pad(tokens, (pad_x_tokens//2, pad_x_tokens-pad_x_tokens//2,
                                      pad_y_tokens//2, pad_y_tokens-pad_y_tokens//2))
            queries = torch.cat([queries, encoder_out_queries], dim=1)
            tokens = torch.cat([tokens, encoder_out_tokens], dim=1)
            c, h, w = queries.shape[-3:]
            queries = rearrange(queries, '(b f) c h w -> (b h w) f c', b=b, f=f)
            # tokens = rearrange(tokens, '(b f) c h w -> (b h w) f c', b=b, f=f)
            tokens = rearrange(tokens, '(b f) c h w -> (b h w) f c', b=b, f=2*f)
            for cross_attn, cross_norm, ffn, ffn_norm in temporal_attn:
                # queries = queries + cross_attn(queries, tokens, tokens, need_weights=False, attn_mask=self.attn_mask[:f,:f])[0]
                queries = queries + cross_attn(queries, tokens, tokens, need_weights=False, attn_mask=torch.cat([self.attn_mask[:f,:f], self.attn_mask[:f,:f]], dim=1))[0]
                queries = cross_norm(queries)
                
                queries = queries + ffn(queries)
                queries = ffn_norm(queries)
            queries = rearrange(queries, '(b h w) f c -> (b f) c h w', b=b, h=h, w=w)
            tokens = rearrange(tokens, '(b h w) f c -> (b f) c h w', b=b, h=h, w=w)
            
            
            if pose_tokens is not None:
                pose_queries = pose_up(pose_queries)
                pose_tokens = pose_up(pose_tokens)
                pose_queries = torch.cat([pose_queries, encoder_out_pose_queries], dim=2)
                pose_tokens = torch.cat([pose_tokens, encoder_out_pose_tokens], dim=2)
                
                for pose_temporal_attn, pose_temporal_norm, spatial_attn, spatial_norm, ffn, ffn_norm in pose_attn_de:
                    pose_queries = pose_queries + pose_temporal_attn(pose_queries, pose_tokens, pose_tokens, need_weights=False, attn_mask=self.attn_mask[:f, :f])[0]
                    pose_queries = pose_temporal_norm(pose_queries)
                    #b, f, h, w, c = queries.shape
                    pose_queries = rearrange(pose_queries, 'b f c -> (b f) 1 c')
                    #queries = rearrange(queries, 'b f h w c -> (b f) (h w) c')
                    queries = rearrange(queries, '(b f) c h w -> (b f) (h w) c', b=b, f=f, h=h, w=w)
                    pose_queries = pose_queries + spatial_attn(pose_queries, queries, queries, need_weights=False, attn_mask=None)[0]
                    pose_queries = spatial_norm(pose_queries)
                    
                    pose_queries = pose_queries + ffn(pose_queries)
                    pose_queries = ffn_norm(pose_queries)
                    queries = rearrange(queries, '(b f) (h w) c -> (b f) c h w', b=b, f=f, h=h, w=w)
                    pose_queries = rearrange(pose_queries, '(b f) 1 c -> b f c', b=b, f=f)
                pose_queries = pose_de_(pose_queries)
                pose_tokens = pose_de_(pose_tokens)

            queries = decoder(queries)
            tokens = decoder(tokens)
        
        queries = self.conv_out(queries)
        queries = rearrange(queries, '(b f) c h w -> b f c h w', b=b, f=f)
        if pose_tokens is not None:
            pose_queries = self.pose_out(pose_queries)
            return queries, pose_queries

        return queries


class UnetBlock(nn.Module):
    def __init__(self, shape, in_c, out_c, residual=True):
        super().__init__()
        self.ln = nn.LayerNorm(shape)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.act = nn.ReLU()
        self.residual = residual
        if residual:
            if in_c == out_c:
                self.shortcut = nn.Identity()
            if in_c != out_c:
                self.shortcut = nn.Conv2d(in_c, out_c, 1, 1, 0)
    def forward(self, input):
        output = self.ln(input)
        output = self.conv1(output)
        output = self.act(output)
        output = self.conv2(output)
        if self.residual:
            output = output + self.shortcut(input)
        output = self.act(output)
        return output
    