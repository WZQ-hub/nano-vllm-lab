import torch



class GPTQLinearMethod:

    def __init__(self, group_size = 128):
        self.group_size = group_size

    def weight_loader(
        self,
        layer,
        param,
        loaded_weight,
        shard_id=None,
    ):
        # weight_shape 是元数据，不参与计算
        if param is layer.weight_shape:
            return

        if shard_id is None:
            if tuple(param.shape) != tuple(loaded_weight.shape):
                raise ValueError(
                    f"shape mismatch: "
                    f"{tuple(param.shape)} vs "
                    f"{tuple(loaded_weight.shape)}"
                )
            param.data.copy_(loaded_weight)
            return

        shard_offset, shard_size = layer.get_quant_shard(shard_id)

        if param is layer.weight_packed:
            param.data[
                shard_offset:shard_offset + shard_size,
                :
            ].copy_(loaded_weight)

        elif param is layer.weight_scale:
            param.data[
                shard_offset:shard_offset + shard_size,
                :
            ].copy_(loaded_weight)

        else:
            raise ValueError("unknown quantized parameter")


    
    # W4A16
    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size: int,
        output_size: int,
        params_dtype,
    ):
        layer.weight_packed = torch.nn.Parameter(
            torch.empty(
                output_size,
                input_size // 8,
                dtype=torch.int32,
            ),
            requires_grad=False
        )

        layer.weight_scale = torch.nn.Parameter(
            torch.empty(
                output_size,
                input_size // self.group_size,
                dtype=params_dtype
            ),
            requires_grad=False
        )

        layer.weight_shape = torch.nn.Parameter(
            torch.empty(
                2,
                dtype=torch.int64
            ),
            requires_grad=False,
        )

        layer.weight_packed.weight_loader = (
            lambda param, loaded_weight, shard_id=None:
                self.weight_loader(
                    layer,
                    param,
                    loaded_weight,
                    shard_id=shard_id
                )
        )
        layer.weight_scale.weight_loader = (
            lambda param, loaded_weight, shard_id=None:
                self.weight_loader(
                    layer,
                    param,
                    loaded_weight,
                    shard_id=shard_id
                )
        )
        layer.weight_shape.weight_loader = self.weight_loader

    def unpack_int4(self, weight_packed):
        packed = weight_packed.to(torch.int64) & 0xffffffff
        shifts = torch.arange(
            8,
            device=packed.device,
            dtype=torch.int64,
        ) * 4

        return ((packed.unsqueeze(-1) >> shifts) & 0xf).reshape(weight_packed.shape[0], -1)


    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor, 
        bias: torch.Tensor = None,
    ):
        # [N, K]
        qweight = self.unpack_int4(layer.weight_packed)

        # [N, K // group_size]
        scales = layer.weight_scale

        n, k = qweight.shape
        qweight = qweight.reshape(n, k // self.group_size, self.group_size)
        scales = scales.unsqueeze(-1)

        weight = (qweight.to(scales.dtype) - 8) * scales

        # [N, K]
        weight = weight.reshape(n, k)

        output = torch.nn.functional.linear(x, weight.to(x.dtype), bias)
        return output
