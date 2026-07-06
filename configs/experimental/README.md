# Experimental configs (NOT used in the paper)

These configs enable **conv-layer quantization** (`quantize_conv: true`), which is **not** part of
the Recti-Q paper. The paper's stated W4 method is **Linear-only** `Int4WeightOnly` (HQQ), which
leaves `nn.Conv2d` weights in full precision. They are kept here only for exploration and are
excluded from the released results.

For paper-faithful ResNet50 runs use the Linear-only configs in the parent `configs/` folder:
`imagenet_c_resnet_linear_all.yaml`, `pacs_resnet_linear_all.yaml`.
</content>
