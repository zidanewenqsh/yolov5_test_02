# compute_loss

```flow
st=>start: start
op1=>operation: build_target()->tcls, tbox, indices, anchors
op2=>operation: BCE_cls, BCE_obj
op3=>operation: smooth_BCE
op4=>operation: Focal Loss
op5=>operation: 遍历 output
sub1=>subroutine: 取索引, tobj, n
cond0=>condition: 循环
ops1=>operation: 按索引取prediction
ops2=>operation: pxy, pwh, pbox
cond1=>condition: n
ops3=>operation: 按索引取prediction
ops4=>operation: iou
ops5=>operation: lbox
ops6=>operation: tobj
cond2=>condition: 多分类
ops7=>operation: onehot
ops8=>operation: lcls
ops9=>operation: lobj
op6=>operation: loss
e=>end: 结束框
st->op1->op2->op3->op4->op5(right)->cond0
cond0(yes)->cond1
cond1(yes)->ops1->ops2->ops3->ops4->ops5->ops6->cond2
cond2(yes)->ops7->ops8(right)->ops9
cond2(no)->ops9
cond1(no)->ops9->cond0
cond0(no)->op6->e
```

# build_target

```flow
st=>start: start
op_def=>operation: 定义变量
cond_nl=>condition: number of anchor groups
op_mt=>operation: Match targets to anchors, 标签乘anchor
cond_nt=>condition: number of targets
op_m=>operation: Matches, 取边与建议框大小相比最大值小于阈值者
op_j=>operation: 按索引取
op_jklm=>operation: j,k,l,m
op_o=>operation: Offsets,stack 1,j,k,l,m
op_j2=>operation: 再次按索引取
op_o2=>operation: offset
op_zero=>operation: offset=0
op_define=>operation: Define
op_append=>operation: Append
e=>end: 结束框
st->op_def->cond_nl
cond_nl(yes)->op_mt(right)->cond_nt
cond_nt(yes)->op_m->op_m->op_j->op_jklm->op_o->op_j2->op_o2->op_define
cond_nt(no)->op_zero->op_define
op_define->op_append(right)->cond_nl
cond_nl(no)->e
```





