# optimizer
optimizer = dict(type='AdamW', lr=0.0002, weight_decay=0.0001)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=0.0, by_epoch=False)
# runtime settings

# gdcld
runner = dict(type='IterBasedRunner', max_iters=300000)  # 表示训练的最大迭代次数为 10000。
checkpoint_config = dict(by_epoch=False, interval=2000) # 表示每 2000 次迭代保存一个检查点。
evaluation = dict(interval=1000, metric='mIoU') #表示每 200 次迭代进行一次评估。

# luding
# runner = dict(type='IterBasedRunner', max_iters=35000)  # 表示训练的最大迭代次数为 10000。
# checkpoint_config = dict(by_epoch=False, interval=200) # 表示每 200 次迭代保存一个检查点。
# evaluation = dict(interval=10000, metric='mIoU') #表示每 10000 次迭代进行一次评估。