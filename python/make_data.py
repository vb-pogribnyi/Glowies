import torch
import numpy as np

data = np.array([
	[-1., -.9, -.8, -.7, -.6],
	[-.5, -.4, -.3, -.2, -.1],
	[.1, .2, .3, .4, .5],
	[.6, .7, .8, .9, 1.]
])
filter = np.array([
	[0.1, 0.2, 0.3],
	[-0.2, 0.2, -0.3]
])
bias = np.array([0.1])

model = torch.nn.Sequential(
	torch.nn.Conv2d(1, 1, (2, 3))
)
model[0].weight.data = torch.tensor(filter).float().reshape(model[0].weight.data.shape)
model[0].bias.data = torch.tensor(bias).float().reshape(model[0].bias.data.shape)

out = model(torch.tensor(data).float().unsqueeze(0).unsqueeze(0))
out = out.detach().double().numpy()[0][0]

print(out)
'''
[[-0.31000003 -0.28       -0.25      ]
 [-0.19000001 -0.16000001 -0.13000001]
 [ 0.01999998  0.05000002  0.07999998]]
'''

np.save('data.npy', data)
np.save('filter_weights.npy', filter)
np.save('filter_bias.npy', filter)
np.save('output.npy', out)

print('done')

