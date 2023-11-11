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

model = torch.nn.Sequential(
	torch.nn.Conv2d(1, 1, (2, 3))
)
model[0].weight.data = torch.tensor(filter).float().reshape(model[0].weight.data.shape)
model[0].bias.data *= 0

out = model(torch.tensor(data).float().unsqueeze(0).unsqueeze(0))
out = out.detach().double().numpy()[0][0]

print(out)
'''
[[-0.41000006 -0.38       -0.34999999]
 [-0.29000002 -0.26000002 -0.23000002]
 [-0.08000001 -0.04999997 -0.02000001]]
'''

np.save('data.npy', data)
np.save('weights.npy', filter)
np.save('output.npy', out)

print('done')

