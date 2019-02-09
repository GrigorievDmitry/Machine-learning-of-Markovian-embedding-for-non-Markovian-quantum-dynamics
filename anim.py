import imageio

images = []
filenames = [f'pic/dynamics_fig{i}.png' for i in range(300)]
for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave('anim.gif', images)